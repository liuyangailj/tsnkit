"""
JRS-NW 算法的 CBC（开源 MILP）重实现。

逻辑与 tsnkit/algorithms/jrs_nw.py 完全一致，仅将 gurobipy 替换为
python-mip + CBC，消除商业 license 限制（CBC 无变量/约束数上限）。

变量布局
--------
route[s][l] ∈ {0,1}   : 流 s 是否经过链路 l
start[s][l] ∈ ℤ≥0     : 流 s 在链路 l 上的发送开始时隙
end  [s][l] ∈ ℤ≥0     : 流 s 在链路 l 上的发送结束时隙

s/l 均继承自 int，可直接用作下标。

状态映射
--------
OPTIMAL / FEASIBLE      → schedulable   (找到可行调度)
INFEASIBLE / INT_INFEAS → unschedulable
其他（NO_SOLUTION_FOUND 等）→ unknown
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import time
import warnings
from typing import Set

import mip

from tsnkit import core as utils

T_M = utils.T_M   # big-M 常量，与 jrs_nw.py 完全一致


class jrs_nw_cbc:
    def __init__(self, workers: int = 1) -> None:
        self.workers = workers

    # ─────────────────────────────────────────────────────────────────────
    # 1. 初始化：加载数据 + 建模型 + 创建变量
    # ─────────────────────────────────────────────────────────────────────

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net  = utils.load_network(net_path)

        self.solver = mip.Model(solver_name=mip.CBC)
        self.solver.verbose  = 0           # 关闭控制台输出
        self.solver.threads  = self.workers

        N = len(self.task)
        L = self.net.num_l

        # 二维变量列表，下标 [stream_int][link_int]
        self.route = [
            [self.solver.add_var(var_type=mip.BINARY,  name=f"r_{i}_{j}") for j in range(L)]
            for i in range(N)
        ]
        self.start = [
            [self.solver.add_var(var_type=mip.INTEGER, lb=0, name=f"s_{i}_{j}") for j in range(L)]
            for i in range(N)
        ]
        self.end = [
            [self.solver.add_var(var_type=mip.INTEGER, lb=0, name=f"e_{i}_{j}") for j in range(L)]
            for i in range(N)
        ]

    # ─────────────────────────────────────────────────────────────────────
    # 2. 约束构建（逻辑与 jrs_nw.py 逐一对应）
    # ─────────────────────────────────────────────────────────────────────

    def prepare(self) -> None:
        self.routing_space = {s: self.get_route_space(s) for s in self.task}
        self.add_route_const()
        self.add_frame_const()
        self.add_flow_trans_const()
        self.add_link_const()
        self.add_delay_const()

    def get_route_space(self, task: utils.Stream) -> Set[utils.Link]:
        _paths = self.net.get_all_path(task.src, task.dst)
        return set(x for y in _paths for x in y.iter_link())

    def add_route_const(self) -> None:
        for s in self.task:
            rs = self.routing_space[s]

            # src 入度 = 0
            inc_src = [self.route[s][l] for l in self.net.get_income_links(s.src) if l in rs]
            if inc_src:
                self.solver += mip.xsum(inc_src) == 0

            # src 出度 = 1
            out_src = [self.route[s][l] for l in self.net.get_outcome_links(s.src) if l in rs]
            if out_src:
                self.solver += mip.xsum(out_src) == 1

            # dst 入度 = 1
            inc_dst = [self.route[s][l] for l in self.net.get_income_links(s.dst) if l in rs]
            if inc_dst:
                self.solver += mip.xsum(inc_dst) == 1

            # dst 出度 = 0
            out_dst = [self.route[s][l] for l in self.net.get_outcome_links(s.dst) if l in rs]
            if out_dst:
                self.solver += mip.xsum(out_dst) == 0

            # 其他端系统：出度 = 0（不转发他人的流）
            for v in self.net.e_nodes:
                if v == s.src:
                    continue
                out_v = [self.route[s][l] for l in self.net.get_outcome_links(v) if l in rs]
                if out_v:
                    self.solver += mip.xsum(out_v) == 0

            # 交换机：入度 = 出度（流守恒），且出度 ≤ 1（单播）
            for v in self.net.s_nodes:
                inc_v = [self.route[s][l] for l in self.net.get_income_links(v)  if l in rs]
                out_v = [self.route[s][l] for l in self.net.get_outcome_links(v) if l in rs]
                if inc_v or out_v:
                    self.solver += mip.xsum(inc_v) == mip.xsum(out_v)
                if out_v:
                    self.solver += mip.xsum(out_v) <= 1

    def add_frame_const(self) -> None:
        for s in self.task:
            for e in self.routing_space[s]:
                # end ≤ period × route  （未选路时强制 end=0）
                self.solver += self.end[s][e] <= s.period * self.route[s][e]
                # end = start + route × t_trans
                self.solver += self.end[s][e] == self.start[s][e] + self.route[s][e] * s.get_t_trans(e)

    def add_flow_trans_const(self) -> None:
        """No-Wait 约束：帧在交换机不缓存，入链路 end+t_proc = 出链路 start。"""
        for s in self.task:
            rs = self.routing_space[s]
            for v in self.net.s_nodes:
                inc_terms = [self.end[s][e] + self.route[s][e] * e.t_proc
                             for e in self.net.get_income_links(v)  if e in rs]
                out_terms = [self.start[s][e]
                             for e in self.net.get_outcome_links(v) if e in rs]
                if inc_terms or out_terms:
                    self.solver += mip.xsum(inc_terms) == mip.xsum(out_terms)

    def add_link_const(self) -> None:
        """链路冲突约束：同一链路上任意两帧不重叠（大 M 互斥）。"""
        for s1, s2 in self.task.get_pairs():
            for l in self.net.links:
                if l not in self.routing_space[s1] or l not in self.routing_space[s2]:
                    continue
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    delta = self.solver.add_var(
                        var_type=mip.BINARY,
                        name=f"d_{l}_{s1}_{s2}_{k1}_{k2}",
                    )
                    r1, r2 = self.route[s1][l], self.route[s2][l]
                    e1, e2 = self.end[s1][l],   self.end[s2][l]
                    b1, b2 = self.start[s1][l], self.start[s2][l]

                    self.solver += (
                        e1 + k1 * s1.period
                        <= b2 + k2 * s2.period - 1 + (2 + delta - r1 - r2) * T_M
                    )
                    self.solver += (
                        e2 + k2 * s2.period
                        <= b1 + k1 * s1.period - 1 + (3 - delta - r1 - r2) * T_M
                    )

    def add_delay_const(self) -> None:
        for s in self.task:
            rs  = self.routing_space[s]
            t0  = mip.xsum(self.start[s][e] for e in self.net.get_outcome_links(s.src) if e in rs)
            t1  = mip.xsum(self.end[s][e]   for e in self.net.get_income_links(s.dst)  if e in rs)
            self.solver += t1 - t0 <= s.deadline

    # ─────────────────────────────────────────────────────────────────────
    # 3. 求解
    # ─────────────────────────────────────────────────────────────────────

    def solve(self, max_seconds: float = 600.0) -> dict:
        """运行 CBC 求解，返回统一结果字典。

        返回
        ----
        result      : utils.Result
        solve_time  : float (秒，CBC 实际耗时)
        sol_count   : int (找到的可行解数量)
        cbc_status  : str (OptimizationStatus 名称)
        """
        t0     = time.perf_counter()
        status = self.solver.optimize(max_seconds=max_seconds)
        solve_time = time.perf_counter() - t0

        sol_count = self.solver.num_solutions

        SCHEDULABLE = {
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        }
        UNSCHEDULABLE = {
            mip.OptimizationStatus.INFEASIBLE,
            mip.OptimizationStatus.INT_INFEASIBLE,
        }

        if status in SCHEDULABLE:
            result = utils.Result.schedulable
        elif status in UNSCHEDULABLE:
            result = utils.Result.unschedulable
        else:
            result = utils.Result.unknown

        return dict(
            result=result,
            solve_time=solve_time,
            sol_count=sol_count,
            cbc_status=status.name,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 4. 结果提取（逻辑与 jrs_nw.py 完全相同，.x 语义一致）
    # ─────────────────────────────────────────────────────────────────────

    def _is_active(self, var) -> bool:
        """CBC 浮点值可能为 0.9999...，用 >0.5 判断 binary=1。"""
        return (var.x or 0.0) > 0.5

    def output(self):
        config         = utils.Config()
        config.gcl     = self.get_gcl()
        config.release = self.get_offset()
        config.route   = self.get_route()
        config.queue   = self.get_queue()
        config._delay  = self.get_delay()
        return config

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in self.routing_space[s]:
                if not self._is_active(self.route[s][l]):
                    continue
                _start = round(self.start[s][l].x)
                _end   = round(self.end[s][l].x)
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append([l, 0, _start + k * s.period, _end + k * s.period, self.task.lcm])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            links = [l for l in self.net.get_outcome_links(s.src)
                     if l in self.routing_space[s] and self._is_active(self.route[s][l])]
            if not links:
                raise ValueError(f"No start link for stream {s}")
            if len(links) > 1:
                warnings.warn("Multiple start links detected.")
            offset.append([s, 0, round(self.start[s][links[0]].x)])
        return utils.Release(offset)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self._is_active(self.route[s][l]):
                    queue.append([s, 0, l, 0])
        return utils.Queue(queue)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self._is_active(self.route[s][l]):
                    route.append([s, l])
        return utils.Route(route)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            src_links = [l for l in self.net.get_outcome_links(s.src)
                         if l in self.routing_space[s] and self._is_active(self.route[s][l])]
            dst_links = [l for l in self.net.get_income_links(s.dst)
                         if l in self.routing_space[s] and self._is_active(self.route[s][l])]
            if not src_links or not dst_links:
                raise ValueError(f"Missing src/dst link for stream {s}")
            if len(src_links) > 1 or len(dst_links) > 1:
                warnings.warn("Multiple src/dst links detected.")
            _start = round(self.start[s][src_links[0]].x)
            _end   = round(self.end[s][dst_links[0]].x) - s.get_t_trans(dst_links[0])
            delay.append([s, 0, _end - _start])
        return utils.Delay(delay)
