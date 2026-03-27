"""
Phase 2 环境模块：基于 TSNKit ls.py 实现的物理调度引擎与 Gym 接口封装。
"""

import os
import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tsnkit import core as utils
from tsnkit.algorithms.ls import ls  # tsnkit底层调度核心类

from sca_drl.common.utils import resolve_path


class DRL_PhysicsEngine(ls):
    """
    继承 tsnkit ls，实现基于时隙的流调度物理引擎。

    主要职责：
    - 封装流时隙分配逻辑，调用基类冲突检测接口
    - 管理调度状态账本和链路利用率
    - 与环境动作接口匹配
    """

    def __init__(self, topo, streams, workers=1):
        self.topo = topo
        self.streams = streams
        super().__init__(workers)  # 调用基类构造

        # 预计算每个流的候选路径
        self.task_routes = {s: self.topo.get_all_path(s.src, s.dst) for s in streams}

    # 初始化状态
    def deep_clear(self):
        """
        彻底清空底层物理世界的残留物，用于 Episode 重置
        """
        self._delay = {}
        # 初始化空白的全局 GCL (基于连续区间)
        self._result = {l: [] for l in self.topo.links}
        self._paths = {}
        self._offset = {}

    def try_allocate_agent_action(self, stream, path):
        """
        尝试分配给定流与路径。

        Args:
            stream: tsnkit 流对象
            path: 选定路径（Path对象）

        Returns:
            (bool success, float max_utilization)
        """
        inject_time = self.find_inject_time(stream, path)
        if inject_time == -1:
            # 无法分配
            return False, 1.0

        # 调用基类写入状态账本
        self._delay[stream] = self.get_nw_delay(stream, path)
        self._paths[stream] = path
        self._offset[stream] = inject_time

        prev_end = inject_time
        max_util = 0.0

        for link in path.links:
            start = prev_end
            end = start + stream.get_t_trans(link)
            prev_end = start + link.t_proc + stream.get_t_trans(link)

            # 链路时隙分配进行占用登记（周期内重复所有时隙）
            for k in stream.get_frame_indexes(self.lcm):
                slot_range = (start + k * stream.period, end + k * stream.period, 0)
                self._result[link].append(slot_range)

            # 底层变量更新（为 tsnkit 检测时序）
            self._result[link].sort(key=lambda x: x[0])

            # 计算最大利用率
            max_util = max(max_util, self._get_utilization(link))

        return True, max_util

    def _get_utilization(self, link):
        """
        计算给定链路的利用率，具体实现依赖基类状态。
        """
        # 可调用基类接口或自定义统计
        if link in self._result:
            intervals = self._result[link]
            busy_slots = sum(end - start for start, end, _ in intervals)
            return busy_slots / self.lcm
        else:
            return 0.0

    def reset(self):
        """
        清理状态，准备新 episode。
        """
        self.deep_clear()


class TSNSchedulingEnv(gym.Env):
    """
    TSN 联合路由调度环境，gym接口封装。

    依赖：
    - DRL_PhysicsEngine 做底层状态管理与动作执行
    - 支持 Phase 1 流 embedding 输入
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config: dict, phase1_embeddings: dict = None):
        super().__init__()

        env_cfg = config["environment"]
        data_cfg = config["data"]

        # 加载拓扑与流
        topo_path = resolve_path(data_cfg["topo_file"])
        task_path = resolve_path(data_cfg["task_file"])
        self.topo = utils.load_network(topo_path)
        self.tasks = utils.load_stream(task_path)
        self.streams = self.tasks.streams

        # 建立物理引擎
        self.engine = DRL_PhysicsEngine(self.topo, self.streams)

        self.k_max = env_cfg["k_max"]
        self.max_steps = env_cfg.get("max_steps_per_episode", 200)

        # Phase 1 embedding 维度及映射
        self.phase1_embeddings = phase1_embeddings
        if phase1_embeddings is not None and len(phase1_embeddings) > 0:
            self.emb_dim = len(next(iter(phase1_embeddings.values())))
        else:
            self.emb_dim = 0

        # 状态空间维度
        self.d_base = 5  # size, period, deadline, is_scheduled, priority 等
        self.d_route = 3 * self.k_max  # 路径特征（示例表示）
        self.d_feature = self.d_base + self.d_route + self.emb_dim

        # 观测空间：每条流一条状态向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.streams), self.d_feature),
            dtype=np.float32,
        )
        # 动作空 间：流索引 × 路径索引
        self.action_space = spaces.MultiDiscrete([len(self.streams), self.k_max])

        # 追踪变量
        self.scheduled = np.zeros(len(self.streams), dtype=bool)
        self.current_step = 0

    def reset(self):
        self.engine.reset()
        self.scheduled[:] = False
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        stream_idx, path_idx = action
        self.current_step += 1

        if self.scheduled[stream_idx]:
            reward = -1.0  # 重复调度惩罚
            done = False
        else:
            stream = self.streams[stream_idx]
            candidate_paths = self.engine.task_routes[stream]
            if path_idx >= len(candidate_paths):
                reward = -1.0  # 超出路径数惩罚
                success = False
            else:
                path = candidate_paths[path_idx]
                success, _ = self.engine.try_allocate_agent_action(stream, path)
                reward = 10.0 if success else -5.0
                if success:
                    self.scheduled[stream_idx] = True

            done = self.current_step >= self.max_steps or self.scheduled.all()

        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self):
        """
        构造观测状态，拼接流基本特征、路径特征和 Phase1 embedding。
        """
        obs = np.zeros((len(self.streams), self.d_feature), dtype=np.float32)

        for i, s in enumerate(self.streams):
            base_feats = [
                s.size / 1500.0,
                np.log(s.period + 1) / 10.0,
                np.log(s.deadline + 1) / 10.0,
                float(self.scheduled[i]),
                getattr(s, "priority", 0) / 7.0,
            ]
            obs[i, :self.d_base] = base_feats

            # 路径特征示例，具体情况视task_routes而定
            candidate_paths = self.engine.task_routes[s]
            for k in range(self.k_max):
                offset = self.d_base + 3 * k
                if k < len(candidate_paths):
                    path = candidate_paths[k]
                    obs[i, offset] = len(path) / 10.0  # hop count
                    obs[i, offset + 1] = 1.0           # 可用标志
                    obs[i, offset + 2] = 0.0           # 拥塞占位符
                else:
                    obs[i, offset:offset+3] = 0.0

            # Phase 1 embedding拼接
            if self.phase1_embeddings:
                sid = getattr(s, "name", None) or getattr(s, "_id", None)
                emb = self.phase1_embeddings.get(sid, np.zeros(self.emb_dim, dtype=np.float32))
                obs[i, -self.emb_dim:] = emb

        return obs

    def render(self, mode="human"):
        pass  # 可根据需要实现可视化
