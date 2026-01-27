import traceback
import sys
import os
import torch
import yaml
import pandas as pd
import numpy as np

# === 【魔法代码】将当前目录加入 sys.path，解决 import 报错 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# ==========================================================

from tsnkit import core as utils

# 引入你的算法模块 (确保上面sys.path生效后，这些可以直接import)
# 如果报错，请检查文件夹名字是否对应 (你提到是 phase2_scheduling)
try:
    from phase1_partitioning.gnn_model import CorrelationModel # 示例，根据你实际类名修改
    from phase2_scheduling.scheduler_agent import PPOAgent
    from common.ksp import KSPCalculator # 假设你需要把 tsnkit stream 转成这个
except ImportError as e:
    print(f"[Import Error] {e}")
    print("请检查文件夹命名是否为 phase1_partitioning, phase2_scheduling, common")
    # 暂时容错，防止一开始跑不通
    pass

class SCADRL_Solver:
    def __init__(self, workers=1) -> None:
        self.workers = workers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. 加载 Config
        self.config = self._load_configs()
        
        # 2. 初始化 Agent (这里只是占位，具体要在 init 中根据网络规模可能重新调整)
        # self.agent = SchedulerAgent(self.config, device=self.device)

    def _load_configs(self):
        """加载你搬过来的 yaml 配置"""
        configs = {}
        # 假设你搬过来的 configs 文件夹里有这些文件
        files = ['base_config.yaml', 'phase1_config.yaml', 'phase2_config.yaml']
        for f_name in files:
            p = os.path.join(current_dir, 'configs', f_name)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    # 简单合并所有配置
                    configs.update(yaml.safe_load(f))
            else:
                print(f"[Warning] Config {f_name} not found at {p}")
        return configs

    def init(self, task_path: str, net_path: str) -> None:
        print(f"[SCA-DRL] Loading task: {task_path}, net: {net_path}")
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        
        # TODO: 数据转换
        # TSNKit 的 self.task (StreamSet) -> SCA-DRL 需要的格式
        # TSNKit 的 self.net (Network) -> SCA-DRL 需要的格式 (NetworkX graph)
        self.sca_graph = self._convert_net_to_nx(self.net)
        self.sca_flows = self._convert_stream_to_flows(self.task)

    def _convert_net_to_nx(self, net):
        """将 TSNKit Network 转为 NetworkX (你的算法可能需要这个)"""
        import networkx as nx
        G = nx.DiGraph()
        # 简单转换示例
        for l in net.links:
            G.add_edge(l.src, l.dst, bandwidth=l.rate) 
        return G

    def _convert_stream_to_flows(self, task):
        """将 TSNKit StreamSet 转为你的 flow 列表"""
        flows = []
        for s in task:
            flows.append({
                'id': int(s),
                'src': s.src,
                'dst': s.dst,
                'size': s.size,
                'period': s.period,
                'deadline': s.deadline
            })
        return flows

    def prepare(self) -> None:
        """
        这里运行 Phase 1: Partitioning (流分组)
        """
        print("[SCA-DRL] Phase 1: Running Correlation-Aware Partitioning...")
        # 模拟调用你的 Phase 1 逻辑
        # self.groups = self.phase1_model(self.sca_graph, self.sca_flows)
        self.groups = [self.sca_flows] # 临时：不做分组，全部当作一组
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()
        
        print(f"[SCA-DRL] Phase 2: Running DRL Scheduling on {len(self.groups)} groups...")
        
        # 加载预训练模型
        model_path = os.path.join(current_dir, 'models', 'ppo_scheduler_final.pth') # 假设你放这了
        if os.path.exists(model_path):
            print(f"[SCA-DRL] Loading model from {model_path}")
            # self.agent.load(model_path)
        
        # === 推理循环 ===
        # for group in self.groups:
        #     state = self.env.reset(group)
        #     action = self.agent.select_action(state)
        #     ...
        
        # ⚠️ 临时借用 ls 算法生成一个可行解，确保 adapter 能跑通流程 ⚠️
        # 等你的 Agent 调试好了，把下面这段换成你的 Agent 推理结果
        from tsnkit.algorithms.ls import ls
        print("[SCA-DRL] (Mocking) Using LS logic to generate placeholder result...")
        ls_solver = ls(self.workers)
        
        # 手动注入数据
        ls_solver.task = self.task
        ls_solver.net = self.net
        # ls_solver.init("", "") # Hack init
        ls_solver.prepare()
        stat = ls_solver.solve()
        
        # 偷梁换柱：把 ls 算出来的结果存到自己这里，方便 output() 调用
        self._mock_ls_solver = ls_solver 
        
        end_time = utils.time_log()
        return utils.Statistics("SCA-DRL", utils.Result.schedulable, end_time - start_time)

    def output(self) -> utils.Config:
        # 如果是用自己的 Agent，这里需要把 Agent 的 GCL 转为 utils.Config
        # 这里暂时返回临时借用的结果
        if hasattr(self, '_mock_ls_solver'):
            return self._mock_ls_solver.output()
        return utils.Config()

# 命令行入口
def benchmark(name, task_path, net_path, output_path="./", workers=1) -> utils.Statistics:
    stat = utils.Statistics(name)
    try:
        test = SCADRL_Solver(workers)
        test.init(task_path, net_path)
        test.prepare()
        stat = test.solve()
        if stat.result == utils.Result.schedulable:
            test.output().to_csv(name, output_path)
        stat.content(name=name)
        return stat
    except Exception as e:
        print("[!]", e, flush=True)
        traceback.print_exc()
        stat.result = utils.Result.error
        stat.content(name=name)
        return stat

if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)