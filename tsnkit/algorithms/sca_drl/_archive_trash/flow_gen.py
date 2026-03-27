import random
import numpy as np
from common.ksp import KSPCalculator

class FlowGenerator:
    def __init__(self, config, G):
        self.cfg = config
        self.G = G
        self.nodes = list(G.nodes())
        self.ksp_solver = KSPCalculator(G)

    def generate_flows(self):
        """
        生成流集合 F。
        返回: List of dictionaries
        """
        flows = []
        num_flows = self.cfg['traffic']['num_flows']
        k = self.cfg['ksp']['k']
        periods = self.cfg['traffic']['period_list']

        print(f"Generating {num_flows} flows with K={k} paths...")

        count = 0
        while count < num_flows:
            # 1. 随机源宿
            src, dst = random.sample(self.nodes, 2)
            
            # 2. 计算 KSP (如果不可达，则重选源宿)
            paths = self.ksp_solver.compute_k_shortest_paths(src, dst, k)
            if not paths:
                continue

            # 3. 随机物理属性
            period = random.choice(periods)
            size = random.randint(*self.cfg['traffic']['size_range'])
            deadline = period  # 简化设定，Deadline等于周期 (Hard Real-time)

            # 4. 构建流对象 (Dict)
            flow = {
                'flow_id': count,
                'src': src,
                'dst': dst,
                'period': period,
                'size': size,
                'deadline': deadline,
                'k_paths': paths  # 这是一个列表，包含 K 条路径
            }
            flows.append(flow)
            count += 1
            
        return flows