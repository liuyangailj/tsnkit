import gymnasium as gym
from gym import spaces

import numpy as np
import networkx as nx
import copy

class SlotManager:
    def __init__(self, topology, time_limit, bandwidth):
        self.topology = topology
        self.time_limit = time_limit
        self.bandwidth = bandwidth
        # 资源表: {(u, v): numpy_array of shape (time_limit,)}
        # 这里简化处理：假设每个时隙要么被占(1)，要么空闲(0)。
        # 实际TSN中可能是剩余带宽值，这里为了简化先用 0/1 掩码，或者累加带宽
        self.link_resources = {}
        self.clear()

    def clear(self):
        """清空所有资源"""
        for u, v in self.topology.edges():
            # 初始化为 0 (表示已用带宽为0)
            self.link_resources[(u, v)] = np.zeros(self.time_limit, dtype=np.float32)

    def allocate(self, path, start_slot, size, period):
        """
        尝试分配资源
        path: 节点列表 [n1, n2, n3...]
        start_slot: 起始时隙
        size: 包大小 (这里简化为占用多少带宽/时间)
        period: 周期
        return: Boolean (是否成功)
        """
        # 简化的分配逻辑检查
        # 真实论文中需要检查整条路径在每个周期的时隙占用
        # 这里仅做演示：检查路径上每条边在 start_slot 是否有空
        
        # 1. Check
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if (u, v) not in self.link_resources:
                # 处理无向图/双向边情况
                if (v, u) in self.link_resources:
                    u, v = v, u
                else:
                    return False # 边不存在
            
            # 简单检查：假设占用 1 个单位时间
            # 实际需要循环 period
            current_slot = start_slot % self.time_limit
            if self.link_resources[(u, v)][current_slot] + size > self.bandwidth:
                return False

        # 2. Allocate
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if (u, v) not in self.link_resources and (v, u) in self.link_resources:
                u, v = v, u
            
            current_slot = start_slot % self.time_limit
            self.link_resources[(u, v)][current_slot] += size
            
        return True

    def get_utilization(self): # 此为目前替代全局资源状态的简易版临时替代方案
        """返回全网平均利用率 (作为 Global State 的一部分)"""
        total_util = []
        for res in self.link_resources.values():
            total_util.append(np.mean(res))
        return np.mean(total_util) if total_util else 0.0

    # [新增] 用于加载外部传入的资源状态
    def load_state(self, link_resources_snapshot):
        """
        加载之前的资源占用快照
        link_resources_snapshot: 字典 {(u, v): slot_array}
        """
        if link_resources_snapshot is None:
            self.clear()
            return
            
        # 深拷贝以防止引用修改
        self.link_resources = copy.deepcopy(link_resources_snapshot)

    def get_state_snapshot(self):
        """返回当前资源占用的深拷贝"""
        return copy.deepcopy(self.link_resources)


class TSNSchedulingEnv(gym.Env):
    def __init__(self, flows, topology, config, initial_resource_state=None, device='cpu'):
        """
        initial_resource_state: (可选) 上一个 Group 调度完后的资源快照
        """
        super(TSNSchedulingEnv, self).__init__()
        
        self.flows = flows
        self.topology = topology
        self.config = config
        self.device = device
        
        # [补回] 关键属性，Agent 初始化和 Mask 生成需要用到
        self.num_flows = len(flows)
        self.num_edges = len(topology.edges())
        
        # [新增] 保存初始资源状态
        self.initial_resource_state = initial_resource_state 
        
        # 定义动作空间: MultiDiscrete([Flow_ID, Path_ID])
        # 假设最大 K=5
        k = config['ksp']['k']
        self.action_space = spaces.MultiDiscrete([self.num_flows, k])
        
        # 定义状态空间: 
        # 1. Flow Features (Size, Period, Deadline) -> dim=3
        # 2. Global Resource State (Mean Utilization) -> dim=1
        # 3. Local Flow State (Scheduled?, Current Path?) -> dim=2
        # 为了简单，我们展平所有流的状态
        # Obs Dim = num_flows * (Flow_Feats + Local_State) + Global_Feats
        #         = num_flows * (3 + 2) + 1
        # 注意：这只是一个简单的特征设计，论文中可能有更复杂的 GNN Embedding
        self.obs_dim = self.num_flows * 5 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        
        # 初始化 SlotManager
        # [解释] 这里从 config 读取参数来设定“棋盘”大小
        self.time_limit = config.get('env_params', {}).get('time_limit', 100)
        self.bandwidth = config.get('env_params', {}).get('bandwidth', 1000)
        self.slot_manager = SlotManager(topology, self.time_limit, self.bandwidth)
        
        # [新增] 如果有初始状态，立即加载
        if self.initial_resource_state is not None:
            self.slot_manager.load_state(self.initial_resource_state)
        
        # === [修改] 直接读取归一化参数 ===
        # config 传入的是 env_params 字典，所以直接 get('normalization')
        norm_config = config.get('normalization', {})
        
        # 使用配置值，如果没配则用默认值兜底 (Safe Fallback)
        self.max_size = float(norm_config.get('max_size', 1500.0))
        self.max_period = float(norm_config.get('max_period', 10000.0))
        self.max_deadline = float(norm_config.get('max_deadline', 10000.0))    
            
        self.reset()

    def reset(self):
        # 1. 重置流状态
        self.current_flow_index = 0
        self.flow_states = {i: {'scheduled': False, 'path_idx': -1, 'slot': -1} 
                           for i in range(len(self.flows))}
        
        # 2. [关键修改] 重置资源
        # 每次 Episode 开始时，不仅要清空当前 Episode 的操作，
        # 还要把“以前 Group 占用的资源”恢复回来。
        if self.initial_resource_state is not None:
            self.slot_manager.load_state(self.initial_resource_state)
        else:
            self.slot_manager.clear()
        
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        flow_id, path_idx = action
        
        # 1. 执行调度逻辑
        # 这里简化：假设 action 直接指定了流和路径
        # 真实逻辑需要调用 Solver 或 heuristic 分配 slot
        
        reward = 0
        done = False
        info = {}
        
        # 检查是否重复调度
        if self.flow_states[flow_id]['scheduled']:
            reward = -1 # 惩罚重复调度
        else:
            # 尝试分配 (Mock)
            # 实际上你需要从 flows[flow_id] 获取 candidate_paths[path_idx]
            # 然后调用 self.slot_manager.allocate(...)
            success = True # 假设分配成功
            
            if success:
                self.flow_states[flow_id]['scheduled'] = True
                self.flow_states[flow_id]['path_idx'] = path_idx
                reward = 1
            else:
                reward = -0.1 # 分配失败惩罚
        
        self.steps += 1
        
        # 检查是否结束 (所有流都调度过，或达到最大步数)
        all_scheduled = all(f['scheduled'] for f in self.flow_states.values())
        if all_scheduled or self.steps >= self.config.get('env_params', {}).get('max_steps', 100):
            done = True
            
        return self._get_observation(), reward, done, info

    def _get_observation(self): # 这里就是输入给神经网络的向量 h 的定义。
        # 构建观察向量
        obs = []
        
        # 1. Global Resource
        util = self.slot_manager.get_utilization() #这里可能要改成更复杂的全局状态
        obs.append(util)
        
        # 2. Flow States
        for i in range(self.num_flows):
            f = self.flows[i]
            # Normalize features (Simple logic)
            obs.append(f['size'] / self.max_size)
            obs.append(f['period'] / self.max_period)
            obs.append(f['deadline'] / self.max_deadline)
            # Status
            obs.append(1.0 if self.flow_states[i]['scheduled'] else 0.0)
            obs.append(self.flow_states[i]['path_idx'] / 5.0) # Normalize path index
            
        return np.array(obs, dtype=np.float32)

    def get_mask(self):
        """
        返回 Action Mask (num_flows,)
        1 表示该流还未被调度 (可选)
        0 表示该流已被调度 (不可选)
        """
        mask = np.ones(self.num_flows, dtype=np.float32)
        for i in range(self.num_flows):
            if self.flow_states[i]['scheduled']:
                mask[i] = 0.0
        return mask