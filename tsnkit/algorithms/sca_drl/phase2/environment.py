import os
import pandas as pd
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces
import torch

# 假设你的项目中可以通过以下方式导入 tsnkit 的 ls 算法类
# 如果路径不同，请根据你的项目结构调整
from tsnkit.algorithms.ls import ls 
# 在 environment_ls_copy.py 的顶部 imports 区域加上：
from tsnkit.core._constants import T_SLOT
# from sca_drl.common.utils import resolve_path

# =====================================================================
# 模块一：物理引擎外壳 (DRL_PhysicsEngine)
# 职责：继承 tsnkit 的 ls.py，劫持其底层计算能力，提供单步交互接口
# =====================================================================
class DRL_PhysicsEngine(ls):
    def __init__(self, task, topo, workers=1):
        """
        初始化物理引擎，对接 tsnkit 的 Task 和 Network 对象
        """
        super().__init__(workers)
        self.task = task
        self.topo = topo
        self.task_routes = {s: self.topo.get_all_path(s.src, s.dst) for s in self.task.streams}
        # 初始化底层的状态账本
        self.deep_clear()

    def deep_clear(self):
        """
        彻底清空底层物理世界的残留物，用于 Episode 重置
        """
        self._delay = {}
        # 初始化空白的全局 GCL (基于连续区间)
        self._result = {l: [] for l in self.topo.links}
        self._paths = {}
        self._offset = {}

    def try_allocate_agent_action(self, task_stream, path):
        """
        核心物理执行器：接收单步动作，调用 tsnkit 检验，成功则写入账本
        返回: (是否成功: bool, 瓶颈链路利用率: float)
        """
        # 1. 呼叫 tsnkit 最核心的物理碰撞检测神仙函数
        inject_time = self.find_inject_time(task_stream, path)
        
        if inject_time == -1:
            # 物理法则判定：无处安放，直接驳回
            return False, 1.0 
            
        # 2. 如果成功，复用 ls.py 的写入逻辑，彻底锁定整个 LCM 内的时隙
        self._delay[task_stream] = self.get_nw_delay(task_stream, path)
        self._paths[task_stream] = path
        self._offset[task_stream] = inject_time

        _prev_end = inject_time
        max_util = 0.0

        for l in path.links:
            _start = _prev_end
            _end = _start + task_stream.get_t_trans(l)
            _prev_end = _start + l.t_proc + task_stream.get_t_trans(l)

            # 在全网 LCM 的时间轴上，铺满这个流的每一个复现周期！
            for k in task_stream.get_frame_indexes(self.task.lcm):
                # 记录格式: (start, end, queue)
                self._result[l].append((_start + k * task_stream.period, _end + k * task_stream.period, 0))
                
            # 必须重新排序，因为 tsnkit 的 match_time 依赖有序列表
            self._result[l].sort(key=lambda x: x[0], reverse=False)

            # 3. 顺手计算当前链路的准确利用率 (用于给 Agent 算惩罚)
            total_occupied = sum([entry[1] - entry[0] for entry in self._result[l]])
            util = total_occupied / self.task.lcm
            if util > max_util:
                max_util = util

        return True, max_util

# =====================================================================
# 模块二：强化学习环境 (TSNEnv)
# 职责：Agent 的翻译官，状态图的绘制者，马尔可夫决策过程的控制台
# =====================================================================
class TSNEnv(gym.Env):
    def __init__(self, env_config: dict, phase1_embeddings: dict = None):
        super(TSNEnv, self).__init__()
        
        # [🔧 Config重构] 读取配置
        self.topo = env_config['topo']
        env_params = env_config.get('environment',{})
        
        self.MAX_FLOWS = env_params.get('max_flows', 130)        
        self.K_MAX = env_params.get('k_max', 5) 
        
        # [🔧 Config重构] 读取奖励权重
        reward_cfg = env_params.get('reward', {}) 
        self.omega_0 = reward_cfg.get('all_success', 10.0)
        self.omega_1 = reward_cfg.get('one_success', 1.0)
        self.omega_2 = reward_cfg.get('hop_penalty', 0.01)
        self.omega_3 = reward_cfg.get('util_penalty', 0.1)
        self.omega_4 = reward_cfg.get('failure', 0.0)       
        
        # [🌟 Embedding接入] 动态获取 embedding 维度
        self.phase1_embeddings = phase1_embeddings
        if self.phase1_embeddings is not None and len(self.phase1_embeddings) > 0:
            sample_emb = next(iter(self.phase1_embeddings.values()))
            self.emb_dim = sample_emb.shape[0] if hasattr(sample_emb, 'shape') else len(sample_emb)
        else:
            self.emb_dim = env_params.get('emb_dim', 1)  # 从 phase2.yaml environment.emb_dim 读取
        
        # 获取所有边，并固化索引 (用于生成固定维度的 global_snapshot)全局快照
        self.edges = self.topo.links 
        self.num_edges = len(self.edges)
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        
        # 动作空间与维度裁剪设定 (Padding & Truncation)，固定为 MAX_FLOWS * K_MAX        
        self.action_space = spaces.Discrete(self.MAX_FLOWS * self.K_MAX)
        
        # 设定观测窗口大小，比如观察未来 1024 个量子化时隙 
        self.T_slot = T_SLOT
        
        # 借用初始 task 算出固定 W （第一个task的流的LCM）
        initial_task = env_config['task'] # 提前把初始考卷拿出来看一眼
        max_acceptable_W = env_config.get("environment", {}).get("max_obs_window", 10000)
        
        # ===========提取初始考卷的 LCM=================
        if hasattr(initial_task, 'lcm'):
            actual_lcm_slots = initial_task.lcm
        elif hasattr(initial_task, '_lcm'):
            actual_lcm_slots = initial_task._lcm
        else:
            actual_lcm_slots = int(np.lcm.reduce([f.period for f in initial_task.streams]))

        # W 在这里被彻底定死！后续 load_new_task 绝对不许再改 self.W！
        if actual_lcm_slots <= max_acceptable_W:
            self.W = actual_lcm_slots
            print(f"🌍 [Env 初始化] 神经网络输入维度已锁定，观测窗口 W 设为初始 LCM: {self.W} 槽")
        else:
            self.W = max_acceptable_W
            print(f"⚠️ [Env 警告] 初始 LCM 过大！视野被强制截断为: {self.W} 槽") 
        #=================================================================
        
        # 特征维度计算 (4个基础属性 + 跳数 + 链路(重叠度 + 各路利用率) + phase1_embedding)
        self.d_feature = 4 + 3 * self.K_MAX + self.emb_dim 
        
        # 3. 观测空间永远固定为 MAX_FLOWS
        self.observation_space = spaces.Dict({
            "flow_tokens": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.MAX_FLOWS, self.d_feature), dtype=np.float32
            ),
            "global_snapshot": spaces.Box(
                low=0.0, high=1.0, 
                shape=(self.num_edges * self.W,), dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, 
                shape=(self.MAX_FLOWS * self.K_MAX,), dtype=np.bool_
            )
        })
                
        # 加载初始任务
        # self.load_new_task(env_config['task'])
        self.load_new_task(env_config['task'], env_config.get('task_file'))
        
    def load_new_task(self, task, task_file_path=None):
        """动态更换考卷，并读取 Phase 1 的分组标签"""
        self.task = task
        self.flows = self.task.streams

        task_lcm = getattr(task, 'lcm', None) or int(np.lcm.reduce([f.period for f in task.streams]))
        print(f"[Env] 换题 | 流数={len(self.flows)} | LCM={task_lcm:,} ns | 可见比={self.W/task_lcm*100:.3f}% | W={self.W} ns")

        # 截断防御：如果流数量超出了我们的 MAX_FLOWS 容忍度，强行截断
        if len(self.flows) > self.MAX_FLOWS:
            print(f"⚠️ 警告: 任务流数量({len(self.flows)}) 超过 MAX_FLOWS({self.MAX_FLOWS})，执行截断！")
            self.flows = self.flows[:self.MAX_FLOWS]
            
        self.num_flows = len(self.flows)
        # 重新挂载物理引擎
        self.physics_engine = DRL_PhysicsEngine(self.task, self.topo)
        
        # 重新计算归一化基准
        self.max_period = max([f.period for f in self.flows]) if self.flows else 1.0
        self.max_size = max([f.size for f in self.flows]) if self.flows else 1.0
        self.max_deadline = max([f.deadline for f in self.flows]) if self.flows else 1.0
        self.max_hops = len(self.topo.nodes)
        
        # 🌟 新增：读取教导主任的分组表
        self.flow_groups = {} # 字典: {流ID: 组号}
        if task_file_path is not None:
            group_csv_path = task_file_path.replace(".csv", "_group.csv")
            if os.path.exists(group_csv_path):
                df_group = pd.read_csv(group_csv_path)
                # 假设 csv 里有 'stream_id' 和 'group_id' 两列
                for _, row in df_group.iterrows():
                    self.flow_groups[str(row['stream_id'])] = int(row['group_id'])
            else:
                print(f"⚠️ 未找到分组表 {group_csv_path}，所有流默认归为 0 组！")
                
        # 🌟 2. 新增：动态读取这张考卷专属的 Embedding 特征
        self.phase1_embeddings = {}
        if task_file_path is not None:
            emb_pt_path = task_file_path.replace(".csv", "_emb.pt")
            if os.path.exists(emb_pt_path):
                # 读入这套卷子专属的 16 维特征字典 {sid: tensor}
                self.phase1_embeddings = torch.load(emb_pt_path, map_location='cpu', weights_only=False)
                loaded_dim = next(iter(self.phase1_embeddings.values())).shape[0]
                if loaded_dim != self.emb_dim:
                    raise ValueError(f"embedding维度不符: 期待 {self.emb_dim}, 实际 {loaded_dim}")
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.flow_states = {}
        for i in range(self.num_flows):
            self.flow_states[i] = {'status': 1}
            
        # 核心：命令物理引擎进行深度清理，抹除上一个 Episode 的痕迹
        self.physics_engine.deep_clear()        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        抽取物理世界的状态，包装成 Transformer 友好的 Tokens 和 Rasterized Window。
        """        
        # ================= 🔍 照妖镜调试代码 =================
        # 我们只在第一把游戏的第 1 步打印一次，防止刷屏
        if not hasattr(self, 'debug_printed') and self.steps == 1:
            print("\n" + "="*50)
            print("🚀 [DEBUG INFO] 深入物理引擎底层探查！")
            print(f"1. 当前设定的窗口大小 W: {self.W}")
            print(f"2. 当前设定的时间槽 T_slot: {self.T_slot}")
            
            # 抽查第 0 条流的真实属性
            f_test = self.flows[0]
            print(f"3. 抽查流 0 的真实周期 (period): {f_test.period}")
            # print(f"   (期望值：如果是 250us 且 T_slot=1000, 这里应该是 250！)")
            print(f"4. 抽查流 0 的真实包大小 (size): {f_test.size}")
            
            # 抽查物理引擎已经排进去的真实区间
            if self.physics_engine._result:
                # 随便找一条有数据的边
                sample_edge = list(self.physics_engine._result.keys())[0]
                sample_intervals = self.physics_engine._result[sample_edge]
                print(f"5. 物理引擎真实分配的区间示例 (边 {sample_edge}):")
                print(f"   {sample_intervals[:5]} ...")
                
                # 检查有没有超出 W=2000 的区间被丢弃？
                max_end_time = max([iv[1] for iv in sample_intervals]) if sample_intervals else 0
                print(f"6. 当前这条边上，被分配的最晚结束时间: {max_end_time}")
                if max_end_time > self.W:
                    print("   🚨 [严重警告] 存在区间超出了 W=2000，Agent 的眼睛被蒙住了！")
            else:
                print("5. 当前没有任何流被排进去 (_result 为空)")
                
            print("="*50 + "\n")
            self.debug_printed = True
        # =====================================================
        
        obs_tokens = []
        pending_flows = [i for i in range(self.num_flows) if self.flow_states[i]['status'] == 1]
        
        # 步骤 A：构建动态重叠度热度图 (基于候选路径)
        link_heat_map = collections.defaultdict(int)
        for flow_idx in pending_flows:
            flow_obj = self.flows[flow_idx]
            routes = self.physics_engine.task_routes[flow_obj]
            # 仅取前 K_MAX 条路径参与热度计算
            for path in routes[:self.K_MAX]:
                for link in path.links:
                    link_heat_map[link] += 1
                    
        # 步骤 B：生成 Token 矩阵
        for i in range(self.num_flows):
            f = self.flows[i]
            status = self.flow_states[i]['status']
            
            norm_period = f.period / self.max_period
            norm_size = f.size / self.max_size
            norm_deadline = f.deadline / self.max_deadline
            
            H_i = np.zeros(self.K_MAX, dtype=np.float32)
            O_i = np.zeros(self.K_MAX, dtype=np.float32)
            # 🌟 新增：动态物理瓶颈利用率
            U_i = np.zeros(self.K_MAX, dtype=np.float32) 
            
            routes = self.physics_engine.task_routes[f]
            num_valid_paths = min(len(routes), self.K_MAX)
            
            for k in range(num_valid_paths):
                path = routes[k]
                links = path.links
                H_i[k] = len(links) / self.max_hops
                
                if status == 1:
                    overlap_score = sum(link_heat_map[link] for link in links)
                    O_i[k] = overlap_score - len(links)
                    
                    # 计算路径上的平均利用率
                    # 🌟 2. 新增：动态物理利用率 (Bottleneck Util)
                    path_link_utils = []
                    for link in links:
                        # 抄底层物理引擎的作业：直接读取真实账本 self._result
                        # self._result[link] 里存的是 [(start, end, queue), ...]
                        
                        # 把这条链路上所有被占用的时间段加起来
                        total_occupied = sum([entry[1] - entry[0] for entry in self.physics_engine._result[link]])
                        
                        # 除以 LCM 得到绝对真实的利用率！
                        util = total_occupied / self.physics_engine.task.lcm
                        
                        path_link_utils.append(util)
                    
                    # 这条候选路径的瓶颈，就是它所有链路中最堵的那条
                    U_i[k] = max(path_link_utils) if path_link_utils else 0.0

            # 🌟 核心修改：动态获取该流的 Embedding
            stream_id = getattr(f, 'name', str(getattr(f, 'id', i)))
            if self.phase1_embeddings is not None and stream_id in self.phase1_embeddings:
                emb = self.phase1_embeddings[stream_id]
                # 处理如果 phase1 吐出的是 GPU Tensor 的情况
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
            else:
                emb = np.zeros(self.emb_dim, dtype=np.float32)
                    
            # padding 补齐 K_MAX，多余位置保持 0
            token = np.concatenate([
                [norm_period, norm_size, norm_deadline, status], 
                H_i, 
                O_i, 
                U_i,
                emb # 假设无 prior，填 0
            ]).astype(np.float32)
            
            obs_tokens.append(token)
            
        # 步骤 B.5：Padding 幽灵流填充
        num_padding = self.MAX_FLOWS - self.num_flows
        for _ in range(num_padding):
            # 创建全 0 特征，但重点是把 status (索引 3) 设为 -2.0
            # 这样 Agent 的 _get_action_mask 就会立刻把它屏蔽掉！
            ghost_token = np.zeros(self.d_feature, dtype=np.float32)
            ghost_token[3] = -2.0 
            obs_tokens.append(ghost_token)

        # 步骤 C：实施方案 B —— 固定窗口时隙栅格化 (Rasterization)
        # 建立一个干净的 [边数, 窗口大小] 0/1 矩阵
        g_window = np.zeros((self.num_edges, self.W), dtype=np.float32)
        
        for link, intervals in self.physics_engine._result.items():
            if link not in self.edge_to_idx:
                continue
            link_idx = self.edge_to_idx[link]
            
            for start, end, _ in intervals:
                # 1. 物理时间 -> 离散格子索引
                # 只截取位于 0 到 W 窗口内的时隙部分
                s_idx = int(start)                
                e_idx = int(end)                
                
                # 3.边界截断保护
                s_idx = max(0, s_idx)
                e_idx = min(self.W, e_idx)
                
                # 4. 绘制到矩阵
                if s_idx < e_idx:
                    g_window[link_idx, s_idx:e_idx] = 1.0
                        
        # 🌟 新增：生成动作掩码 (Action Mask)
        # 动作空间大小是 num_flows * K_MAX
        action_mask = np.zeros(self.MAX_FLOWS * self.K_MAX, dtype=np.bool_)
        
        # 🌟 1. 动态探测当前活跃组别
        # pending_flows 之前已经在上面算过了：pending_flows = [i for i in range(self.num_flows) if self.flow_states[i]['status'] == 1]
        current_active_group = 0
        if pending_flows:
            # 找到所有待排流的所属组，取最小的那个作为当前活动组
            active_groups = []
            for i in pending_flows:
                f = self.flows[i]
                sid = getattr(f, 'name', str(getattr(f, 'id', i)))
                active_groups.append(self.flow_groups.get(sid, 0)) # 查不到默认给 0
            current_active_group = min(active_groups)
        
        # 🌟 2. 实施三重门禁    
        for i in range(self.num_flows):
            f = self.flows[i]
            sid = getattr(f, 'name', str(getattr(f, 'id', i)))
            my_group = self.flow_groups.get(sid, 0)
            
            # 三重门禁：只有 status=1 的流，且属于当前活动组，才能被考虑；其他一律屏蔽
            
            if self.flow_states[i]['status'] == 1 and my_group == current_active_group: # 只有待调度的流才能选
                routes = self.physics_engine.task_routes[f]
                num_valid_paths = min(len(routes), self.K_MAX)
                
                for k in range(num_valid_paths):
                    action_idx = i * self.K_MAX + k
                    action_mask[action_idx] = True # 这个动作是合法的！
                    
        # 把 mask 一并打包返回
        return {
            'flow_tokens': np.array(obs_tokens, dtype=np.float32),
            'global_snapshot': g_window.flatten(), # 展平该窗口，喂给 Agent
            'action_mask': action_mask # 🌟 塞进 obs 字典里
        }

    def step(self, action):
            """
            环境推演：解密动作，调用物理引擎，结算稠密奖励。
            """
            flow_idx = action // self.K_MAX
            path_idx = action % self.K_MAX
            
            # 🛡️ 致命拦截：如果在你的环境中经常触发这个，说明你的 Masking 代码写崩了！
            if flow_idx >= self.num_flows or self.flow_states[flow_idx]['status'] != 1:
                # 给出极大的负惩罚，并截断，防止死循环
                return self._get_observation(), -10.0, True, False, {"error": "Masking Failed: Invalid flow selected!"}
                
            f = self.flows[flow_idx]
            routes = self.physics_engine.task_routes[f]
            
            # 🛡️ 路径拦截
            if path_idx >= len(routes):
                return self._get_observation(), -5.0, True, False, {"error": "Masking Failed: Invalid path selected!"}
                
            path = routes[path_idx]
            
            # ==========================================
            # 🔓 架构师的万能开锁器：破解 tsnkit 封装的 Path
            # ==========================================
            hop_count = 1  # 默认保底跳数
            
            if hasattr(path, 'edges'):
                # 如果它封装了边列表，跳数 = 边的数量
                hop_count = len(path.edges)
            elif hasattr(path, 'links'):
                hop_count = len(path.links)
            elif hasattr(path, 'nodes'):
                # 如果它封装了节点列表，跳数 = 节点数 - 1
                hop_count = len(path.nodes) - 1
            else:
                # 🚨 如果全都没命中，直接启动 X光机，打印它的底裤！
                print(f"\n🚨 [架构师拦截] 未知 Path 结构！")
                print(f"   对象内容: {path}")
                print(f"   包含的属性和方法: {dir(path)}")
                
                # 既然不知道多长，为了让训练不崩溃，先强行给个跳数 2
                hop_count = 2 
                
            # 确保跳数绝对不能小于 1
            hop_count = max(1, hop_count)            
            
            hop_penalty = self.omega_2 * hop_count #跳数/延时惩罚           
            
            # --- 核心：将真实对象抛给底层 tsnkit 引擎 ---
            is_success, util_score = self.physics_engine.try_allocate_agent_action(f, path)
            
            # 🌟 革命性的奖励重构 (稠密 & 宽容)
            if is_success:
                self.flow_states[flow_idx]['status'] = 0 
                # 成功排入，基础分 +1.0。
                # 引入负载均衡惩罚：如果这条路很堵(util_score高)，稍微扣一点分 (比如 1.0 - 0.5*0.8 = 0.6分)
                # 鼓励 Agent 去找不堵的、空闲的路径！
                # 加入防御性 max(..., 0.01)，防止扣分超标变成惩罚成功！
                reward = max(self.omega_1 - (self.omega_3 * util_score) - hop_penalty, 0.01)
                
            else:
                self.flow_states[flow_idx]['status'] = -1 
                # 失败了，不要给巨大的 -xi。我们给一个极其微小的惩罚，或者干脆给 0！
                # 这里的哲学是：没排进去，你只是没拿到那 1.0 分而已。不要因为物理无解而让梯度崩溃。
                reward = self.omega_4  # 👈 核心改变！你也可以写 -0.1，但绝对不能是 -2.0 这种大数字。
                
            self.steps += 1
            
            all_processed = all(state['status'] != 1 for state in self.flow_states.values())
            terminated = all_processed or self.steps >= self.num_flows
            
            info = {
                'flow_idx': flow_idx,
                'is_allocated': is_success,
                'bottleneck_util': util_score if is_success else 1.0,
                'hop_count': hop_count if is_success else 0  
            }
            
            # ==========================================
            # 🏆 架构师的终极大奖 (Global Jackpot)
            # ==========================================
            if terminated:
                success_count = sum(1 for state in self.flow_states.values() if state['status'] == 0)
                p_success = success_count / self.num_flows
                info['group_success_rate'] = p_success
                

                if p_success == 1.0:
                    # 如果 100% 调度成功，大奖
                    jackpot_bonus = self.omega_0 
                    reward += jackpot_bonus
                    
                    # (可选) 在终端里稍微撒个花，方便你观察它有没有“开窍”
                    print(f"\n🎉 [环境撒花] 达成 100% 完美调度！发放全局通关大奖: +{jackpot_bonus}")
                
            return self._get_observation(), reward, terminated, False, info