## 📂 项目文件结构 (Project Structure)
```text
📁 /
    📄 environment_ls_copy.py
    📄 scheduler_agent_transformer.py
    📄 train.py
    📄 __init__.py
```

# 💻 源代码上下文 (Source Code Context)

## File: `environment_ls_copy.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\environment_ls_copy.py`

```python
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

# 假设你的项目中可以通过以下方式导入 tsnkit 的 ls 算法类
# 如果路径不同，请根据你的项目结构调整
from tsnkit.algorithms.ls import ls 
# 在 environment_ls_copy.py 的顶部 imports 区域加上：
from tsnkit.core._constants import T_SLOT

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
    def __init__(self, env_config):
        super(TSNEnv, self).__init__()
        
        # 1. 解析物理数据，构建网络拓扑        
        self.topo = env_config['topo']   
        self.MAX_FLOWS = env_config.get('max_flows', 130)        
        self.K_MAX = env_config.get('k_max', 5) 
        
        # 获取所有边，并固化索引 (用于生成固定维度的 g_global)
        self.edges = self.topo.links 
        self.num_edges = len(self.edges)
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        
        # 2. 空间与维度裁剪设定 (Padding & Truncation)        
        # 动作空间永远固定为 MAX_FLOWS * K_MAX
        self.action_space = spaces.Discrete(self.MAX_FLOWS * self.K_MAX)
        
        # 设定观测窗口大小 (方案 B：Fixed-Window Rasterization)
        # 比如观察未来 1024 个量子化时隙
        self.T_slot = T_SLOT
        
        # =====借用初始 task 算出固定 W （第一个task的流的LCM）
        initial_task = env_config['task'] # 提前把初始考卷拿出来看一眼
        max_acceptable_W = env_config.get('max_obs_window', 10000) 
        
        # 提取初始考卷的 LCM
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
        
        # 特征维度计算 (4个基础属性 + 跳数 + 重叠度 + 各路利用率 + 先验 = 5 + 3*K_MAX)
        self.d_feature = 5 + 3 * self.K_MAX  
        
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
        
        self.lambda_2 = env_config.get('lambda_2', 0.5)
        self.xi = env_config.get('xi', 2.0)
        self.alpha = env_config.get('alpha', 10.0)
        
        # 加载初始任务
        self.load_new_task(env_config['task'])
        
    def load_new_task(self, task):
        """
        动态更换考卷的接口。用于 Stage 2 训练。
        """
        self.task = task
        self.flows = self.task.streams
        
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

                    
            # padding 补齐 K_MAX，多余位置保持 0
            token = np.concatenate([
                [norm_period, norm_size, norm_deadline, status], 
                H_i, 
                O_i, 
                U_i,
                [0.0] # 假设无 prior，填 0
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
        
        for i in range(self.num_flows):
            if self.flow_states[i]['status'] == 1: # 只有待调度的流才能选
                f = self.flows[i]
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
            
            hop_penalty = 0.05 * hop_count            
            
            # --- 核心：将真实对象抛给底层 tsnkit 引擎 ---
            is_success, util_score = self.physics_engine.try_allocate_agent_action(f, path)
            
            # 🌟 革命性的奖励重构 (稠密 & 宽容)
            if is_success:
                self.flow_states[flow_idx]['status'] = 0 
                # 成功排入，基础分 +1.0。
                # 引入负载均衡惩罚：如果这条路很堵(util_score高)，稍微扣一点分 (比如 1.0 - 0.5*0.8 = 0.6分)
                # 鼓励 Agent 去找不堵的、空闲的路径！
                # 加入防御性 max(..., 0.01)，防止扣分超标变成惩罚成功！
                reward = max(1.0 - (self.lambda_2 * util_score) - hop_penalty, 0.01)
                
            else:
                self.flow_states[flow_idx]['status'] = -1 
                # 失败了，不要给巨大的 -xi。我们给一个极其微小的惩罚，或者干脆给 0！
                # 这里的哲学是：没排进去，你只是没拿到那 1.0 分而已。不要因为物理无解而让梯度崩溃。
                reward = 0.0  # 👈 核心改变！你也可以写 -0.1，但绝对不能是 -2.0 这种大数字。
                
            self.steps += 1
            
            all_processed = all(state['status'] != 1 for state in self.flow_states.values())
            terminated = all_processed or self.steps >= self.num_flows
            
            info = {
                'flow_idx': flow_idx,
                'is_allocated': is_success,
                'bottleneck_util': util_score if is_success else 1.0,
                'hop_count': hop_count if is_success else 0  # 🌟 把跳数塞给上帝（你）看！
            }
            
            # 🌟 删除了原本在 terminated 时追加的 self.alpha * p_success 
            # 因为步骤上的 +1.0 累积起来，本身就已经在最大化全局成功率了！
            if terminated:
                success_count = sum(1 for state in self.flow_states.values() if state['status'] == 0)
                p_success = success_count / self.num_flows
                info['group_success_rate'] = p_success
                
                # ==========================================
                # 🏆 架构师的终极大奖 (Global Jackpot)
                # ==========================================
                if p_success == 1.0:
                    # 如果 100% 调度成功，在最后一步把大奖砸给它！
                    # self.alpha 目前配置里是 10.0。你可以根据刺激程度去调大它（比如 20.0 或 50.0）
                    jackpot_bonus = self.alpha 
                    reward += jackpot_bonus
                    
                    # (可选) 在终端里稍微撒个花，方便你观察它有没有“开窍”
                    print(f"\n🎉 [环境撒花] 达成 100% 完美调度！发放全局通关大奖: +{jackpot_bonus}")
                
            return self._get_observation(), reward, terminated, False, info
```

## File: `scheduler_agent_transformer.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\scheduler_agent_transformer.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

# =====================================================================
# 核心工具：CleanRL 标配的正交初始化函数
# 作用：保证深层网络梯度传播的稳定性，防止梯度消失或爆炸
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# 模块一：神经网络架构 (Congestion-Aware Transformer)
# =====================================================================
class CongestionAwareTransformer(nn.Module):
    def __init__(self, num_flows, k_max, d_feature, global_dim, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.num_flows = num_flows
        self.k_max = k_max
        self.d_model = d_model
        
        # 1. 状态嵌入层 (State Embedding) - 严格对齐论文 ReLU 公式
        self.flow_embedder = nn.Sequential(
            layer_init(nn.Linear(d_feature, d_model)),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.global_embedder = nn.Sequential(
            layer_init(nn.Linear(global_dim, d_model * 2)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model * 2, d_model)),
            nn.LayerNorm(d_model)
        )
        
        # 2. 拥塞感知核心大脑 (Transformer Encoder 共享躯干)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,  
            norm_first=True    
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Actor 策略输出头 (Actor Head)
        # 注意：Actor 最后一层初始化 std=0.01，让初始动作概率分布尽量均匀，鼓励早期探索
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, k_max), std=0.01) 
        )
        
        # 4. Critic 价值评估头 (Critic Head)
        # 注意：Critic 最后一层初始化 std=1.0
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, 1), std=1.0)
        )

    def forward(self, flow_tokens, global_snapshot):
        # ==========================================
        # 1. 构造 Transformer 的 Attention Mask (防污染)
        # ==========================================
        is_padding = (flow_tokens[..., 3] == -2.0) # [B, MAX_FLOWS]
        
        # global token 永远是真实的，不能被 mask 掉
        global_mask = torch.zeros((flow_tokens.shape[0], 1), dtype=torch.bool, device=flow_tokens.device)
        padding_mask = torch.cat([is_padding, global_mask], dim=1) # [B, MAX_FLOWS + 1]
        
        # ==========================================
        # 2. Embedding 与 纯净的 Transformer 编码
        # ==========================================
        
        flow_emb = self.flow_embedder(flow_tokens)         
        global_emb = self.global_embedder(global_snapshot).unsqueeze(1)         
        seq_emb = torch.cat([flow_emb, global_emb], dim=1) 
        
        # 🌟 核心：必须传给 Transformer，从源头掐断幽灵流的干扰！
        out_seq = self.transformer(seq_emb, src_key_padding_mask=padding_mask)                
        
        # 拆分特征
        out_flow = out_seq[:, :-1, :]       # [B, MAX_FLOWS, d_model] 所有Batch，从头取到倒数第二个token，所有特征
        global_repr = out_seq[:, -1, :]     # [B, d_model] 所有Batch，取最后一个token（全局token），所有特征
        
        # ==========================================
        # 3. 融合你的优雅版 Mask-aware Pooling (算 Critic)
        # ==========================================
        valid_mask = (~is_padding).float().unsqueeze(-1) # [B, MAX_FLOWS, 1]
        
        masked_flow_repr = out_flow * valid_mask
        sum_flow = masked_flow_repr.sum(dim=1)              # [B, d_model]
        num_flow = valid_mask.sum(dim=1).clamp(min=1.0)     # [B, 1]
        pooled_repr = sum_flow / num_flow                   # [B, d_model]
        
        # 🌟 把全局视野加回来，让 Critic 看到拥堵情况
        final_critic_state = pooled_repr + global_repr
        value = self.critic_head(final_critic_state)        # [B, 1]
        
        # ==========================================
        # 4. 算 Actor (策略分布)
        # ==========================================
        logits_2d = self.actor_head(out_flow)              
        logits = logits_2d.reshape(-1, self.num_flows * self.k_max) 
        
        return logits, value


# =====================================================================
# 模块二：PPO 智能体与训练引擎 (CleanRL Style)
# =====================================================================
class PPOAgent:
    def __init__(self, env, d_model=64, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, ent_coef=0.0, vf_coef=0.5, k_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_flows = env.MAX_FLOWS
        self.k_max = env.K_MAX
        self.d_feature = env.d_feature
        self.global_dim = env.observation_space['global_snapshot'].shape[0]
        
        self.network = CongestionAwareTransformer(
            num_flows=self.num_flows, k_max=self.k_max, 
            d_feature=self.d_feature, global_dim=self.global_dim,
            d_model=d_model
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.k_epochs = k_epochs

# 🚨 彻底删除旧的 _get_action_mask 函数！我们不再自己算了！

    def get_action_and_value(self, obs, action=None):
        flow_tokens = torch.FloatTensor(obs['flow_tokens']).to(self.device) 
        g_global = torch.FloatTensor(obs['global_snapshot']).to(self.device)
        mask = torch.BoolTensor(obs['action_mask']).to(self.device) # 🌟 接收环境小抄
        
        # 智能升维
        if len(flow_tokens.shape) == 2:
            flow_tokens = flow_tokens.unsqueeze(0) 
        if len(g_global.shape) == 1:
            g_global = g_global.unsqueeze(0)
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # 🌟 Mask 同步升维
            
        logits, value = self.network(flow_tokens, g_global)
        
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)
            
        logits = logits.masked_fill(~mask, -1e8) 
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            return action.item(), probs.log_prob(action), probs.entropy(), value.squeeze()
        else:
            # 兼容 update 时传进来的 Tensor
            return action, probs.log_prob(action), probs.entropy(), value.squeeze()

    def update(self, rollouts):
        b_flow_tokens = torch.FloatTensor(np.array(rollouts['flow_tokens'])).to(self.device)
        b_global = torch.FloatTensor(np.array(rollouts['global_snapshot'])).to(self.device)
        b_actions = torch.LongTensor(rollouts['actions']).to(self.device)
        b_logprobs = torch.FloatTensor(rollouts['logprobs']).to(self.device)
        b_rewards = torch.FloatTensor(rollouts['rewards']).to(self.device)
        b_values = torch.FloatTensor(rollouts['values']).to(self.device)        
        # 🌟 修复核心：装载历史 Mask！
        b_masks = torch.BoolTensor(np.array(rollouts['action_masks'])).to(self.device) 
        b_dones = torch.BoolTensor(np.array(rollouts['dones'])).to(self.device) # 🌟 新增：装载历史 dones 信息
        
        batch_size = len(b_rewards)
        
        with torch.no_grad(): 
            advantages = torch.zeros_like(b_rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    nextnonterminal = 0.0
                    nextvalues = 0.0 
                else:
                    # 🌟 核心救命代码：如果是 True(1.0)，这步就是 0.0 (切断联系)
                    # 如果是 False(0.0)，这步就是 1.0 (保持相连)
                    if b_dones[t]: # 🌟 如果这一轮结束了
                        nextnonterminal = 0.0 # 🌟 切断联系 
                    else:
                        nextnonterminal = 1.0 # 🌟 保持相连
                        
                    nextvalues = b_values[t + 1]
                    # nextnonterminal = 1.0 
                    # nextvalues = b_values[t + 1]
                delta = b_rewards[t] + self.gamma * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # ===== 提取到这里！在整个 800 步的 batch 级别做归一化 =====
        advantages_std = advantages.std()
        if not (torch.isnan(advantages_std) or advantages_std == 0.0):
            advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)
        # ==========================================================
        
        b_inds = np.arange(batch_size)
        for epoch in range(self.k_epochs): 
            np.random.shuffle(b_inds) 
            
            for start in range(0, batch_size, 64):   
                end = start + 64
                mb_inds = b_inds[start:end]
                
                if len(mb_inds) <= 1:
                    continue
                
                logits, newvalues = self.network(b_flow_tokens[mb_inds], b_global[mb_inds])
                
                # 🌟 拿出当时的护盾，继续保护现在的网络计算
                mask = b_masks[mb_inds] 
                logits = logits.masked_fill(~mask, -1e8)
                
                probs = Categorical(logits=logits)
                newlogprob = probs.log_prob(b_actions[mb_inds]) 
                entropy = probs.entropy().mean() 
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                # std = mb_advantages.std()
                # if not (torch.isnan(std) or std == 0.0):
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalues.squeeze() - returns[mb_inds]) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy + v_loss * self.vf_coef

                self.optimizer.zero_grad()  
                loss.backward()             
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) 
                self.optimizer.step()      
                
        return pg_loss.item(), v_loss.item(), entropy.item()
```

## File: `train.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\train.py`

```python
import sys
import os
import time
import glob
import random
import numpy as np
import torch
from tsnkit import core as utils 
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from environment_ls_copy import TSNEnv
from scheduler_agent_transformer import PPOAgent

def train():
    print("="*60)
    print("🚀 SCA-DRL Phase 2: 终极泛化训练引擎启动 (Padding & Masking) 🚀")
    print("="*60)

    # 1. 自动扫描数据风暴文件夹
    data_dir = "../data/data_storm"
    topo_path = os.path.join(data_dir, "0_topo.csv")
    task_files = glob.glob(os.path.join(data_dir, "*_task.csv"))
    
    if not os.path.exists(topo_path) or len(task_files) == 0:
        print("❌ 找不到数据！请先运行 generate_data.py 生成 data_storm 文件夹！")
        return

    print(f"[1/4] 成功发现多路径网格拓扑与 {len(task_files)} 份随机流量考卷！")
    topo = utils.load_network(topo_path)

    # 随意加载一个 task 用于初始化环境
    initial_task = utils.load_stream(task_files[0])

    env_config = {
        'task': initial_task,
        'topo': topo,
        'max_flows': 130,          
        'k_max': 5,                
        'obs_window_size': 5000,   
        'lambda_2': 0.5,           
        'xi': 2.0,                 
        'alpha': 10.0              
    }

    print("[2/4] 初始化 TSNEnv (容量: 100流) 与 Transformer PPOAgent...")
    # 初始学习率
    initial_lr = 3e-4
    
    env = TSNEnv(env_config)
    agent = PPOAgent(env=env, d_model=64, lr=initial_lr, k_epochs=4, clip_coef=0.2, ent_coef=0.01)
    
    # ==========================================
    # 🛡️ MLOps: 实验追踪与日志持久化系统
    # ==========================================
    # 1. 生成全局唯一的实验时间戳
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    # 2. 为本次实验建立【专属】的 TensorBoard 和 模型保存 文件夹
    run_dir = f"./runs/stage2_multipath_{current_time}"
    model_dir = f"./models/stage2_{current_time}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 3. 拦截器：把所有的 print() 输出同时打在屏幕上，并实时存入硬盘 txt
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # 👈 核心救命代码：强制每写一行就存入硬盘，就算立刻断电死机，日志也在！

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # 将系统标准输出和报错输出全部接管到我们的 log 文件中
    sys.stdout = Logger(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout  

    # 4. 启动 TensorBoard 记录器
    writer = SummaryWriter(log_dir=run_dir)

    print(f"🚀 SCA-DRL 训练引擎启动！")
    print(f"📁 本次实验日志路径: {run_dir}/train_log.txt")
    print(f"💾 本次模型保存路径: {model_dir}")
    
    total_iterations = 200         
    episodes_per_iter = 8         
    
    print(f"[3/4] 训练规划: 共 {total_iterations} 轮, 每轮收集 {episodes_per_iter} 份考卷")
    print("[4/4] ⚔️ 面对风暴吧！Transformer！\n")

    start_time = time.time()
    
    for iteration in range(1, total_iterations + 1):
        # ==========================================
        # 📉 架构师的微调法宝：线性学习率衰减 (Linear LR Annealing)
        # ==========================================
        # 计算当前剩余比例 (第 1 轮是 1.0，第 100 轮是 0.0)
        frac = 1.0 - (iteration - 1.0) / total_iterations
        # 计算当前应该使用的学习率
        lr_now = frac * initial_lr
        
        # 强行霸道地修改 PyTorch 优化器里的学习率
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now        
        
        rollouts = { 'flow_tokens': [], 'global_snapshot': [], 'action_masks': [], 'actions': [], 
                     'logprobs': [], 'rewards': [], 'values': [], 'dones': [] } # 新增 'dones' 用于存储每步是否结束的信息
        
        iter_rewards = []
        iter_success_rates = []
        iter_avg_hops = [] # 新增：每轮的平均跳数统计

        # --- 循环刷题阶段 ---
        for ep in range(episodes_per_iter):
            # 🌟 核心：随机抽一张考卷，并让环境更新它的物理引擎！
            random_task_file = random.choice(task_files)
            new_task = utils.load_stream(random_task_file)   
            # new_task = utils.load_stream(task_files[1])         
            env.load_new_task(new_task)
            
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_hops = []
            
            # 哪怕最大容量是 100，我们这一局实际只需要调度真实的 env.num_flows 次
            for step in range(env.num_flows):
                rollouts['flow_tokens'].append(obs['flow_tokens'])
                rollouts['global_snapshot'].append(obs['global_snapshot'])
                rollouts['action_masks'].append(obs['action_mask'])
                
                action, logprob, entropy, value = agent.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts['actions'].append(action)
                rollouts['logprobs'].append(logprob.item())
                rollouts['rewards'].append(reward)
                rollouts['values'].append(value.item())
                rollouts['dones'].append(terminated) # 🌟存储这一轮是否结束的信息,这步是不是把大结局记下来
                
                ep_reward += reward
                obs = next_obs
                
                # 🌟 新增：如果这步排流成功了，把 info 里传出来的跳数记下来
                if info.get('is_allocated', False) and 'hop_count' in info:
                    ep_hops.append(info['hop_count'])
                
                if terminated:
                    if 'group_success_rate' in info:
                        iter_success_rates.append(info['group_success_rate'])
                    
                    # 🌟 新增：计算这个 Episode 的平均跳数，存进全局列表
                    if ep_hops:
                        iter_avg_hops.append(sum(ep_hops) / len(ep_hops))
                    else:
                        iter_avg_hops.append(0.0)
                    
                    break
            
            iter_rewards.append(ep_reward)

        # --- PPO 梯度回传 ---
        pg_loss, v_loss, ent = agent.update(rollouts)

        avg_reward = np.mean(iter_rewards)
        avg_success = np.mean(iter_success_rates) * 100 if iter_success_rates else 0.0
        avg_hop = np.mean(iter_avg_hops) if iter_avg_hops else 0.0 # 🌟 新增：算出这 8 局的整体平均跳数
        
        # print(f"Iter {iteration:03d} | 成功率: {avg_success:5.1f}% | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")
        print(f"Iter {iteration:03d} | 总流数：{env.num_flows} | 成功率: {avg_success:5.1f}% | 均跳数: {avg_hop:4.2f} | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")

        writer.add_scalar("Metrics/1_Success_Rate", avg_success, iteration)
        writer.add_scalar("Metrics/2_Avg_Reward", avg_reward, iteration)
        writer.add_scalar("Metrics/3_Avg_Hops", avg_hop, iteration) # 🌟 新增：画出跳数下降的完美曲线！
        
        writer.add_scalar("Metrics/4_Learning_Rate", lr_now, iteration) # 🌟 新增：监控学习率的下降轨迹
        
        writer.add_scalar("Loss/1_Policy_Loss", pg_loss, iteration)
        writer.add_scalar("Loss/2_Value_Loss", v_loss, iteration)
        writer.add_scalar("Loss/3_Entropy", ent, iteration)

        # 每 50 轮或者最后一轮，保存一次模型
        if iteration % 50 == 0 or iteration == total_iterations:
            # 注意这里：把硬编码的 "./models" 换成了我们刚才动态生成的 model_dir
            save_path = os.path.join(model_dir, f"ppo_stage2_iter_{iteration}.pth")
            torch.save(agent.network.state_dict(), save_path)
            print(f"💾 模型已保存至: {save_path}")

    writer.close()
    print("\n🎉 训练完美收官！")

if __name__ == "__main__":
    train()
```

## File: `__init__.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\__init__.py`

```python

```
