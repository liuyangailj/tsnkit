import random
import numpy as np

# 导入你项目中的 tsnkit 核心工具 (对应你 ls.py 里的 from .. import core as utils)
# 请根据你的实际目录结构调整 import 路径
from tsnkit import core as utils 

# 导入我们刚刚写好的环境类
from environment_ls_copy import TSNEnv

def run_smoke_test():
    print("="*50)
    print("🚀 SCA-DRL 环境冒烟测试启动 🚀")
    print("="*50)

    # ---------------------------------------------------------
    # 1. 模拟加载真实的 tsnkit 物理数据
    # ---------------------------------------------------------
    # ⚠️ 注意：请将这里的路径替换为你本地真实存在的测试流和拓扑文件！
    task_path = "D:/PY_Project/tsnkit/99_task.csv"  # 替换为真实的流配置文件路径
    topo_path = "D:/PY_Project/tsnkit/99_topo.csv"    # 替换为真实的拓扑配置文件路径
    
    try:
        task = utils.load_stream(task_path)
        topo = utils.load_network(topo_path)
        print(f"✅ 成功加载拓扑与流数据！共有 {len(task.streams)} 条流，{len(topo.links)} 条链路。")
    except Exception as e:
        print(f"❌ 数据加载失败，请检查文件路径是否正确。错误信息: {e}")
        return

    # ---------------------------------------------------------
    # 2. 实例化 TSNEnv
    # ---------------------------------------------------------
    env_config = {
        'task': task,
        'topo': topo,
        'k_max': 5,                # 强制填充到 5 条路径
        'obs_window_size': 1024,   # 观察未来 1024 个微秒的栅格
        'lambda_2': 0.5,
        'xi': 2.0,
        'alpha': 10.0
    }
    
    try:
        env = TSNEnv(env_config)
        print("✅ TSNEnv 实例化成功！物理引擎与包装器正常挂载。")
    except Exception as e:
        print(f"❌ TSNEnv 实例化失败！错误信息: {e}")
        return

    # ---------------------------------------------------------
    # 3. 测试 Reset 功能与状态维度检查
    # ---------------------------------------------------------
    obs, info = env.reset()
    
    flow_tokens = obs['flow_tokens']
    g_global = obs['global_snapshot']
    
    print("\n" + "-"*50)
    print("🔍 状态空间 (Observation Space) 维度审查:")
    print(f"  👉 Flow Tokens 形状: {flow_tokens.shape} (预期: [{env.num_flows}, {env.d_feature}])")
    print(f"  👉 Global Snapshot 形状: {g_global.shape} (预期: [{env.num_edges * env.W}])")
    print("-" * 50 + "\n")

    # ---------------------------------------------------------
    # 4. 模拟智能体交互 (随机动作验证)
    # ---------------------------------------------------------
    print("🏃 开始执行随机调度步进测试...")
    terminated = False
    step_cnt = 0
    total_reward = 0.0

    while not terminated:
        # --- 模拟 Action Masking (找出合法的动作池) ---
        pending_flows = [i for i, state in env.flow_states.items() if state['status'] == 1]
        
        if not pending_flows:
            print("  ⚠️ 警告：环境未返回 terminated，但已经没有待调度的流了！")
            break
            
        # 随机选一条待调度的流
        flow_idx = random.choice(pending_flows)
        
        # 找出这条流合法的路径数量 (防止越界被系统秒杀)
        f = env.flows[flow_idx]
        routes = env.physics_engine.task_routes[f]
        num_valid_paths = min(len(routes), env.K_MAX)
        
        # 随机选一条有效路径
        path_idx = random.randint(0, num_valid_paths - 1)
        
        # 将 2D 决策扁平化为 1D 动作索引
        action = flow_idx * env.K_MAX + path_idx

        # --- 核心交互执行 ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_cnt += 1
        
        # --- 打印单步战报 ---
        status_icon = "🟢" if info['is_allocated'] else "🔴"
        util = info['bottleneck_util']
        
        print(f"  [Step {step_cnt:02d}] 选流:{flow_idx:02d} 选路:{path_idx:02d} | "
              f"结果:{status_icon} | 利用率:{util:.2f} | 即时奖励:{reward:.2f}")

    # ---------------------------------------------------------
    # 5. 最终结果评估
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("\n🔍 [Debug] 物理引擎底层的真实时间槽账本 (GCL 局部展示):")
    # 随便挑一条有数据的链路打印出来看看
    for link, intervals in env.physics_engine._result.items():
        if len(intervals) > 0:
            print(f"  🔗 链路 {link} 被占用的时间区间: {intervals[:5]} ... (共 {len(intervals)} 个周期片段)")
            break # 打印一条感受一下就够了，不然控制台会被几万个区间淹没
    print("🎉 测试完成！环境闭环运行正常！")
    print(f"  👉 总步数: {step_cnt}")
    print(f"  👉 累计总奖励 (含成功率分): {total_reward:.2f}")
    
    if 'group_success_rate' in info:
        print(f"  👉 物理引擎结算组成功率: {info['group_success_rate']*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_smoke_test()