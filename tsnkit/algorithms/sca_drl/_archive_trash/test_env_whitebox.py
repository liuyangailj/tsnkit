import time
import numpy as np

# 导入你项目中的 tsnkit 核心工具
from tsnkit import core as utils 

# 导入你的真实环境类
from environment_ls_copy import TSNEnv

def run_whitebox_test():
    print("="*60)
    print("🚀 SCA-DRL 环境核心逻辑：白盒透视测试启动 🚀")
    print("="*60)

    # ---------------------------------------------------------
    # 1. 加载极简测试数据 (强烈建议使用只有几条流的小规模测试集)
    # ---------------------------------------------------------
    # ⚠️ 请确保这两个文件是一个极简的拓扑和少量的流 (比如 3~5 条)
    task_path = "D:/PY_Project/tsnkit/99_task.csv"  # 建议换成类似 3_task.csv
    topo_path = "D:/PY_Project/tsnkit/99_topo.csv"  
    
    try:
        task = utils.load_stream(task_path)
        topo = utils.load_network(topo_path)
        print(f"✅ 成功加载测试数据！共有 {len(task.streams)} 条流，{len(topo.links)} 条链路。")
    except Exception as e:
        print(f"❌ 数据加载失败，请检查文件路径！错误信息: {e}")
        return

    # ---------------------------------------------------------
    # 2. 实例化 TSNEnv
    # ---------------------------------------------------------
    env_config = {
        'task': task,
        'topo': topo,
        'k_max': 5,                
        'obs_window_size': 20,   # 使用我们推导出的 2000 超周期窗口
        'lambda_2': 0.5,
        'xi': 2.0,
        'alpha': 10.0
    }
    
    env = TSNEnv(env_config)
    print(f"✅ 环境装载完毕。窗口大小 W={env.W}")

    # 重置环境
    obs, info = env.reset()
    
    # ---------------------------------------------------------
    # 3. 模拟人工调度 (白盒透视核心)
    # ---------------------------------------------------------
    # 为了直观，我们只在屏幕上打印前 80 个时间槽的画面
    display_W = min(env.W, 80) 
    
    print("\n" + "👀 "*10 + "开始人工步进透视" + " 👀"*10)
    
    # 我们连续手动走 3 步，强行排前 3 条流
    test_steps = min(env.num_flows, 3)
    for i in range(test_steps):
        time.sleep(1) # 停顿一下，制造动画感
        
        # 🎯 人工指令：强行安排第 i 条流，走它的第 0 条候选路径
        flow_idx = i
        path_idx = 0
        action = flow_idx * env.K_MAX + path_idx

        # 尝试提取这条流真实经过的路径节点，打印出来方便核对
        f = env.flows[flow_idx]
        routes = env.physics_engine.task_routes[f]
        if len(routes) > 0:
            actual_path = routes[path_idx].links
            path_str = " -> ".join([str(link) for link in actual_path])
        else:
            path_str = "无可用路径"

        # 与环境交互
        obs, reward, terminated, truncated, info = env.step(action)

        # ================= 打印直观战报 =================
        print("\n" + "="*80)
        print(f"👉 [Step {env.steps}] 指令：将 [流 {flow_idx}] 排入 [路径 0] ({path_str})")
        print("-" * 80)
        print(f"✅ [引擎反馈] 是否成功: {info.get('is_allocated')} | 奖励 Reward: {reward:.4f} | 拥塞度: {info.get('bottleneck_util', 0):.4f}")

        # 查看状态更新是否正确
        statuses = [env.flow_states[k]['status'] for k in range(min(env.num_flows, 10))]
        print(f"🔄 [前10条流 Status] (0=成功, 1=待排, -1=失败): {statuses}")

        print(f"\n👀 [视觉快照] g_window 局部甘特图 (截取时间槽 0 ~ {display_W}):")
        # 打印刻度尺
        ruler_10 = "".join([f"{x:<10}" for x in range(0, display_W, 10)])
        print(" "*14 + ruler_10)
        print(" "*14 + "|" + ".........|" * (display_W // 10))

        # 将展平的 global_snapshot 还原为矩阵
        g_window_flat = obs['global_snapshot']
        g_window = g_window_flat.reshape((env.num_edges, env.W))

        idx_to_edge = {v: k for k, v in env.edge_to_idx.items()}

        # 遍历所有边，打印时间占用甘特图
        active_edges = 0
        for edge_i in range(env.num_edges):
            row = g_window[edge_i, :display_W]
            
            # 为了过滤噪音，我们只打印“有流量经过”的边
            if np.sum(row) > 0: 
                edge_name = f"Edge {idx_to_edge.get(edge_i, str(edge_i))}"
                # 核心魔法：把 1.0 变成方块 █，0.0 变成点 .
                visual = "".join(["█" if val > 0.5 else "." for val in row])
                print(f"{edge_name:>13} [{visual}]")
                active_edges += 1

        if active_edges == 0:
            print(" "*13 + "(⚠️ 所有链路在当前截取的时间段内均为空 0.0)")

        print("="*80)

        if terminated:
            print("\n🏁 [测试结束] 环境返回 Terminated = True！")
            break

if __name__ == "__main__":
    run_whitebox_test()