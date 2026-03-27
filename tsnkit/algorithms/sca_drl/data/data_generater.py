import os
import pandas as pd
import networkx as nx
import random
    
def generate_industrial_dataset(num_tasks, save_dir="data_storm"):
    os.makedirs(save_dir, exist_ok=True)
    
    # =========================================================
    # 1. 制造完整的“工业控制环网 + 跨界捷径”拓扑 (9个SW + 27个ES)
    # =========================================================
    G = nx.Graph()
    
    # 1.1 添加骨干交换机 (SW: 0~8)
    ring_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 0)
    ]
    chord_edges = [(2, 7),(3, 6)] # 中间的捷径
    G.add_edges_from(ring_edges + chord_edges)
    
    # 1.2 添加端系统 (ES: 9~35) 并连接到对应的 SW
    es_nodes = []
    es_id = 9
    for sw_id in range(0, 9): # 对于每个交换机 0~8        
        for _ in range(3): # 每个交换机外挂 3 个ES
            G.add_edge(es_id, sw_id)
            es_nodes.append(es_id)
            es_id += 1
            
    # 转换为双向的有向图
    G_dir = G.to_directed()
    
    topo_data = []
    for u, v in G_dir.edges():
        # 默认参数 (宽阔的高速公路)
        rate = 1      # 1Gbps
        t_prop = 0    
            
        topo_data.append({
            'link': f"({u}, {v})",
            'q_num': 8,       
            'rate': rate,        
            't_proc': 2000,   
            't_prop': t_prop       
        })
    
    topo_df = pd.DataFrame(topo_data)
    topo_path = os.path.join(save_dir, "0_topo.csv")
    topo_df.to_csv(topo_path, index=False)
    print(f"✅ 完整异构拓扑生成完毕: 36个节点 (9 SW + 27 ES), {G_dir.number_of_edges()}条有向边。")

    # =========================================================
    # 2. 制造流量风暴 Task：随机生成 N 条复杂的异构流
    # =========================================================
    
    MIN_FLOWS = 60
    MAX_FLOWS = 90            
    
    # # 周期池 (纳秒): 250us, 500us, 750us, 1ms, 2ms, 3ms    
    # period_pool = [250000, 500000, 750000, 1000000, 2000000, 3000000]
    # 周期池 (纳秒): 200us, 400us, 800us, 600us 1.5ms, 3ms    
    period_pool = [200000, 400000, 800000, 600000, 1500000, 3000000]
    
    # 核心节点
    core_es_nodes = [9,10,11,21,22,23] # SW_0 和 SW_4
    edge_es_nodes = [n for n in es_nodes if n not in core_es_nodes]    

    print(f"🌪️ 正在生成 {num_tasks} 份高压任务考卷...")       
    all_generated_tasks = [] # 用于暂存所有卷子
    
    for task_id in range(1, num_tasks + 1):
        # 为了让模型学会适应不同数量的流，我们在 MIN_FLOWS 到 MAX_FLOWS 之间随机抽取
        num_flows = random.randint(MIN_FLOWS,MAX_FLOWS) 
        
        task_data = []
        for stream_id in range(num_flows):            
            
            # 🛡️ 策略 3：80/20 法则制造严重的空间拥塞
            if random.random() < 0.8:
                # 80% 核心业务流：向心或离心
                if random.random() < 0.5:
                    src = random.choice(edge_es_nodes)
                    dst = random.choice(core_es_nodes)
                else:
                    src = random.choice(core_es_nodes)
                    dst = random.choice(edge_es_nodes)
            else:
                # 20% 背景杂波流
                src = random.choice(edge_es_nodes)
                dst = random.choice(edge_es_nodes)
            
            # 过滤逻辑：判断它们是不是属于同一个交换机，如果是则重新选目的节点
            while (src - 9) // 3 == (dst - 9) // 3 or src == dst:
                dst = random.choice(es_nodes)       
                
            period = random.choice(period_pool)
            size = random.randint(100, 1500) # 模拟 100B 到 1500B 的以太网帧
            
            task_data.append({
                'stream': stream_id,
                'src': src,
                'dst': f"[{dst}]", # 严格对齐 tsnkit 的 List 字符串格式
                'size': size,
                'period': period,
                'deadline': period, # Deadline 默认等于周期
                'jitter': period
            })
        
        # 不直接存盘，而是打包放进列表
        all_generated_tasks.append({
            'num_flows': num_flows, 
            'data': task_data})
    
    # =========================================================
    # 🌟 3. 核心大招：按难度排序并执行 8:1:1 分层切分！
    # =========================================================
    print("⚖️ 正在执行严谨的难度分层切分 (Stratified Split 8:1:1)...")
    # 3.1 按照流数量排序
    all_generated_tasks.sort(key=lambda x: x['num_flows'])
    # 3.2 切分为训练集、验证集、测试集
    train_count, val_count, test_count = 0, 0, 0
    
    for i, task_bundle in enumerate(all_generated_tasks):
        mol_val = i % 10
        if mol_val < 8:
            prefix = ""
            train_count += 1
        elif mol_val == 8:
            prefix = "val_"
            val_count += 1
        else:
            prefix = "test_"
            test_count += 1
            
        file_name = f"{prefix}{i+1}_task.csv"
        pd.DataFrame(task_bundle['data']).to_csv(os.path.join(save_dir, file_name), index=False)
    
    print(f"✅ 完美切分完毕！生成了 {train_count} 份训练集，{val_count} 份验证集，{test_count} 份测试集。")
    print("📈 它们的难度分布已经达到了统计学上的完全一致！")

if __name__ == "__main__":
    # 建议至少生成 500 份，这样验证集和测试集各有 50 份，大数定律生效！
    generate_industrial_dataset(num_tasks=500)