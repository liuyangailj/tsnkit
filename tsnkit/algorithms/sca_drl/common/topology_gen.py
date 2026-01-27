import networkx as nx
import random
import matplotlib.pyplot as plt

class TopologyGenerator:
    def __init__(self, num_nodes, seed=42):
        self.num_nodes = num_nodes
        self.seed = seed
        
    def generate_graph(self):
        """生成拓扑图 G (NetworkX Object)"""
        # 使用 Barabasi-Albert 模型生成无标度网络 (类似真实网络)
        # m=2 表示每次新加入节点连接 2 个旧节点
        G = nx.barabasi_albert_graph(self.num_nodes, 2, seed=self.seed)
        
        # 转换为有向图 (因为 TSN链路是双向全双工，通常建模为双向有向边)
        G = G.to_directed()
        
        # 为每条链路添加初始属性 (带宽、传播时延等，暂时设为默认)
        for u, v in G.edges():
            G.edges[u, v]['bandwidth'] = 1e9  # 1 Gbps
            G.edges[u, v]['propagation_delay'] = 1  # 1 us
            
        return G

    def save_graph(self, G, path):
        nx.write_gml(G, path)
        print(f"Topology saved to {path}")
        
    def visualize_graph(self, G, save_path=None):
        """
        绘制并保存拓扑图
        """
        plt.figure(figsize=(10, 8)) # 设置画布大小
        
        # 使用 spring_layout 布局，会让连接多的节点（Hub）在中间，比较美观
        # 固定 seed 保证每次画出来的形状一样
        pos = nx.spring_layout(G, seed=self.seed) 
        
        # 1. 画节点
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', edgecolors='black')
        
        # 2. 画标签 (节点ID)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # 3. 画边 (带箭头)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=15, width=1.0)
        
        plt.title(f"Generated Network Topology (N={self.num_nodes})")
        plt.axis('off') # 关掉坐标轴

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Topology image saved to {save_path}")
        else:
            plt.show()
        
        plt.close() # 关闭画布释放内存