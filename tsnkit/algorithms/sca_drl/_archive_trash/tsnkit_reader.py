"""
TSNKit CSV Reader for SCA-DRL

Converts TSNKit CSV format to SCA-DRL internal format.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from pathlib import Path

def read_tsnkit_streams(task_path: str) -> List[Dict]:
    """
    Read TSNKit task CSV and convert to SCA-DRL flow format.

    Args:
        task_path: Path to TSNKit task CSV file
                   Format: stream,src,dst,size,period,deadline,jitter

    Returns:
        List of flow dicts with keys:
        - id: stream ID
        - src: source node
        - dst: destination node (extracted from list like [15])
        - size: frame size in bytes
        - period: period in microseconds
        - deadline: deadline in microseconds
        - jitter: jitter in microseconds
    """
    try:
        df = pd.read_csv(task_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task file not found: {task_path}")

    flows = []
    for _, row in df.iterrows():
        stream_id = row.get('stream', row.get('id', 0))

        # Handle dst field which is in list format like "[15]"
        dst_raw = row['dst']
        if isinstance(dst_raw, str):
            dst_list = eval(dst_raw)
        else:
            dst_list = dst_raw

        if isinstance(dst_list, list) and len(dst_list) > 0:
            dst = dst_list[0]
        else:
            raise ValueError(f"Invalid dst format: {dst_raw}")

        flow = {
            'id': int(stream_id),
            'src': int(row['src']),
            'dst': int(dst),
            'size': int(row['size']),
            'period': int(row['period']),
            'deadline': int(row['deadline']),
            'jitter': int(row['jitter'])
        }
        flows.append(flow)

    return flows


def read_tsnkit_network(topo_path: str) -> Tuple[nx.DiGraph, Dict]:
    """
    Read TSNKit topology CSV and convert to NetworkX graph.

    Args:
        topo_path: Path to TSNKit topology CSV file
                   Format: link,q_num,rate,t_proc,t_prop

    Returns:
        Tuple of:
        - NetworkX DiGraph (undirected for path finding)
        - Link info dict: {(src, dst): {'rate': rate, 't_proc': t_proc, 'q_num': q_num}}
    """
    try:
        df = pd.read_csv(topo_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Topology file not found: {topo_path}")

    # Create undirected graph for path finding (TSNKit uses bidirectional links)
    G = nx.Graph()
    link_info = {}

    for _, row in df.iterrows():
        # Parse link field which is in tuple format like "(0, 1)"
        link_raw = row['link']
        if isinstance(link_raw, str):
            link_tuple = eval(link_raw)
        else:
            link_tuple = link_raw

        src, dst = int(link_tuple[0]), int(link_tuple[1])

        # Add edge to graph
        G.add_edge(src, dst)

        # Store link info
        link_info[(src, dst)] = {
            'rate': float(row['rate']),
            't_proc': int(row['t_proc']),
            'q_num': int(row['q_num']),
            't_prop': int(row['t_prop'])
        }

    return G, link_info


def compute_ksp_paths(G: nx.Graph, flows: List[Dict], k: int = 5) -> List[Dict]:
    """
    Compute K-shortest paths for each flow using NetworkX.

    Args:
        G: NetworkX graph
        flows: List of flow dicts from read_tsnkit_streams
        k: Number of shortest paths to compute

    Returns:
        List of flow dicts with added 'k_paths' key containing list of paths
    """
    flows_with_paths = []

    for flow in flows:
        src, dst = flow['src'], flow['dst']

        try:
            # Get all simple paths and take top K
            all_paths = list(nx.all_simple_paths(G, src, dst))

            if len(all_paths) == 0:
                raise ValueError(f"No path found from {src} to {dst}")

            # Sort by path length and take K
            all_paths.sort(key=lambda p: len(p))
            ksp_paths = all_paths[:k] if k < len(all_paths) else all_paths

            # Update flow with K-shortest paths
            flow_copy = flow.copy()
            flow_copy['k_paths'] = ksp_paths
            flows_with_paths.append(flow_copy)

        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists from {src} to {dst} in the network")

    return flows_with_paths


def read_tsnkit_data(task_path: str, topo_path: str, k: int = 5) -> Tuple[List[Dict], nx.Graph]:
    """
    Main function: Read TSNKit CSV files and convert to SCA-DRL format.

    Args:
        task_path: Path to TSNKit task CSV file
        topo_path: Path to TSNKit topology CSV file
        k: Number of shortest paths to compute for each flow

    Returns:
        Tuple of:
        - List of flow dicts with k_paths
        - NetworkX graph
    """
    # Read streams
    flows = read_tsnkit_streams(task_path)

    # Read network
    G, link_info = read_tsnkit_network(topo_path)

    # Compute K-shortest paths
    flows_with_paths = compute_ksp_paths(G, flows, k)

    return flows_with_paths, G


if __name__ == "__main__":
    # Test the reader
    import sys

    if len(sys.argv) >= 3:
        task_file = sys.argv[1]
        topo_file = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    else:
        # Default test files
        task_file = "1_task.csv"
        topo_file = "1_topo.csv"
        k = 5

    print(f"Reading {task_file} and {topo_file}...")

    flows, G = read_tsnkit_data(task_file, topo_file, k)

    print(f"\nLoaded {len(flows)} flows:")
    for flow in flows:
        print(f"  Flow {flow['id']}: {flow['src']} -> {flow['dst']}")
        print(f"    Size: {flow['size']}, Period: {flow['period']}")
        print(f"    {len(flow['k_paths'])} paths found")
        for i, path in enumerate(flow['k_paths']):
            print(f"      Path {i}: {path}")

    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
