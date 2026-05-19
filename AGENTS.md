# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**TSNKit** (v0.3.0) is an open-source scheduling and benchmarking toolkit for Time-Sensitive Networking (TSN), implementing IEEE 802.1Qbv scheduling algorithms. Published at RTAS 2024. The active development branch (`refactor-phase1-phase2`) adds a two-stage Deep Reinforcement Learning scheduler (`sca_drl`) on top of the existing classical solvers.

## Setup & Installation

```bash
# Install from source (standard)
pip install -e .

# If using Cython acceleration for simulation
pip install -r requirements-cython.txt
python setup.py build_ext --inplace
```

Dependencies include `z3-solver`, `gurobipy` (commercial), `docplex`/CP Optimizer (commercial), `networkx`, `numpy`, `pandas`, `torch` (for sca_drl). Some algorithms require commercial solver licenses (Gurobi, CPLEX).

## Common Commands

```bash
# Generate synthetic datasets
python -m tsnkit.data.generator

# Run a single algorithm (e.g., LS list-scheduler)
python -m tsnkit.algorithms.ls <task.csv> <topo.csv>

# Validate simulation on a scheduled result
python -m tsnkit.simulation.tas <task.csv> <output_dir/>

# Run benchmark across algorithms and instances
python -m tsnkit.test.benchmark --methods ALL --ins 1-16

# Run validation tests (CI gates)
python tsnkit/test/debug/_validate_classes.py
python tsnkit/test/debug/_validate_io.py
python tsnkit/test/debug/_validate_simulator.py

# SCA-DRL Phase 1 — train GNN partitioner
python tsnkit/algorithms/sca_drl/phase1/train.py

# SCA-DRL Phase 1 — batch inference (produces embeddings for Phase 2 dataset)
python tsnkit/algorithms/sca_drl/phase1/batch_infer.py

# SCA-DRL Phase 2 — train PPO agent
python tsnkit/algorithms/sca_drl/phase2/train.py   # (entry point varies by run config)
```

## Architecture

### Standard Algorithm Interface

All 19 classical schedulers (`tsnkit/algorithms/*.py`) share the same four-method contract:

```python
class AlgorithmName:
    def init(task_path, net_path) -> None   # parse CSV inputs
    def prepare() -> None                   # build solver constraints
    def solve() -> Statistics               # run solver, return result + timing
    def output() -> Config                  # emit GCL/route/queue assignments

def benchmark(name, task_path, net_path, output_path, workers) -> Statistics
```

The `benchmark()` function is the uniform entry point used by the test harness.

### Core Data Model (`tsnkit/core/`)

- **`_network.py`** — `Node` (switch/end-station), `Link` (with rate, propagation delay, queue count), `Network` (graph with shortest/all paths), `Path`; loaded via `load_network()` from topology CSV.
- **`_stream.py`** — `Stream` (src, dst, size, period, deadline, jitter, routing path), `StreamSet` (collection with routing management); loaded via `load_stream()` from task CSV.
- **`_config.py`** — Output format: `GCL` (gate control list), `Route`, `Release`, `Queue`, `Delay`, `Size`. All algorithms produce a `Config` object that serialises to `*--GCL.csv`, `*--ROUTE.csv`, etc.
- **`_io.py`** — `Result` enum (`schedulable`, `unschedulable`, `unknown`, `error`), `Statistics` (algo_time, total_time, algo_mem, total_mem).
- **`_constants.py`** — Global constants: `T_SLOT=1000 ns`, `T_PROC`, `T_LIMIT=7200 s`, `MAX_NUM_QUEUE=8`.

### SCA-DRL Two-Stage Scheduler (`tsnkit/algorithms/sca_drl/`)

Active development area. Two-stage pipeline:

**Phase 1 — GNN Stream Partitioning** (`phase1/`)
- `model.py`: `GNNPartitionModel` — two-layer GAT (Graph Attention Network) that embeds streams using topology context; outputs are clustered with spectral clustering.
- `train.py` / `infer.py` / `batch_infer.py`: training, single inference, and batch inference pipelines.
- Checkpoints saved under `phase1/models/` (timestamped).

**Phase 2 — Transformer-PPO Scheduling** (`phase2/`)
- `agent.py`: `CongestionAwareTransformer` — Actor-Critic PPO agent with mask-aware attention for variable-length stream sets; schedules each partition produced by Phase 1.
- Training runs saved in `phase2/runs/`, model checkpoints in `phase2/models/`.

**Integration** (`sca_drl.py` at `tsnkit/algorithms/sca_drl.py`)
- `SCADRLScheduler` wraps both phases behind the standard `init/prepare/solve/output` interface so it is a drop-in replacement for classical algorithms in benchmarks.

**Shared utilities** (`common/utils.py`, `common/config.py`), datasets (`data/`), and documentation (`doc/`) live alongside the phase directories.

### Simulation (`tsnkit/simulation/`)

- `tas.py`: Time-Aware Shaper simulator — replays a scheduled `Config` frame-by-frame, verifies deadline compliance and queue isolation. Uses `match_time()` binary search (Cython-accelerated via `cython/simulation_core.pyx` when compiled).
- Simulation is used both for validation post-scheduling and in Phase 2 reward computation.

### Data Generation (`tsnkit/data/`)

- `generator.py`: `DatasetGenerator` — combinatorial CSV dataset generation (task + topology pairs).
- `dataset_spec.py`: named topology generators (tree, mesh, fat-tree, etc.).

## CI Pipeline (`.github/workflows/workflow.yml`)

Four gates run on every PR: `validate_classes`, `validate_io`, `validate_simulator`, `validate_algorithms`. Algorithm validation runs inside a Docker container with CP Optimizer and compares changed algorithms against baseline methods (`jrs_nw`, `ls`, `smt_wa`).

## Key Conventions

- Input: two CSVs — `*_task.csv` (streams) and `*_topo.csv` (network topology).
- Output: multiple CSVs per algorithm run (`--GCL.csv`, `--ROUTE.csv`, `--QUEUE.csv`, `--RELEASE.csv`).
- All timing is in nanoseconds (`T_SLOT = 1000 ns`).
- Algorithms that use commercial solvers (Gurobi → `jrs_*`, CPLEX → `cp_wa`, `jrs_mc`) will fail without valid licenses.
- SCA-DRL code contains mixed Chinese/English comments; this is intentional (research team convention).
- Mypy is configured (`.mypy.ini`); run `mypy tsnkit/` for type checking.

## SCA-DRL 实验代码结构

### 关键路径
- Phase 1 入口：`tsnkit/algorithms/sca_drl/phase1/`
- Phase 2 入口：`tsnkit/algorithms/sca_drl/phase2/`
- 配置文件：`sca_drl/configs/`
- TensorBoard 日志：`phase2/runs/` 下按时间戳命名
- 流/拓扑参数：硬编码在 `data_generator.py` 中，尚未抽离

### 已确认状态（2026-05-08）
- TensorBoard 已记录 reward / success rate ✅
- Phase 1/2 调用入口已确认 ✅
- 参数未抽离，修改需直接改 data_generator.py ⚠️

### 硬性约束
- 方法名固定：SCA-DRL
- 当前拓扑：Fig.9 简化版（9 SW+27 ES）
- 不改 Phase 1/2 核心逻辑，只加日志和 baseline
- 改代码前先报思路，等确认
- TODO 一次不超过 5 条

### 当前拓扑：Fig.9 简化版
- 9 个交换机（SW 0~8）+ 27 个端系统（ES 9~35）
- SW 连接：环形 0-1-2-3-4-5-6-7-8-0，加弦边 (2,7) (3,6)
- ES 连接：每个 SW 外挂 3 个 ES（3 跳直连）
- 节点总数：36，实现在 data_generator.py
### _meta 文件位置（论文背景）
C:\Users\1\Nutstore\1\Obsidian_vault\Zettelkasten\_meta 

### 代码背景
- 项目基于 tsnkit（https://github.com/ChuanyuXue/tsnkit）fork 而来
- tsnkit 是多算法测试床，包含 19 个经典调度算法，不要修改
- SCA-DRL 是新增算法，位于 `tsnkit/algorithms/sca_drl/`
- Phase2 环境依赖 tsnkit 原生的 `tsnkit/algorithms/ls.py`（链路调度基础层）
- 对比实验可直接调用 tsnkit 中已有的经典算法，无需重新实现