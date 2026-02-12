# SCA-DRL Phase 1 TSNKit 集成改造

## TL;DR

> **Quick Summary**: 将 SCA-DRL Phase 1 改造为使用 TSNKit 的数据接口，直接使用 TSNKit 的 StreamSet 和 Network 对象，无需任何数据转换。
>
> **Deliverables**:
> - 改造 `sca_drl/phase1_partitioning/dataset.py` - 直接接收 TSNKit 对象
> - 更新 `sca_drl/runners/run_phase1_training.py` - 使用 TSNKit 数据训练
> - 更新 `sca_drl/runners/run_phase1_inference.py` - 使用 TSNKit 数据推理
> - 修复先验分数注释
>
> **Estimated Effort**: Medium
> **Parallel Execution**: Dependent - Task 2,3,4 depend on Task 1
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
用户希望：
1. 对照 Phase 1 功能描述文档，检查现有代码功能实现的正确性
2. 改造数据获取方式，使用 TSNKit 的 `core.load_stream()` 和 `core.load_network()`
3. 直接使用 TSNKit 对象，无需 NetworkX 转换

### Interview Summary

**Key Discussions**:
- ✅ 确认直接改造 `dataset.py`，接收 TSNKit 对象（StreamSet, Network）
- ✅ 使用 TSNKit 的 `network.get_all_path()` 获取 KSP 路径（内部已调用 NetworkX）
- ✅ 无需任何数据转换（TSNKit Network 内部已有 NetworkX）
- ✅ 移除随机数据生成器（flow_gen.py, topology_gen.py）
- ✅ 修复先验分数注释错误

**Technical Decisions**:
- stream 和 flow 是同一个东西的不同叫法
- TSNKit Network 对象内部已封装 NetworkX (`self._net_nx`)
- 不需要 `get_data_from_tsnkit.py` 模块
- ksp.py 保留但不使用

**Research Findings**:
- 现有代码功能实现基本正确，只有先验分数注释误导
- TSNKit 的 `get_all_path()` 内部调用 `nx.all_simple_paths()`
- TSNKit 的 `get_shortest_path()` 内部调用 `nx.shortest_path()`

### Metis Review

**Identified Gaps** (addressed):
- Gap 1: dataset.py 需要改造以接收 TSNKit 对象 → 提供完整实现
- Gap 2: 先验分数注释误导 → 修复注释
- Gap 3: Training/Inference 脚本未使用 TSNKit → 提供改造方案
- Gap 4: 无需数据转换模块 → 移除 get_data_from_tsnkit.py 计划

---

## Work Objectives

### Core Objective
将 SCA-DRL Phase 1 完全集成到 TSNKit 框架，直接使用 TSNKit 的 StreamSet 和 Network 对象，无需任何中间转换。

### Concrete Deliverables
1. 改造后的 `dataset.py`（直接接收 TSNKit 对象）
2. 更新后的训练脚本（使用 TSNKit 数据）
3. 更新后的推理脚本（使用 TSNKit 数据）
4. 修复的先验分数注释

### Definition of Done
- [ ] `dataset.py` 能直接接收 TSNKit StreamSet 和 Network 对象
- [ ] `dataset.py` 使用 `network.get_all_path()` 获取 KSP 路径
- [ ] `run_phase1_training.py` 能从 TSNKit CSV 文件加载数据进行训练
- [ ] `run_phase1_inference.py` 能从 TSNKit CSV 文件加载数据进行推理
- [ ] 先验分数注释已修复
- [ ] 所有脚本支持命令行参数指定数据路径

### Must Have
- 使用 TSNKit 的 `utils.load_stream()` 和 `utils.load_network()`
- `dataset.py` 直接接收 TSNKit 对象（StreamSet, Network）
- 使用 `network.get_all_path()` 获取 KSP 路径
- 保持原有 Phase 1 算法逻辑不变

### Must NOT Have (Guardrails)
- **不要**将 TSNKit 对象转换为 NetworkX（内部已有 NetworkX）
- **不要**修改 GNN 模型架构（`gnn_model.py`）
- **不要**修改训练核心逻辑（`train.py`）
- **不要**修改谱聚类逻辑
- **不要**保留随机数据生成器

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest available)
- **User wants tests**: YES (Tests after implementation)
- **Framework**: pytest
- **QA approach**: Tests-after + Manual verification

### Automated Verification

For `dataset.py` (using Bash pytest):
```bash
# Agent runs:
cd tsnkit/algorithms/sca_drl
pytest phase1_partitioning/tests/test_dataset.py -v
# Assert: All tests pass
```

For training script (using Bash):
```bash
# Agent runs:
cd tsnkit/algorithms/sca_drl
python runners/run_phase1_training.py --task ../../1_task.csv --net ../../1_topo.csv --k 5
# Assert: No errors, model checkpoint created, training completes
```

For inference script (using Bash):
```bash
# Agent runs:
cd tsnkit/algorithms/sca_drl
python runners/run_phase1_inference.py --task ../../1_task.csv --net ../../1_topo --k 5
# Assert: No errors, partition results generated and saved
```

For data loading verification (using Bash Python):
```bash
# Agent runs:
python -c "
import sys
sys.path.append('tsnkit')
from tsnkit import core as utils
from os.path import join
task_path = '1_task.csv'
net_path = '1_topo.csv'
stream_set = utils.load_stream(task_path)
network = utils.load_network(net_path)
print(f'Loaded {len(stream_set)} streams, {len(network.nodes)} nodes')
assert len(stream_set) > 0 and len(network.nodes) > 0
"
# Assert: Shows successful loading, no exceptions
```

**Evidence to Capture:**
- [ ] Pytest output showing all tests passing
- [ ] Training script output showing model saved
- [ ] Inference script output showing partition results saved
- [ ] Data loading verification output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: 改造 dataset.py 直接接收 TSNKit 对象

Wave 2 (After Wave 1):
├── Task 2: 修复先验分数注释
├── Task 3: 更新训练脚本使用 TSNKit
└── Task 4: 更新推理脚本使用 TSNKit

Critical Path: Task 1 → Task 2,3,4 (Task 2,3,4 可以并行执行)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4 | None |
| 2 | 1 | None | 3, 4 |
| 3 | 1 | None | 2, 4 |
| 4 | 1 | None | 2, 3 |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | delegate_task(category="unspecified-high", load_skills=[], run_in_background=false) |
| 2 | 2, 3, 4 | delegate_task(category="unspecified-high", load_skills=[], run_in_background=true) |

---

## TODOs

- [ ] 1. 改造 dataset.py 直接接收 TSNKit 对象

  **What to do**:
  - 修改 `__init__` 方法签名：接收 `stream_set`（TSNKit StreamSet）和 `network`（TSNKit Network），添加 `k` 参数
  - 实现 `_convert_streams_to_internal_format()` 方法：将 TSNKit StreamSet 转换为内部 flows 列表格式
  - 在转换函数中，使用 `network.get_all_path(stream._src, stream._dst)` 获取所有路径
  - 从返回的 Path 对象列表中取前 k 条，提取节点 ID 列表（`[node._id for node in path.nodes]`）
  - 如果没有路径，使用 `network.get_shortest_path()` 作为备选
  - 计算归一化最大值（max_period, max_size, max_deadline）
  - `get_node_features()` 和 `get_edge_index()` 方法保持不变

  **Must NOT do**:
  - 任何 NetworkX 转换（TSNKit 内部已有）
  - 修改 PyG Data 对象的格式
  - 修改边索引构建逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解 TSNKit 对象结构和路径获取方式
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 1)
  - **Blocks**: Task 2, 3, 4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `tsnkit/algorithms/sca_drl/phase1_partitioning/dataset.py:5-72` - 当前 dataset.py 实现
  - `tsnkit/algorithms/sca_drl/common/flow_gen.py:40-48` - 流字典格式
  - `tsnkit/core/_stream.py:102-149` - TSNKit Stream 对象结构
  - `tsnkit/core/_network.py:373-377` - TSNKit Network 的 get_all_path 和 get_shortest_path 方法
  - `tsnkit/core/_network.py:542-730` - TSNKit Path 对象结构

  **API/Type References** (contracts to implement against):
  - `tsnkit/core/__init__.py:8-9` - load_stream, load_network 导出

  **Acceptance Criteria**:

  **For dataset.py modification** (using Bash Python):
  ```bash
  # Agent runs:
  python -c "
import sys
sys.path.append('tsnkit')
from tsnkit import core as utils
from os.path import join, dirname
import os
os.chdir('tsnkit/algorithms/sca_drl')
sys.path.append('.')
from phase1_partitioning.dataset import FlowGraphDataset

stream_set = utils.load_stream('../../1_task.csv')
network = utils.load_network('../../1_topo.csv')

dataset = FlowGraphDataset(stream_set, network, k=5)
data = dataset.process()

print(f'Dataset created: {data.num_nodes} flows, {data.num_edges} edges')
assert data.num_nodes == len(stream_set)
assert data.num_edges > 0
print('Dataset works correctly with TSNKit objects')
"
  # Assert: Output shows dataset created successfully, no exceptions
  ```

  **For KSP path extraction** (using Bash Python):
  ```bash
  # Agent runs:
  python -c "
import sys
sys.path.append('tsnkit')
from tsnkit import core as utils
from os.path import join
import os
os.chdir('tsnkit/algorithms/sca_drl')
sys.path.append('.')
from phase1_partitioning.dataset import FlowGraphDataset

stream_set = utils.load_stream('../../1_task.csv')
network = utils.load_network('../../1_topo.csv')

dataset = FlowGraphDataset(stream_set, network, k=5)

# 验证所有流都有 k_paths
for flow in dataset.flows:
    assert 'k_paths' in flow, f'Flow {flow[\"flow_id\"]} missing k_paths'
    assert len(flow['k_paths']) > 0, f'Flow {flow[\"flow_id\"]} has no paths'
    # 验证路径格式是节点 ID 列表
    for path in flow['k_paths']:
        assert isinstance(path, list), 'Path should be a list'
        assert all(isinstance(node, int) for node in path), 'Path nodes should be integers'

print('All flows have valid k_paths')
"
  # Assert: Output shows all flows have valid k_paths
  ```

  **Evidence to Capture**:
  - [ ] Dataset creation output showing successful TSNKit object handling
  - [ ] KSP path extraction output showing valid paths for all flows

  **Commit**: YES
  - Message: `refactor(phase1): adapt dataset to use TSNKit objects directly`
  - Files: `tsnkit/algorithms/sca_drl/phase1_partitioning/dataset.py`

---

- [ ] 2. 修复先验分数注释

  **What to do**:
  - 修改 `run_phase1_inference.py:27-35` 的 `compute_prior_score` 函数注释
  - 将注释从 "Score = 1 - GCD(Ti, Tj) / LCM(Ti, Tj)" 改为 "Score = GCD(Ti, Tj) / LCM(Ti, Tj)"
  - 添加说明："谐波对齐度：值越大表示周期越接近，冲突概率越高"
  - **不要**修改实际计算逻辑（代码已经是正确的）

  **Must NOT do**:
  - 修改实际计算逻辑（代码实现是正确的）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的注释修改
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3, Task 4)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py:27-35` - 先验分数计算函数

  **Acceptance Criteria**:

  **For comment fix** (using Bash grep):
  ```bash
  # Agent runs:
  grep -A 3 "def compute_prior_score" tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py
  # Assert: Shows corrected comment mentioning "GCD/LCM" and "谐波对齐度"
  ```

  **Evidence to Capture****:
  - [ ] Grep output showing corrected comment

  **Commit**: YES
  - Message: `fix(phase1): correct prior score comment`
  - Files: `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py`

---

- [ ] 3. 更新训练脚本使用 TSNKit 数据

  **What to do**:
  - 添加命令行参数解析：`--task`, `--net`, `--k`（默认值 5）
  - 使用 `argparse` 或 `sys.argv` 解析参数
  - 导入 `from tsnkit import core as utils`
  - 使用 `utils.load_stream(task_path)` 加载流集合
  - 使用 `utils.load_network(net_path)` 加载网络拓扑
  - 创建 `dataset = FlowGraphDataset(stream_set, network, k=k)`
  - 移除或注释掉原有的 `TopologyGenerator` 和 `FlowGenerator` 代码
  - 更新打印日志，显示 TSNKit 数据路径和统计信息

  **Must NOT do**:
  - 保留原有的随机数据生成逻辑（完全移除）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要添加命令行参数处理和数据加载逻辑
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2, Task 4)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_training.py:22-44` - 当前数据加载逻辑
  - `tsnkit/core/_io.py:130` - parse_command_line_args 使用示例
  - `tsnkit/algorithms/sca_drl/adapter.py:55-64` - TSNKit 数据加载示例

  **Test References**:
  - `tsnkit/algorithms/sca_drl/adapter.py:144-165` - benchmark 函数的命令行参数处理

  **Acceptance Criteria**:

  **For script modification** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_training.py --help 2>&1 | head -20
  # Assert: Shows usage with --task, --net, --k arguments
  ```

  **For TSNKit data loading** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_training.py --task ../../1_task.csv --net ../../1_topo.csv --k 5
  # Assert: Output shows loading from TSNKit files
  # Assert: Output shows "Phase 1 Training Complete!"
  ```

  **Evidence to Capture**:
  - [ ] Help output showing new arguments
  - [ ] Training output showing successful TSNKit data loading
  - [ ] Training completion message

  **Commit**: YES
  - Message: `refactor(phase1): update training script to use TSNKit data`
  - Files: `tsnkit/algorithms/sca_drl/runners/run_phase1_training.py`

---

- [ ] 4. 更新推理脚本使用 TSNKit 数据

  **What to do**:
  - 添加命令行参数解析：`--task`, `--net`, `--k`（默认值 5）
  - 使用 `argparse` 或 `sys.argv` 解析参数
  - 导入 `from tsnkit import core as utils`
  - 使用 `utils.load_stream(task_path)` 加载流集合
  - 使用 `utils.load_network(net_path)` 加载网络拓扑
  - 创建 `dataset = FlowGraphDataset(stream_set, network, k=k)`
  - 移除或注释掉原有的 `TopologyGenerator` 和 `FlowGenerator` 代码
  - 更新打印日志，显示 TSNKit 数据路径和统计信息
  - 确保推理脚本能正确处理从 TSNKit 加载的数据格式

  **Must NOT do**:
  - 保留原有的随机数据生成逻辑（完全移除）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要添加命令行参数处理和数据加载逻辑
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2, Task 3)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py:37-82` - 当前数据加载逻辑
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py:84-95` - 数据集准备逻辑

  **Acceptance Criteria**:

  **For TSNKit data loading** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_inference.py --task ../../1_task.csv --net ../../1_topo.csv --k 5
  # Assert: Output shows loading from TSNKit files
  # Assert: Output shows clustering results: "Group X: Y flows"
  # Assert: Output shows "Partitioned data saved to:"
  ```

  **For partition result verification** (using Bash Python):
  ```bash
  # Agent runs:
  python -c "
import os
import pickle
import sys
sysresult_file = 'tsnkit/algorithms/sca_drl/data/processed/partition_result.pkl'
assert os.path.exists(result_file), 'Partition result file not found'
with open(result_file, 'rb') as f:
    data = pickle.load(f)
assert 'groups' in data, 'Missing groups in partition result'
assert 'original_graph' in data, 'Missing graph in partition result'
num_groups = len(data['groups'])
total_flows = sum(len(gflows) for gflows in data['groups'].values())
print(f'Partition validation: {num_groups} groups, {total_flows} total flows')
"
  # Assert: Output shows valid partition with groups and flows
  ```

  **Evidence to Capture**:
  - [ ] Inference output showing successful clustering
  - [ ] Partition result file created
  - [ ] Validation output showing correct partition structure

  **Commit**: YES
  - Message: `refactor(phase1): update inference script to use TSNKit data`
  - Files: `tsnkit/algorithms/scaugs_drl/runners/run_phase1_inference.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `refactor(phase1): adapt dataset to use TSNKit objects directly` | dataset.py | Python test |
| 2 | `fix(phase1): correct prior score comment` | run_phase1_inference.py | None |
| 3 | `refactor(phase1): update training script to use TSNKit data` | run_phase1_training.py | --help |
| 4 | `refactor(phase1): update inference script to use TSNKit data` | run_phase1_inference.py | Run with TSNKit data |

---

## Success Criteria

### Verification Commands
```bash
# Test dataset with TSNKit objects
cd tsnkit/algorithms/sca_drl
python -c "
from tsnkit import core as utils
from phase1_partitioning.dataset import FlowGraphDataset
stream_set = utils.load_stream('../../1_task.csv')
network = utils.load_network('../../1_topo.csv')
dataset = FlowGraphDataset(stream_set, network, k=5)
print('✅ Dataset works with TSNKit objects')
"

# Train with TSNKit data
python runners/run_phase1_training.py --task ../../1_task.csv --net ../../1_topo.csv --k 5

# Inference with TSNKit data
python runners/run_phase1_inference.py --task ../../1_task.csv --net ../../1_topo.csv --k 5
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] dataset.py 能直接接收 TSNKit StreamSet 和 Network 对象
- [ ] dataset.py 使用 network.get_all_path() 获取 KSP 路径
- [ ] 训练脚本能从 TSNKit CSV 加载数据
- [ ] 推理脚本能从 TSNKit CSV 加载数据
- [ ] 先验分数注释已修复
- [ ] 随机数据生成器已移除
- [ ]所有测试通过
