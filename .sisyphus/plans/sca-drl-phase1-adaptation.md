# SCA-DRL Phase 1 TSNKit 适配改造

## TL;DR

> **Quick Summary**: 将 SCA-DRL Phase 1 算法改造为使用 TSNKit 的数据接口，替换原有的随机数据生成器，并验证功能实现的正确性。
>
> **Deliverables**:
> - `tsnkit/algorithms/sca_drl/phase1_partitioning/get_data_from_tsnkit.py` - TSNKit 数据适配器
> - `tsnkit/algorithms/sca_drl/runners/run_phase1_training.py` - 更新训练脚本使用 TSNKit 数据
> - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py` - 更新推理脚本使用 TSNKit 数据
> - 验修复了先验分数的注释错误
>
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential dependencies
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
用户希望：
1. 对照 Phase 1 功能描述文档，检查现有代码功能实现的正确性
2. 改造数据获取方式，使用 TSNKit 的 `core.load_stream()` 和 `core.load_network()` 替换原有随机数据生成

### Interview Summary

**Key Discussions**:
- 用户提供了 `Phase1 优化版.md` 作为功能规范文档
- 当前代码使用 `TopologyGenerator` 和 `FlowGenerator` 随机生成数据
- `get_data_from_tsnkit.py` 文件为空，需要实现

**Research Findings**:
- 现有 `dataset.py` 中的归一化、边索引构建逻辑**正确**
- `gnn_model.py` 中的 GraphSAGE 编码器和 MLP 预测头实现**正确**
- `train.py` 中的自监督训练逻辑（正负采样 + BCE Loss）**正确**
- `run_phase1_inference.py` 中的谱聚类实现**正确**
- **问题1**: `run_phase1_inference.py:27-35` 中先验分数的注释错误（代码正确，但注释误导）
- **问题2**: 数据未从 TSNKit 加载，而是随机生成

### Metis Review

**Identified Gaps** (addressed):
- Gap 1: `get_data_from_tsnkit.py` 为空 → 提供完整实现
- Gap 2: 先验分数注释误导 → 修复注释
- Gap 3: Training/Inference 脚本未使用 TSNKit 数据 → 提供改造方案

---

## Work Objectives

### Core Objective
将 SCA-DRL Phase 1 完全适配到 TSNKit 框架，确保能使用 TSNKit 标准输入格式进行训练和推理。

### Concrete Deliverables
1. 完整的 TSNKit 数据适配器实现
2. 更新后的训练脚本（使用 TSNKit 数据）
3. 更新后的推理脚本（使用 TSNKit 数据）
4. 修复的先验分数注释

### Definition of Done
- [ ] `get_data_from_tsnkit.py` 包含完整的 TSNKit → NetworkX/Flows 转换逻辑
- [ ] `run_phase1_training.py` 能从 TSNKit CSV 文件加载数据进行
- [ ] `run_phase1_inference.py` 能从 TSNKit CSV 文件加载数据进行推理
- [ ] 先验分数注释已修复
- [ ] 所有脚本可以通过命令行参数指定 TSNKit 数据路径

### Must Have
- 使用 TSNKit 的 `utils.load_stream()` 和 `utils.load_network()`
- 正确处理 TSNKit 的 StreamSet 和 Network 对象
- 正确转换 TSNKit 的 Path 对象为节点 ID 列表
- 保持原有 Phase 1 算法逻辑不变

### Must NOT Have (Guardrails)
- **不要**修改 GNN 模型架构（`gnn_model.py`）
- **不要**修改训练核心逻辑（`train.py` 的训练循环）
- **不要**修改谱聚类逻辑
- **不要**破坏原有的随机数据生成方式（保留作为备选）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest, unittest available)
- **User wants tests**: YES (Tests after implementation)
- **Framework**: pytest
- **QA approach**: Tests-after + Manual verification of data conversion

### Automated Verification

For `get_data_from_tsnkit.py` (using Bash pytest):
```bash
# Agent runs:
pytest tsnkit/algorithms/sca_drl/phase1_partitioning/tests/test_data_adapter.py -v
# Assert: All tests pass (conversion correctness)
```

For training script (using Bash):
```bash
# Agent runs:
cd tsnkit/algorithms/sca_drl
python runners/run_phase1_training.py --task ../../1_task.csv --net ../../1_topo.csv
# Assert: No errors, model checkpoint created
```

For inference script (using Bash):
```bash
# Agent runs:
cd tsnkit/algorithms/sca_drl
python runners/run_phase1_inference.py --task ../../1_task.csv --net ../../1_topo.csv
# Assert: Partition results generated and saved
```

For data conversion (using Bash Python):
```bash
# Agent runs:
python -c "
import sys
sys.path.append('tsnkit')
import os
os.chdir('tsnkit/algorithms/sca_drl')
from phase1_partitioning.get_data_from_tsnkit import load_data_from_tsnkit
flows, graph, max_vals = load_data_from_tsnkit('../../1_task.csv', '../../1_topo.csv', k=5)
print(f'Loaded {len(flows)} flows')
assert len(flows) > 0
assert len(graph.nodes) > 0
assert len(graph.edges) > 0
"
# Assert: Output shows successful loading, no exceptions
```

**Evidence to Capture:**
- [ ] Pytest output showing all tests passing
- [ ] Training script output showing model saved
- [ ] Inference script output showing partition results saved
- [ ] Python data conversion output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Implement get_data_from_tsnkit.py
├── Task 2: Fix prior score comment in run_phase1_inference.py
└── Task 3: Update run_phase1_training.py to use TSNKit data

Wave 2 (After Wave 1):
└── Task 4: Update run_phase1_inference.py to use TSNKit data

Critical Path: Task 1 → Task 3 → Task 4
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 4 | 2 |
| 2 | None | None | 1, 3 |
| 3 | 1 | None | 2 |
| 4 | 1 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2, 3 | delegate_task(category="unspecified-high", load_skills=[], run_in_background=true) |
| 2 | 4 | delegate_task(category="unspecified-high", load_skills=[], run_in_background=false) |

---

## TODOs

- [ ] 1. Implement get_data_from_tsnkit.py

  **What to do**:
  - 实现 `convert_tsnkit_to_nx(network)` 函数：将 TSNKit Network → NetworkX DiGraph
  - 实现 `convert_tsnkit_streams_to_flows(stream_set, network, k=5)` 函数：将 TTSNKit StreamSet → 流列表
  - 实现 `get_max_values(stream_set)` 函数：获取用于归一化的最大值
  - 实现 `load_data_from_tsnkit(task_path, net_path, k=5)` 函数：统一数据加载接口
  - 添加单元测试代码在 `if __name__ == "__main__"` 中

  **Must NOT do**:
  - 修改 TSNKit 的 core 模块
  - 破坏原有的数据结构（保持输出格式与 Phase 1 兼容）

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `unspecified-high`
    - Reason: 涉及多个数据结构转换，需要仔细处理
  - **Skills**: []
    - 理由: 不需要特殊技能，标准 Python 编码

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `tsnkit/algorithms/sca_drl/common/flow_gen.py:40-48` - Flow 字典格式
  - `tsnkit/algorithms/sca_drl/common/topology_gen.py:10-24` - NetworkX 图构建模式
  - `tsnkit/algorithms/sca_drl/adapter.py:66-87` - TSNKit 转换示例代码
  - `tsnkit/core/_network.py:42-95` - Node 和 Link 对象结构
  - `tsnkit/core/_stream.py:102-149` - Stream 对象结构
  - `tsnkit/core/_network.py:373-377` - get_all_path 和 get_shortest_path 方法

  **API/Type References** (contracts to implement against):
  - `tsnkit/core/__init__.py:8-9` - load_stream, load_network 导出

  **Test References** (testing patterns to follow):
  - `tsnkit/core/_network.py:753-821` - Network 类的单元测试示例

  **Documentation References** (specs and requirements):
  - Phase1 优化版.md - 算法功能规范

  **External References** (libraries and frameworks):
  - NetworkX: https://networkx.org/documentation/stable/ - DiGraph 构建和操作

  **WHY Each Reference Matters** (explain the relevance):
  - `flow_gen.py:40-48`: 必须输出完全相同的流字典格式，否则 Phase 1 无法使用
  - `adapter.py:66-87`: 已有 TSNKit 转换示例，可直接参考或扩展
  - `core/_network.py:373-377`: 必须使用 get_all_path 获取 KSP，这是 TSNKit 的标准方法

  **Acceptance Criteria**:

  **If TDD (tests enabled):**
  - [ ] Test file created: tsnkit/algorithms/sca_drl/phase1_partitioning/tests/test_data_adapter.py
  - [ ] Test covers: NetworkX conversion correctness
  - [ ] Test covers: StreamSet to flows conversion
  - [ ] pytest tests pass

  **Automated Verification (ALWAYS include, choose by deliverable type):**

  **For Python module changes** (using Bash pytest):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python phase1_partitioning/get_data_from_tsnkit.py
  # Assert: Output shows "✅ Success!", no exceptions
  # Assert: Loaded N flows, M nodes, K edges
  ```

  **For data verification** (using Bash Python):
  ```bash
  # Agent runs:
  python -c "
import sys
sys.path.append('tsnkit/algorithms/sca_drl')
from phase1_partitioning.get_data_from_tsnkit import load_data_from_tsnkit
flows, graph, _ = load_data_from_tsnkit('../../1_task.csv', '../../1_topo.csv', k=5)
# 验证流格式
assert all('flow_id' in f and 'src' in f and 'dst' in f and 'k_paths' in f for f in flows), 'Flow format incorrect'
# 验证 KSP 存在
assert all(len(f['k_paths']) > 0 for f in flows), 'Empty KSP for some flows'
print('Data format validation passed')
"
  # Assert: Output shows "Data format validation passed"
  ```

  **Evidence to Capture:**
  - [ ] Module execution output showing successful test run
  - [ ] Data validation output showing all assertions passed

  **Commit**: YES
  - Message: `feat(phase1): implement TSNKit data adapter`
  - Files: `tsnkit/algorithms/sca_drl/phase1_partitioning/get_data_from_tsnkit.py`
  - Pre-commit: `pytest phase1_partitioning/tests/test_data_adapter.py` (if test file exists)

---

- [ ] 2. Fix prior score comment in run_phase1_inference.py

  **What to do**:
  - 修改 `run_phase1_inference.py:27-35` 的注释
  - 将注释从 "Score = 1 - GCD(Ti, Tj) / LCM(Ti, Tj)" 改为 "Score = GCD(Ti, Tj) / LCM(Ti, Tj)"
  - 说明这是"谐波对齐度"，值越大表示周期越接近，冲突概率越高

  **Must NOT do**:
  - 修改实际计算逻辑（代码已经是正确的）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的注释修改
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1, Task 3)
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py:27-35` - 先验分数计算函数

  **Acceptance Criteria**:

  **For code comment changes** (using Bash grep):
  ```bash
  # Agent runs:
  grep -n "compute_prior_score" tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py
  # Assert: Shows function definition with corrected comment
  ```

  **Evidence to Capture:**
  - [ ] Grep output showing corrected comment

  **Commit**: YES
  - Message: `fix(phase1): correct prior score comment`
  - Files: `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py`

---

- [ ] 3. Update run_phase1_training.py to use TSNKit data

  **What to do**:
  - 添加命令行参数解析：`--task` 和 `--net`（支持指定 TSNKit CSV 文件路径）
  - 如果提供了 `--task` 和 `--net` 参数，使用 `load_data_from_tsnkit` 加载数据
  - 如果没有提供，保留原有的 `TopologyGenerator` 和 `FlowGenerator` 随机生成逻辑
  - 导入 `get_data_from_tsnkit.py` 中的 `load_data_from_tsnkit` 函数
  - 更新打印日志，显示数据来源（TSNKit 文件或随机生成）

  **Must NOT do**:
  - 删除原有的随机数据生成逻辑（保留作为备选）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要处理两种数据来源，逻辑较复杂
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (but after Task 1 completes)
  - **Blocks**: None
  - **Blocked By**: Task 1 (需要依赖 `get_data_from_tsnkit.py`)

  **References**:

  **Pattern References**:
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_training.py:22-80` - 当前数据加载逻辑
  - `tsnkit/core/_io.py:130` - parse_command_line_args 使用示例

  **Test References**:
  - `tsnkit/algorithms/sca_drl/adapter.py:164` - benchmark 函数的命令行参数处理

  **Acceptance Criteria**:

  **For script modification** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_training.py --help 2>&1 | grep -E "(task|net)"
  # Assert: Shows --task and --net arguments
  ```

  **For TSNKit data loading** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_training.py --task ../../1_task.csv --net ../../1_topo.csv
  # Assert: Output shows "Loading from TSNKit file..."
  # Assert: Training completes and model is saved
  ```

  **For fallback to random data** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_training.py
  # Assert: Output shows "Generating random data..."
  # Assert: Training completes and model is saved
  ```

  **Evidence to Capture:**
  - [ ] Help output showing new arguments
  - [ ] Training output with TSNKit data source
  - [ ] Training output with random data source (fallback)

  **Commit**: YES
  - Message: `feat(phase1): add TSNKit data support to training script`
  - Files: `tsnkit/algorithms/sca_drl/runners/run_phase1_training.py`

---

- [ ] 4. Update run_phase1_inference.py to use TSNKit data

  **What to do**:
  - 添加命令行参数解析：`--task` 和 `--net`（支持指定 TSNKit CSV 文件路径）
  - 如果提供了 `--task` 和 `--net` 参数，使用 `load_data_from_tsnkit` 加载数据
  - 如果没有提供，保留原有的随机数据生成逻辑
  - 导入 `get_data_from_tsnkit.py` 中的 `load_data_from_tsnkit` 函数
  - 更新打印日志，显示数据来源
  - 确保推理脚本能正确处理从 TSNKit 加载的数据格式

  **Must NOT do**:
  - 删除原有的随机数据生成逻辑（保留作为备选）

    **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要处理两种数据来源，逻辑较复杂
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 2)
  - **Blocks**: None
  - **Blocked By**: Task 1 (需要依赖 `get_data_from_tsnkit.py`)

  **References**:

  **Pattern References**:
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py:37-82` - 当前数据加载逻辑
  - `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py:84-95` - 数据集准备逻辑

  **Acceptance Criteria**:

  **For TSNKit data loading** (using Bash):
  ```bash
  # Agent runs:
  cd tsnkit/algorithms/sca_drl
  python runners/run_phase1_inference.py --task ../../1_task.csv --net ../../1_topo.csv
  # Assert: Output shows "Loading from TSNKit file..."
  # Assert: Clustering results shown: Group X: Y flows
  # Assert: Partition results saved to file
  ```

  **For verification with output** (using Bash):
  ```bash
  # Agent runs:
  python -c "
import os
import pickle
result_file = 'tsnkit/algorithms/sca_drl/data/processed/partition_result.pkl'
assert os.path.exists(result_file), 'Partition result file not found'
with open(result_file, 'rb') as f:
    data = pickle.load(f)
assert 'groups' in data, 'Missing groups in partition result'
assert 'original_graph' in data, 'Missing graph in partition result'
print('Partition result validation passed')
"
  # Assert: Output shows "Partition result validation passed"
  ```

  **Evidence to Capture:**
  - [ ] Inference output showing successful clustering
  - [ ] Partition result file created
  - [ ] Validation output showing correct structure

  **Commit**: YES
  - Message: `feat(phase1): add TSNKit data support to inference script`
  - Files: `tsnkit/algorithms/sca_drl/runners/run_phase1_inference.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(phase1): implement TSNKit data adapter` | get_data_from_tsnkit.py | pytest (if tests exist) |
| 2 | `fix(phase1): correct prior score comment` | run_phase1_inference.py | None |
| 3 | `feat(phase1): add TSNKit data support to training script` | run_phase1_training.py | python --help |
| 4 | `feat(phase1): add TSNKit data support to inference script` | run_phase1_inference.py | Run with TSNKit data |

---

## Success Criteria

### Verification Commands
```bash
# Test data adapter
cd tsnkit/algorithms/sca_drl
python phase1_partitioning/get_data_from_tsnkit.py

# Train with TSNKit data
python runners/run_phase1_training.py --task ../../1_task.csv --net ../../1_topo.csv

# Inference with TSNKit data
python runners/run_phase1_inference.py --task ../../1_task.csv --net ../../1_topo.csv
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] get_data_from_tsnkit.py 能正确转换 TSNKit 数据
- [ ] 训练脚本能从 TSNKit CSV 加载数据
- [ ] 推理脚本能从 TSNKit CSV 加载数据
- [ ] 先验分数注释已修复
- [ ] 所有测试通过
