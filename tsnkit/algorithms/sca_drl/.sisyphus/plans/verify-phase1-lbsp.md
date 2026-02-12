# 验证 phase1_lbsp_training.py 可运行性

## TL;DR

> **目标**: 验证 `phase1_lbsp_training.py` 代码是否可以正常运行
> **方法**: 使用 TSNKit 自带的测试数据，创建驱动脚本进行快速验证
>
> **预计时间**: ~2-5 分钟（快速训练验证）
> **并行执行**: 否（顺序执行）

---

## Context

### Original Request
用户希望验证 `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\phase1_lbsp_training.py` 是否可以运行

### 现状分析

**已完成的分析**:
- 所有 Python 依赖项已安装（tsnkit 0.3.0, torch, torch_geometric, numpy, networkx, matplotlib, yaml）
- `phase1_lbsp_training.py` 可以正常导入（无语法错误）
- TSNKit 包自带 260+ 组测试数据文件（位于 `C:\Users\liuya\miniconda3\envs\tsnkit\Lib\site-packages\tsnkit\test\data\`）

**发现的问题**:
1. `phase1_lbsp_training.py` 没有 `if __name__ == "__main__"` 入口，仅包含类和函数定义
2. 脚本需要 TSNKit 格式的 CSV 数据文件（task.csv 和 topo.csv）

**解决方案**:
创建一个验证驱动脚本，导入 `phase1_lbsp_training.py` 中的模块，使用 TSNKit 自带的测试数据进行快速训练验证。

---

## Work Objectives

### Core Objective
创建验证脚本并执行，证明 `phase1_lbsp_training.py` 中的所有组件可以正常工作。

### Concrete Deliverables
- 验证驱动脚本: `verify_phase1_lbsp.py`
- 训练检查点: `gnn_phase1_verification.pth`
- 验证输出日志

### Definition of Done
- [x] 验证脚本创建成功
- [x] 脚本能够加载 TSNKit 测试数据
- [x] Dataset 处理正常（节点、边正确解析）
- [x] GNN 模型初始化成功
- [x] 训练循环运行 50 个 epoch 无错误
- [x] 模型保存成功

### Must Have
- 使用 TSNKit 自带的测试数据（10_task.csv 和 10_topo.csv）
- 快速验证（50 epochs，不追求训练质量）
- 输出详细的验证步骤和结果

### Must NOT Have
- 不进行完整训练（仅验证可运行性）
- 不修改原始 `phase1_lbsp_training.py`
- 不创建额外的测试数据文件

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (Python 环境)
- **User wants tests**: Manual verification (验证脚本本身即为测试)
- **Framework**: None (纯 Python 脚本)

### Automated Verification

验证脚本将：
1. 导入所有必需的模块（phase1_lbsp_training.py, torch, tsnkit）
2. 创建 TSNPhase1Dataset 并加载数据
3. 初始化 GNNPartitionModel
4. 运行 50 个 epoch 的训练
5. 保存模型
6. 输出成功/失败信息

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1:
└── Task 1: 创建验证驱动脚本

Wave 2 (After Wave 1):
└── Task 2: 执行验证脚本并捕获输出
```

---

## TODOs

- [x] 1. 创建验证驱动脚本

  **What to do**:
  - 创建 `verify_phase1_lbsp.py` 文件
  - 导入 `phase1_lbsp_training.py` 中的模块
  - 配置使用 TSNKit 自带的测试数据（10_task.csv, 10_topo.csv）
  - 设置快速训练参数（50 epochs）
  - 添加详细的输出和错误处理

  **Must NOT do**:
  - 不要修改原始 `phase1_lbsp_training.py`
  - 不要进行长时间的完整训练

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件创建，逻辑简单清晰
  - **Skills**: None
    - Reason: 纯 Python 脚本，无需特定技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\phase1_lbsp_training.py:1-275` - 完整的源代码，了解需要导入的类和函数

  **API/Type References**:
  - `TSNPhase1Dataset(task_path, topo_path, config)` - 数据集类构造函数
  - `GNNPartitionModel(input_dim, hidden_dim, output_dim)` - GNN 模型构造函数
  - `train_phase1(model, dataset, config, device)` - 训练函数

  **Documentation References**:
  - TSNKit 测试数据路径: `C:\Users\liuya\miniconda3\envs\tsnkit\Lib\site-packages\tsnkit\test\data\10_task.csv`
  - TSNKit 测试数据路径: `C:\Users\liuya\miniconda3\envs\tsnkit\Lib\site-packages\tsnkit\test\data\10_topo.csv`

  **Acceptance Criteria**:

  **Automated Verification**:
  ```python
  # Agent runs:
  python -c "import verify_phase1_lbsp"
  # Assert: Module imports successfully without error

  # Agent runs:
  python verify_phase1_lbsp.py
  # Assert: Script completes without exception
  # Assert: Output contains "SUCCESS: Training completed successfully!"
  ```

  **Evidence to Capture**:
  - [ ] 脚本文件创建成功
  - [ ] 脚本完整输出（终端输出）

  **Commit**: NO

- [x] 2. 执行验证脚本并捕获输出

  **What to do**:
  - 运行 `python verify_phase1_lbsp.py`
  - 捕获完整输出
  - 检查是否有错误
  - 验证训练检查点文件是否创建

  **Must NOT do**:
  - 如果脚本失败，不要继续执行

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单命令执行，简单验证
  - **Skills**: None
    - Reason: 纯 Bash 命令

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\verify_phase1_lbsp.py` - 刚创建的验证脚本

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Agent runs:
  python verify_phase1_lbsp.py 2>&1 | tee verification_output.txt
  # Assert: Exit code is 0
  # Assert: Output contains "Dataset created successfully!"
  # Assert: Output contains "Model created successfully!"
  # Assert: Output contains "Training completed successfully!"
  # Assert: Output contains "phase1_lbsp_training.py is ready to run!"

  # Agent runs:
  ls -lh gnn_phase1_verification.pth
  # Assert: File exists and size > 0
  ```

  **Evidence to Capture**:
  - [ ] 完整的终端输出（保存到 verification_output.txt）
  - [ ] gnn_phase1_verification.pth 文件是否存在
  - [ ] 训练过程中的 epoch 输出（确认训练实际运行）

  **Commit**: NO

---

## Success Criteria

### Verification Commands
```bash
python verify_phase1_lbsp.py
# Expected: 完整输出，包含成功消息
```

### Final Checklist
- [x] 验证脚本执行完成
- [x] 无 Python 错误或异常
- [x] Dataset 加载成功（节点数 > 0）
- [x] 模型训练完成（50 epochs）
- [x] 检查点文件保存成功
- [x] 输出包含 "phase1_lbsp_training.py is ready to run!"

### Expected Output Structure

```
=== Phase 1 LBSP Training Verification ===
Device: cpu

Task file: ...\10_task.csv
Topo file: ...\10_topo.csv
Files exist: True

=== Step 1: Creating Dataset ===
[Phase1] Loading TSNKit data from ...
[Phase1] Graph Created: Nodes(Streams)=10, Conflict Edges=X
Dataset created successfully!

=== Step 2: Creating Model ===
Model created successfully!
Model parameters: XXXX

=== Step 3: Starting Training (Quick Verification) ===
--- Start Training Phase 1 ---
Epoch 0, Loss: X.XXXX
Epoch 10, Loss: X.XXXX
Epoch 20, Loss: X.XXXX
Epoch 30, Loss: X.XXXX
Epoch 40, Loss: X.XXXX
Model saved to gnn_phase1_verification.pth

[SUCCESS] Training completed successfully!

=== Verification Summary ===
✅ All components verified:
   - TSNKit data loading: OK
   - Dataset processing: OK
   - Model initialization: OK
   - Training loop: OK
   - Model saving: OK

phase1_lbsp_training.py is ready to run!
```
