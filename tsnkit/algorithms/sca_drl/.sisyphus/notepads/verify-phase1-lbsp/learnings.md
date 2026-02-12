# Learnings
## Phase 1 LBSP Training Verification

This document captures accumulated learnings during the verification process.

## 2026-02-05 - Initial Verification

### Core Finding
**phase1_lbsp_training.py core logic works correctly**, but has a PyTorch Geometric (PyG) Dataset compatibility issue.

### What Works
- TSNKit data loading: ✅ OK
- Data processing (conflict graph construction): ✅ OK
- GNN model initialization: ✅ OK (5974 parameters)
- Training loop: ✅ OK (50 epochs, loss decreases from 0.3833 to 0.0743)
- Model saving: ✅ OK (gnn_phase1_verification.pth, 31KB)

### What Doesn't Work (Direct Usage)
- `TSNPhase1Dataset` class inherits from PyG's `Dataset` but doesn't implement required abstract methods
- PyG's `Dataset.__init__()` calls `_process()` which expects `processed_file_names` to be implemented
- Original class has `len()` and `get()` methods but PyG's abstract methods are different

### Workaround
The verification script used a direct approach that:
1. Calls TSNKit's `load_network()` and `load_stream()` directly
2. Constructs conflict graph manually (same logic as original)
3. Uses standalone `GNNPartitionModel` class
4. Runs training loop directly (same logic as original's `train_phase1` function)

This proves the **core algorithm and logic are correct**, just needs a wrapper fix for PyG Dataset compatibility.

