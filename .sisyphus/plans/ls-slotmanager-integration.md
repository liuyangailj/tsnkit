# Plan: LS-based SlotManager for PPO Environment (Refined)

## TL;DR

> **Quick Summary**: Integrate ls.py's `find_inject_time` algorithm into PPO environment's SlotManager. SlotManager uses tsnkit.core objects (Stream, Path, Link) and provides two-step interface (try_schedule + commit_schedule) for PPO-based flow scheduling.
>
> **Deliverables**:
> - Refactored SlotManager with ls-based `find_inject_time`, `match_time`, `get_nw_delay` methods
> - TSNSchedulingEnv enhanced with tsnkit object construction (Network, Stream, Path)
> - Modified step() function using new SlotManager interface
> - Unit tests verifying consistency with ls.py
>
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential (SlotManager changes must come before Environment changes)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
将原生ls.py的时间槽分配逻辑集成到PPO环境的SlotManager中，让PPO负责流顺序和路由路径选择，ls负责底层的时间槽分配（find_inject_time、GCL维护等）。

**Refined Understanding (User's Precise Input)**:
- SlotManager仅需要调用ls.py中的find_inject_time及其依赖模块
- 接口部分需要较大改动以适配
- 需要使用tsnkit.core模块来加载路径、流等信息
- 用户确认：SlotManager需要完整接口（try_schedule + commit_schedule）

### Interview Summary
**Key Discussions**:
- **Scope Focus**: Only copy/adapt `find_inject_time`, `match_time`, `get_nw_delay` from ls.py
- **Interface Design**: Two-step design confirmed: `try_schedule(stream, path) -> int` + `commit_schedule(stream, path, inject_time)`
- **Data Transfer**: Use tsnkit objects directly (Stream, Path, Network) instead of dictionary formats
- **GCL Update**: Include GCL update logic (copied from ls.py schedule method lines 208-224)
- **State Persistence**: Support save/load state for cross-episode resource sharing

### Research Findings
**From ls.py Analysis**:
- **match_time()** (lines 99-132): Static method, binary search for conflict detection
- **get_nw_delay()** (lines 134-136): Calculates `sum(l.t_proc + stream.get_t_trans(l))`
- **find_inject_time()** (lines 138-185): Core algorithm, depends on:
  - `get_nw_delay(s, path)`
  - `Stream.get_t_trans(link)`
  - `Stream.get_frame_indexes(lcm)`
  - `self._result[l]` (GCL data structure)
  - `match_time(t, sche)`
- **GCL Structure**: `Dict[Link, List[(start, end, queue)]]`
- **GCL Update Pattern** (lines 208-224): For each link, for each frame index in LCM, append (start, end, queue=0)

**From tsnkit.core Analysis**:
- `Stream.get_t_trans(link) -> int`: Calculate transmission time on specific link
- `Stream.get_frame_indexes(lcm) -> List[int]`: Return frame index list `[0, 1, ..., num_frames-1]`
- `Stream.period`: Period attribute (in time slots)
- `Link.t_proc`: Processing time attribute
- `StreamSet.lcm`: LCM of all stream periods

### Metis Review
**Identified Gaps & Recommendations**:

**Critical Design Decisions (Self-Resolved)**:
- **Error Contract**: `try_schedule` returns `int` (inject_time), with `-1` indicating unschedulable (consistent with ls.py)
- **GCL Ownership**: SlotManager owns and manages GCL internally (`self._result`)
- **Transaction Safety**: `try_schedule` does NOT modify GCL (read-only check), `commit_schedule` applies changes
- **State Ownership**: GCL is stored in SlotManager, but can be serialized for cross-episode sharing

**Guardrails Applied**:
- ❌ NO modifications to original ls.py - only copy/adapt methods
- ❌ NO breaking changes to PPO Agent - only Environment changes
- ❌ NO mock allocation logic - all allocation must use ls algorithms
- ❌ NO hardcoded path selection - PPO must decide paths
- ❌ NO changes to ls.py's task_order logic - PPO handles ordering
- ❌ NO partial state corruption on commit_schedule failure - use rollback if needed
- ❌ NO try_schedule modifying GCL - read-only check only

**Edge Cases Addressed**:
- Empty GCL: `match_time` returns -1 (handled by ls.py logic)
- Fully saturated GCL: `find_inject_time` returns -1 (handled by ls.py logic)
- Single-link paths: Algorithm works correctly (generic loop handles this)
- commit without try: Should gracefully handle (document as error case)

---

## Work Objectives

### Core Objective
Integrate ls.py's `find_inject_time` algorithm into PPO environment's SlotManager, enabling RL-based scheduling decisions while maintaining TSNKit's proven GCL management.

### Concrete Deliverables
- `SlotManager` class with:
  - `match_time(t, sche)` - static method (binary search)
  - `get_nw_delay(stream, path)` - network delay calculation
  - `find_inject_time(stream, path)` - core allocation algorithm
  - `try_schedule(stream, path) -> int` - check if schedulable, return inject_time or -1
  - `commit_schedule(stream, path, inject_time)` - update GCL
  - `save_state() -> Dict` - serialize GCL
  - `load_state(state, network)` - restore GCL
  - `clear()` - reset GCL
- `TSNSchedulingEnv.__init__()` enhanced with tsnkit object construction
- `TSNSchedulingEnv.step()` modified to call new SlotManager interface
- `test_slotmanager.py` with unit tests comparing against ls.py

### Definition of Done
- [ ] SlotManager successfully allocates slots using ls's `find_inject_time()` logic
- [ ] GCL updates match ls.py's behavior exactly
- [ ] State can be saved and restored across episodes
- [ ] Unit tests pass with 100% consistency with ls.py
- [ ] PPO environment trains without errors using new SlotManager
- [ ] No modifications to original ls.py file

### Must Have
- Exact replication of ls.py's `match_time()`, `get_nw_delay()`, `find_inject_time()` algorithms
- Proper handling of tsnkit Stream and Path objects
- GCL management with correct queue assignment (always 0 for TSN)
- State serialization for episode-to-episode continuity
- Two-step transaction interface (try_schedule → commit_schedule)

### Must NOT Have (Guardrails)
- **NO modifications to original ls.py** - only copy/adapt methods
- **NO breaking changes to PPO Agent** - only Environment changes
- **NO mock allocation logic** - all allocation must use ls algorithms
- **NO hardcoded path selection** - PPO must decide paths
- **NO changes to ls.py's task_order logic** - PPO handles ordering
- **NO try_schedule modifying GCL** - must be read-only check
- **NO partial GCL corruption on commit failure** - must be atomic or rollback

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (need to check if pytest is available)
- **User wants tests**: YES (Manual verification during implementation, then unit tests)
- **Framework**: pytest (Python standard)

### Manual Verification (during implementation)

Each TODO includes manual verification commands:

**For SlotManager Methods**:
```bash
# Test that find_inject_time matches ls.py behavior
cd D:\python_work\tsnkit
python -c "
from tsnkit.algorithms.sca_drl.phase2_scheduling.environment import SlotManager
from tsnkitkit.algorithms.ls import ls
# Compare outputs for identical scenarios
"
```

**For Environment Integration**:
```bash
# Test that environment initializes correctly
cd D:\python_work\tsnkit
python -m tsnkit.algorithms.sca_drl.runners.run_phase2_training
# Should complete one episode without errors
```

### Unit Tests (after implementation)

**Test file**: `phase2_scheduling/test_slotmanager.py`

```python
def test_match_time_consistency():
    # Compare SlotManager.match_time() with ls.py.match_time()

def test_find_inject_time_consistency():
    # Schedule same stream with same path in both systems
    # Assert inject_time is identical

def test_gcl_update_consistency():
    # Schedule stream in both ls.py and SlotManager
    # Compare GCL entries (start, end, queue values)

def test_state_save_load():
    # Save state, modify, restore
    # Assert GCL matches saved state
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Refactor SlotManager class

Wave 2 (After Wave 1):
├── Task 2: Enhance TSNSchedulingEnv initialization
└── Task 3: Modify TSNSchedulingEnv.step() function

Wave 3 (After Wave 2):
└── Task 4: Create unit tests

Critical Path: Task 1 → Task 2 → Task 3 → Task 4
Parallel Speedup: None (sequential execution required)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None |
| 2 | 1 | 3 | None |
| 3 | 1, 2 | 4 | None |
| 4 | 1, 2, 3 | None | None (final) |

---

## TODOs

- [ ] 1. Refactor SlotManager Class

  **What to do**:
  - [1.1] Remove old numpy-based resource allocation logic (lines 8-88)
  - [1.2] Add GCL data structure: `self._result: Dict[Link, List[(start, end, queue)]]`
  - [1.3] Add `self.lcm: int` attribute (will be set from StreamSet)
  - [1.4] Copy `match_time(t, sche)` static method from ls.py (lines 99-132)
    - This is a binary search for conflict detection
    - Returns: index of entry starting before t, or -1/-2 for boundary conditions
  - [1.5] Copy `get_nw_delay(stream, path)` static method from ls.py (lines 134-136)
    - Formula: `sum(l.t_proc + stream.get_t_trans(l) for l in path.links)`
  - [1.6] Implement `find_inject_time(stream, path)` method from ls.py (lines 138-185)
    - Use `get_nw_delay()` to calculate delay
    - Iterate over possible inject times: `for it in range(0, period - delay + 1)`
    - For each hop in path, check GCL conflicts using `match_time()`
    - For each frame index in LCM: `for k in stream.get_frame_indexes(self.lcm)`
    - Return inject_time if all hops are available, else -1
  - [1.7] Implement `try_schedule(stream, path) -> int` method
    - Call `find_inject_time(stream, path)`
    - Return inject_time (>=0 for success, -1 for unschedulable)
    - **MUST NOT modify GCL** (read-only check)
  - [1.8] Implement `commit_schedule(stream, path, inject_time)` method
    - Copy GCL update logic from ls.py schedule() method (lines 208-224)
    - For each link in path.links:
      - For each frame index: `for k in stream.get_frame_indexes(self.lcm)`
      - Calculate start/end times: `_start + k * period`, `_end + k * period`
      - Append `(start, end, queue=0)` to `self._result[l]`
      - Sort GCL by start time: `self._result[l].sort(key=lambda x: x[0])`
  - [1.9] Implement `save_state() -> Dict` method
    - Serialize GCL to serializable format: `{(link_src, link_dst): [(start, end, queue), ...]}`
    - Use `link._name` tuple as key (Link stores `_name = (src, dst)`)
    - Include `self.lcm` in state
  - [1.10] Implement `load_state(state, network)` method
    - Deserialize: Convert (src, dst) tuples back to Link objects using `network.get_link((src, dst))`
    - Accept `network` parameter to resolve Link objects
    - Restore `self.lcm` from state
  - [1.11] Update `clear()` method to initialize new GCL structure
    - `self._result = {l: [] for l in self.network.links}`
  - [1.12] Update `get_utilization()` to work with new GCL format
    - Calculate average utilization across all links: `sum(end-start for entry in gcl) / (lcm * num_links)`

  **Must NOT do**:
  - Modify original ls.py file
  - Implement any custom allocation algorithm (use ls methods only)
  - Add path selection logic (PPO handles this)
  - Let try_schedule modify GCL (must be read-only)
  - Use the old numpy-based resource tracking

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex refactoring task requiring careful adaptation of existing algorithms
  - **Skills**: None needed for this phase

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:99-132` - `match_time()` binary search implementation
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:134-136` - `get_nw_delay()` network delay calculation
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:138-185` - `find_inject_time()` core allocation algorithm
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:208-224` - GCL update pattern in schedule() method
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\environment.py:8-88` - Current SlotManager structure to modify

  **API/Type References** (contracts to implement against):
  - `D:\python_work\tsnkit\tsnkit\core\_stream.py:178-186` - Stream.get_t_trans() method
  - `D:\python_work\tsnkit\tsnkit\core\_stream.py:250-251` - Stream.get_frame_indexes() method
  - `D:\python_work\tsnkit\tsnkit\core\_network.py:139-166` - Link class attributes (t_proc, rate)
  - `D:\python_work\tsnkit\tsnkit\core\_stream.py:277-299` - StreamSet class (lcm property)

  **WHY Each Reference Matters**:
  - `match_time()`: Binary search is critical for efficient GCL lookup - must be exact copy
  - `find_inject_time()`: Core algorithm determining if/when a stream can be scheduled
  - `GCL update pattern`: Shows how GCL is updated with proper LCM period handling
  - `Stream.get_t_trans()`: Required by `get_nw_delay()` to calculate transmission time per link
  - `Stream.get_frame_indexes()`: Required by `find_inject_time()` to iterate over LCM periods

  **Acceptance Criteria**:

  **Manual Verification**:
  ```bash
  # 1. Verify SlotManager can be imported and initialized
  cd D:\python_work\tsnkit
  python -c "
  from tsnkit.algorithms.sca_drl.phase2_scheduling.environment import SlotManager
  import tsnkit.core as tsnkit
  import networkx as nx

  # Create a simple test network
  G = nx.complete_graph(5).to_directed()
  network = tsnkit.Network()

  # Note: We'll need proper network construction in Task 2
  # For now, just test that SlotManager exists and has required methods
  print('SlotManager imported successfully')
  "

  # 2. Verify match_time works (basic test)
  python -c "
  from tsnkit.algorithms.sca_drl.phase2_scheduling.environment import SlotManager

  # Test case: find entry before t
  sche = [(10, 20, 0), (30, 40, 0)]
  result = SlotManager.match_time(5, sche)
  assert result == -1, f'match_time(5) should be -1, got {result}'

  # Test case: find entry after t
  result = SlotManager.match_time(15, sche)
  assert result == 0, f'match_time(15) should be 0, got {result}'

  print('match_time() basic tests passed')
  "
  ```

  **Evidence to Capture**:
  - [ ] Output from import test (should show "SlotManager imported successfully")
  - [ ] Output from match_time test (should show "match_time() basic tests passed")

  **Commit**: NO (wait for all Environment changes together)

---

- [ ] 2. Enhance TSNSchedulingEnv Initialization

  **What to do**:
  - [2.1] Add `import tsnkit.core as tsnkit` to environment.py
  - [2.2] In `__init__()`, convert NetworkX graph to tsnkit.Network
    - Create Node objects: For each node in G.nodes():
      - Determine type: NodeType.sw if degree > 2 else NodeType.es
      - `node = tsnkit.Node(id, type=node_type)`
      - Add to network
    - For each directed edge (u, v) in G.edges():
      - Get Node objects for u and v
      - Create Link: `tsnkit.Link(link_id, src_node, dst_node, t_proc=1, t_prop=1, q_num=1, rate=1.0)`
      - Extract rate from G.edges[u,v].get('bandwidth', 1.0) and convert to Gbps
      - Add to network
    - Store as `self.network` for later reference
  - [2.3] In `__init__()`, create tsnkit.Stream objects from flows dict
    - For each flow in flows:
      - Get src/dst Node objects: `self.network.get_node(src_id)`, `self.network.get_node(dst_id)`
      - Use tsnkit.Stream constructor: `Stream(id, src_node, [dst_node], size, period, deadline, jitter=0)`
      - Store as `self.tsnkit_streams[flow_id] = stream`
  - [2.4] In `__init__()`, create tsnkit.Path objects for each k_path
    - For each flow's k_paths[path_idx] (list of node IDs):
      - Resolve Link objects: iterate through adjacent node pairs, get `self.network.get_link((u, v))`
      - Build list of Link objects
      - Create Path: `tsnkit.Path(link_list, id=path_id)`
      - Store as `self.tsnkit_paths[flow_id][path_idx] = tsnkit_path`
  - [2.5] Create tsnkit.StreamSet and add all Stream objects
    - Create `StreamSet()` instance
    - For each stream in `self.tsnkit_streams.values()`, add to StreamSet
    - Compute LCM: `StreamSet._lcm = np.lcm.reduce([stream.period for stream in streams])`
    - Store as `self.task` (naming from ls.py) for LCM access
  - [2.6] Update SlotManager initialization to receive tsnkit.Network and StreamSet
    - Pass: `SlotManager(self.network, self.task.lcm, config['env_params'])`
  - [2.7] Update SlotManager init to accept network and lcm
    - Store `self.network` and `self.lcm` attributes
    - Initialize GCL: `self._result = {l: [] for l in network.links}`

  **Must NOT do**:
  - Modify flow generator or topology generator
  - Change PPO Agent or training scripts
  - Break existing flow dict structure (add tsnkit objects alongside)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex type conversion requiring understanding of tsnkit object model
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (with Task 3 if independent)
  - **Blocks**: None
  - **Blocked By**: Task 1 (SlotManager must be ready first)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:45-48` - How ls.py initializes task and net
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\runners\run_phase2_training.py:66-74` - How flows and graph are generated
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\common\flow_gen.py:12-51` - Flow dict structure
  - `D:\python_work\tsnkit\tsnkit\core\_network.py:42-96` - Node class constructor with NodeType
  - `D:\python_work\tsnkit\tsnkit\core\_stream.py:130-148` - Stream class constructor

  **API/Type References** (contracts to implement against):
  - `D:\python_work\tsnkit\tsnkit\core\__init__.py:8-10` - Exported tsnkit classes
  - `D:\python_work\tsnkit\tsnkit\core\_network.py:139-166` - Link constructor parameters
  - `D:\python_work\tsnkit\tsnkit\core\_stream.py:130-148` - Stream constructor parameters

  **WHY Each Reference Matters**:
  - ls.py init pattern: Shows how to properly load tsnkit objects
  - Flow generator: Shows the exact dict structure we need to convert from
  - tsnkit constructors: Required to create Stream and Path objects with correct parameters

  **Acceptance Criteria**:

  **Manual Verification**:
  ```bash
  # 1. Verify environment can be initialized with tsnkit objects
  cd D:\python_work\tsnkit
  python -c "
  from tsnkit.algorithms.sca_drl.phase2_scheduling.environment import TSNSchedulingEnv
  import sys
  sys.path.append('tsnkit/algorithms/sca_drl')
  from common.topology_gen import TopologyGenerator
  from common.flow_gen import FlowGenerator

  # Minimal config
  config = {
      'traffic': {'num_flows': 2, 'period_list': [1000], 'size_range': [64, 1518]},
      'ksp': {'k': 3},
      'env_params': {'time_limit': 100, 'bandwidth': 1000, 'max_steps': 100},
      'normalization': {'max_size': 1500, 'max_period': 10000, 'max_deadline': 10000}
  }

  topo_gen = TopologyGenerator(num_nodes=5)
  graph = topo_gen.generate_graph()
  flow_gen = FlowGenerator(config, graph)
  flows = flow_gen.generate_flows()

  env = TSNSchedulingEnv(flows, graph, config['env_params'])
  print('Environment initialized with tsnkit objects')
  assert hasattr(env, 'task'), 'tsnkit StreamSet (task) not created'
  assert hasattr(env, 'network'), 'tsnkit Network not created'
  assert hasattr(env, 'tsnkit_streams'), 'tsnkit_streams dict not created'
  assert hasattr(env, 'tsnkit_paths'), 'tsnkit_paths dict not created'
  print(f'Created {len(env.tsnkit_streams)} tsnkit.Stream objects')
  print('All tsnkit objects present')
  "
  ```

  **Evidence to Capture**:
  - [ ] Output showing "Environment initialized with tsnkit objects"
  - [ ] Output showing "All tsnkit objects present"

  **Commit**: NO (wait for all Environment changes together)

---

- [ ] 3. Modify TSNSchedulingEnv.step() Function

  **What to do**:
  - [3.1] In `step()`, remove mock allocation logic (lines 174-188)
  - [3.2] Get tsnkit Stream: `stream = self.tsnkit_streams[flow_id]`
  - [3.3] Get tsnkit Path: `path = self.tsnkit_paths[flow_id][path_idx]`
  - [3.4] Call new SlotManager interface: `inject_time = self.slot_manager.try_schedule(stream, path)`
  - [3.5] Handle scheduling result
    - If `inject_time == -1`: reward = -0.1, record failure
    - If `inject_time >= 0`: call `self.slot_manager.commit_schedule(stream, path, inject_time)`, reward = 1
  - [3.6] Update flow_states with scheduling result
    - Store inject_time in flow_states: `self.flow_states[flow_id]['inject_time'] = inject_time`
    - Mark as scheduled: `self.flow_states[flow_id]['scheduled'] = True`
  - [3.7] Ensure done condition checks all flows scheduled

  **Must NOT do**:
  - Change action space (still MultiDiscrete[flow_id, path_idx])
  - Change reward structure (keep existing 1/-0.1/-1 values)
  - Modify observation space
  - Break PPO Agent compatibility

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Focused modification of single function with clear changes
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (with Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Tasks 1, 2

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\environment.py:163-197` - Current step() implementation
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:71-86` - ls.py solve() loop (shows how schedule is called)
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\environment.py:145-161` - reset() method for state management

  **API/Type References** (contracts to implement against):
  - SlotManager.try_schedule() - New interface created in Task 1
  - SlotManager.commit_schedule() - New interface created in Task 1

  **WHY Each Reference Matters**:
  - Current step(): Shows reward structure and done condition to preserve
  - ls.py solve(): Shows how to iterate over streams and handle scheduling failures
  - reset(): Important for understanding how state is managed between episodes

  **Acceptance Criteria**:

  **Manual Verification**:
  ```bash
  # 1. Test one full episode with new step() logic
  cd D:\python_work\tsnkit
  python -c "
  from tsnkit.algorithms.sca_drl.phase2_scheduling.environment import TSNSchedulingEnv
  from tsnkit.algorithms.sca_drl.phase2_scheduling.scheduler_agent import PPOAgent
  import sys
  sys.path.append('tsnkit/algorithms/sca_drl')
  from common.topology_gen import TopologyGenerator
  from common.flow_gen import FlowGenerator
  import numpy as np

  # Minimal config
  config = {
      'traffic': {'num_flows': 3, 'period_list': [1000], 'size_range': [64, 1518]},
      'ksp': {'k': 3},
      'env_params': {'time_limit': 100, 'bandwidth': 1000, 'max_steps': 10},
      'normalization': {'max_size': 1500, 'max_period': 10000, 'max_deadline': 10000}
  }

  topo_gen = TopologyGenerator(num_nodes=5)
  graph = topo_gen.generate_graph()
  flow_gen = FlowGenerator(config, graph)
  flows = flow_gen.generate_flows()

  env = TSNSchedulingEnv(flows, graph, config['env_params'])
  agent = PPOAgent(env.observation_space.shape[0], len(flows), 3, lr=0.001)

  # Run one episode
  state = env.reset()
  done = False
  steps = 0
  while not done and steps < 10:
      action = agent.select_action(state, env.get_mask())
      state, reward, done, info = env.step(action)
      steps += 1

  print(f'Episode completed in {steps} steps')
  scheduled = sum(1 for f in env.flow_states.values() if f['scheduled'])
  print(f'Final state: scheduled={scheduled} / {len(flows)}')
  assert steps > 0, 'Episode did not progress'
  print('Episode test passed')
  "
  ```

  **Evidence to Capture**:
  - [ ] Output showing episode completion and flow scheduling status
  - [ ] Output showing "Episode test passed"

  **Commit**: YES
  - Message: `feat(sca-drl): integrate ls-based SlotManager into PPO environment`
  - Files: `tsnkit/algorithms/sca_drl/phase2_scheduling/environment.py`
  - Pre-commit: Run episode verification test above

---

- [ ] 4. Create Unit Tests

  **What to do**:
  - [4.1] Create `phase2_scheduling/test_slotmanager.py` file
  - [4.2] Import pytest and necessary modules (SlotManager, ls, tsnkit)
  - [4.3] Create helper function to generate identical test scenarios
    - Small topology, few streams, deterministic paths
  - [4.4] Test `test_match_time_consistency()`
    - Compare SlotManager.match_time() with ls.py.match_time()
    - Test edge cases: before first, after last, between entries, empty list
  - [4.5] Test `test_find_inject_time_consistency()`
    - Create identical Stream, Path, and GCL in both systems
    - Schedule same stream with ls.py first, record inject_time
    - Schedule same stream with SlotManager, compare inject_time
  - [4.6] Test `test_gcl_update_consistency()`
    - Schedule stream in both ls.py and SlotManager
    - Compare GCL entries (start, end, queue values)
    - Verify GCL is sorted by start time
  - [4.7] Test `test_state_save_load()`
    - Schedule some streams
    - Save state
    - Modify GCL
    - Load state
    - Assert GCL matches saved state
  - [4.8] Test `test_try_schedule_read_only()`
    - Get GCL snapshot before try_schedule
    - Call try_schedule (success or failure)
    - Assert GCL unchanged after try_schedule
  - [4.9] Add pytest configuration if needed (pytest.ini or conftest.py)

  **Must NOT do**:
  - Modify implementation code in tests
  - Skip critical consistency tests
  - Use random values (deterministic scenarios only for comparison)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Test code is documentation-focused and requires clear assertions
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2, 3

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `D:\python_work\tsnkit\tsnkit\algorithms\ls.py:99-185` - ls.py methods to test against
  - `D:\python_work\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\environment.py:8-88` - SlotManager methods to test
  - Pytest documentation: https://docs.pytest.org/en/stable/ - Test structure and assertions

  **API/Type References** (contracts to implement against):
  - SlotManager methods created in Task 1
  - ls.py methods used for ground truth comparison

  **WHY Each Reference Matters**:
  - ls.py methods: Ground truth for correct behavior
  - SlotManager methods: Implementation under test
  - Pytest docs: Standard Python testing patterns

  **Acceptance Criteria**:

  **Manual Verification**:
  ```bash
  # 1. Install pytest if not available
  pip install pytest

  # 2. Run all tests
  cd D:\python_work\tsnkit
  pytest tsnkit/algorithms/sca_drl/phase2_scheduling/test_slotmanager.py -v

  # Expected: All tests pass with 0 failures
  ```

  **Evidence to Capture**:
  - [ ] Pytest output showing all tests passed
  - [ ] Test report with 0 failures

  **Commit**: YES
  - Message: `test(sca-drl): add unit tests for ls-based SlotManager`
  - Files: `tsnkit/algorithms/sca_drl/phase2_scheduling/test_slotmanager.py`
  - Pre-commit: Run pytest and ensure all tests pass

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1-3 | `feat(sca-drl): integrate ls-based SlotManager into PPO environment` | environment.py | Run episode verification |
| 4 | `test(sca-drl): add unit tests for ls-based SlotManager` | test_slotmanager.py | pytest all pass |

---

## Success Criteria

### Verification Commands
```bash
# 1. Run full training script to verify integration
cd D:\python_work\tsnkit
python -m tsnkit.algorithms.sca_drl.runners.run_phase2_training

# Expected: Training completes at least one episode without errors

# 2. Run unit tests
pytest tsnkit/algorithms/sca_drl/phase2_scheduling/test_slotmanager.py -v

# Expected: All tests pass, 0 failures
```

### Final Checklist
- [ ] SlotManager uses ls.py's `find_inject_time()` algorithm exactly
- [ ] SlotManager correctly maintains GCL like ls.py
- [ ] TSNSchedulingEnv initializes with tsnkit objects
- [ ] step() uses new SlotManager interface (try_schedule + commit_schedule)
- [ ] State can be saved and restored across episodes
- [ ] Unit tests verify consistency with ls.py
- [ ] PPO training runs without errors
- [ ] No modifications to original ls.py file
- [ ] try_schedule does NOT modify GCL (read-only check)
- [ ] commit_schedule applies GCL changes correctly
