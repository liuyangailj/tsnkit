class SCADRL_Wrapper:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device
        # 加载 PPO 模型
        self.agent = load_ppo_model(model_path) 

    def init(self, task_path: str, net_path: str) -> None:
        # 使用我们之前讨论的 adapter 加载数据
        self.nx_graph, self.flows = load_data_from_tsnkit(task_path, net_path)
        # 初始化环境
        self.env = TSNSchedulingEnv(self.flows, self.nx_graph, ...)

    def prepare(self) -> None:
        # 这里的 prepare 可以用来做 reset 或者特征预处理
        self.state = self.env.reset()

    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()
        
        # --- 运行你的 PPO 推理逻辑 ---
        done = False
        while not done:
            action = self.agent.select_action(self.state)
            self.state, reward, done, _ = self.env.step(action)
        
        # 判断是否成功调度所有流
        is_success = self.env.check_success()
        
        end_time = utils.time_log()
        result_status = utils.Result.schedulable if is_success else utils.Result.unschedulable
        
        return utils.Statistics("-", result_status, end_time - start_time)

    def output(self) -> utils.Config:
        # 将 env 中的调度结果转换为 tsnkit 的 Config 格式用于输出
        return convert_env_state_to_config(self.env)