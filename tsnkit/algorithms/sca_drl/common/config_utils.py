import yaml
import os

def recursive_merge(base_dict, update_dict):
    """
    递归合并两个字典。
    如果 key 对应的值都是字典，则继续递归合并；
    否则，update_dict 的值覆盖 base_dict。
    """
    if not isinstance(base_dict, dict) or not isinstance(update_dict, dict):
        return update_dict

    for k, v in update_dict.items():
        if k in base_dict and isinstance(base_dict[k], dict) and isinstance(v, dict):
            recursive_merge(base_dict[k], v)
        else:
            base_dict[k] = v
    return base_dict

def load_config(*config_paths):
    """
    加载并合并多个配置文件。
    参数:
        *config_paths: 配置文件路径列表。
                       后传入的配置会覆盖先传入的配置。
                       如果没有传入参数，默认加载 'configs/base_config.yaml'。
    """
    # 获取项目根目录 (假设此文件在 src/common/)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 默认回退
    if not config_paths:
        config_paths = ["configs/base_config.yaml"]
    
    final_config = {}
    
    for path in config_paths:
        # 自动处理路径
        if not os.path.isabs(path):
            full_path = os.path.join(root_dir, path)
        else:
            full_path = path
        
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    current_config = yaml.safe_load(f) or {}
                    # 将当前配置合并到最终配置中
                    recursive_merge(final_config, current_config)
                # print(f"[Info] Loaded config: {path}") # 可选：打印加载日志
            except Exception as e:
                print(f"[Error] Failed to load config {path}: {e}")
        else:
            print(f"[Warning] Config file not found: {full_path}")
            
    return final_config