import os
from pathlib import Path

# ==========================================
# ⚙️ 配置区域 (Configuration) - 请在此处填入你的信息
# ==========================================

# 1. 目标路径列表：可以是具体的 .py 文件，也可以是整个文件夹
# 如果是文件夹，脚本会自动抓取里面所有的 .py 文件。
TARGET_PATHS = [
    # 示例：
    # "D:/my_research_project/phase1_code",
    # "D:/my_research_project/phase2_code/main_ppo.py",
    "./"  # 默认抓取当前运行目录下的所有文件，请根据需要替换
]

# 2. 输出的 Markdown 文件路径和名称
OUTPUT_FILE = "Project_Code_Context.md"

# 3. 忽略的目录或文件（避免把无关的包或缓存喂给 AI，浪费 Token）
IGNORE_DIR_NAMES = {".git", "__pycache__","_archive_trash", "venv", "env",
                    ".vscode", ".obsidian", ".idea","common", 
                    "results", "checkpoints"}

IGNORE_FILE_NAMES = {os.path.basename(__file__)}
# ==========================================
# 🛠️ 核心执行逻辑 (Core Logic)
# ==========================================

def generate_tree_str(paths):
    """生成简单的文件树状图，帮助 AI 理解项目结构"""
    tree_lines = []
    tree_lines.append("## 📂 项目文件结构 (Project Structure)\n```text")
    
    for target in paths:
        target_path = Path(target)
        if not target_path.exists():
            continue
            
        if target_path.is_file() and target_path.suffix == '.py':
            tree_lines.append(f"📄 {target_path.name}")
        elif target_path.is_dir():
            tree_lines.append(f"📁 {target_path.name}/")
            for root, dirs, files in os.walk(target_path):
                # 过滤忽略的文件夹
                dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES]
                level = root.replace(str(target_path), '').count(os.sep)
                indent = ' ' * 4 * (level + 1)
                
                # 打印相对路径下的目录
                if root != str(target_path):
                    tree_lines.append(f"{indent}📁 {Path(root).name}/")
                
                # 打印 .py 文件
                sub_indent = ' ' * 4 * (level + 2) if root != str(target_path) else ' ' * 4 * (level + 1)
                for f in files:
                    if f.endswith('.py') and f not in IGNORE_FILE_NAMES:
                        tree_lines.append(f"{sub_indent}📄 {f}")
    
    tree_lines.append("```\n")
    return "\n".join(tree_lines)

def extract_and_format_code(paths):
    """提取代码并格式化为 Markdown"""
    md_content = []
    md_content.append("# 💻 源代码上下文 (Source Code Context)\n")
    
    for target in paths:
        target_path = Path(target)
        if not target_path.exists():
            print(f"⚠️ 警告: 路径不存在 -> {target_path}")
            continue

        files_to_process = []
        if target_path.is_file() and target_path.suffix == '.py':
            files_to_process.append(target_path)
        elif target_path.is_dir():
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES]
                for file in files:
                    if file.endswith('.py') and file not in IGNORE_FILE_NAMES:
                        files_to_process.append(Path(root) / file)

        for file_path in files_to_process:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # 使用相对路径作为标题，比纯文件名更好，能体现层级关系
                md_content.append(f"## File: `{file_path.name}`")
                md_content.append(f"> 路径: `{file_path.absolute()}`\n")
                md_content.append("```python")
                md_content.append(code)
                md_content.append("```\n")
                print(f"✅ 已成功提取: {file_path.name}")
            except Exception as e:
                print(f"❌ 无法读取文件 {file_path.name}: {e}")
                
    return "\n".join(md_content)

def main():
    print("🚀 开始扫描和合并代码...")
    
    # 1. 生成树状图
    tree_content = generate_tree_str(TARGET_PATHS)
    
    # 2. 提取代码
    code_content = extract_and_format_code(TARGET_PATHS)
    
    # 3. 组合并写入文件
    final_output = tree_content + "\n" + code_content
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"\n🎉 大功告成！所有代码已完美合并至: {Path(OUTPUT_FILE).absolute()}")
        print("💡 现在你可以直接将该 Markdown 文件的内容复制或者上传给 AI 了。")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

if __name__ == "__main__":
    main()