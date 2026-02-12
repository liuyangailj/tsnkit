# === Cell 1: 设置 matplotlib 在 notebook 中显示 ===
%matplotlib inline
# === Cell 2: 运行仿真并绘制图表 ===
import sys
sys.argv = ['tas.py', './1_task.csv', './']  # 模拟命令行参数
from tsnkit.simulation.tas import simulation
# 运行仿真（draw_results=True 会自动调用 draw 函数）
log, output = simulation('./1_task.csv', './', draw_results=True)
print("\n仿真完成！图表应该显示在上方")