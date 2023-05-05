import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('数据文件.xls')

# 定义一组颜色
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

# 创建子图网格
fig, axs = plt.subplots(4, 2, figsize=(15, 10))

# 对每个区域进行绘图
for i, column in enumerate(df.columns[1:]):
    # 确定子图的位置
    ax = axs[i // 2, i % 2]

    # 在当前子图上绘制数据，并设置颜色
    ax.plot(df['day'], df[column], label=column, color=colors[i])

    # 添加图例
    ax.legend()

    # 开启网格线
    ax.grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()