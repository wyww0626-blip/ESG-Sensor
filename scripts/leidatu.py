import numpy as np
import matplotlib.pyplot as plt

# 设置字体（如Arial不可用，可更换为其他字体）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 指标标签（5个指标）
labels = [
    "语义一致性",
    "信息覆盖度",
    "表达相似性",
    "简洁性与冗余度",
    "逻辑结构"
]

# 更新后的各模型得分数据
scores_dict = {
    "Deepseek-reasoner": [7.2311, 6.1480, 6.2570, 5.9093, 6.1106],
    "Baichuan4":         [7.1728, 6.0883, 6.2008, 6.0048, 6.0306],
    "Glm-4-plus":        [7.7419, 6.7085, 6.6708, 7.1400, 6.6980]
}

# 为闭合雷达图，将每组数据的首项重复一遍，同时标签也重复第一个
for model in scores_dict:
    scores_dict[model] += scores_dict[model][:1]
labels += labels[:1]

# 计算每个指标的角度
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=True)

# 创建极坐标子图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 定义颜色和线型（学术风格）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
line_styles = ['-', '--', '-.']

# 绘制各模型的雷达图曲线与填充区域（不显示数据点数值）
for idx, (model, scores) in enumerate(scores_dict.items()):
    ax.plot(angles, scores, line_styles[idx], linewidth=2, markersize=7,
            label=model, color=colors[idx])
    ax.fill(angles, scores, alpha=0.1, color=colors[idx])

# 隐藏默认的极坐标刻度标签
ax.set_xticks([])

# 定义每个标签的独立距离（顺序与labels[:-1]一致，单位为半径）
label_distances = [12, 11, 12, 13, 11]
for angle, label, dist in zip(angles[:-1], labels[:-1], label_distances):
    ax.text(angle, dist, label, fontsize=13, fontweight='bold',
            ha='center', va='center', rotation=0)

# 设置y轴刻度和网格样式
ax.set_yticks([2, 4, 6, 8])
ax.set_yticklabels(["2", "4", "6", "8"], fontsize=12, color="grey")
ax.set_ylim(0, 10)
ax.grid(color='gray', linestyle='--', linewidth=0.5)

# 添加标题
plt.title("deepeekscore 评估", fontsize=16, fontweight='bold', y=1.1)

# 将图例放置在左下角，并向左靠拢
plt.legend(loc='lower left', bbox_to_anchor=(-0.5, -0.1), fontsize=12)

# 保存图片并显示
plt.savefig("zh_deepeekscore_leidatu_updated.png", dpi=300, bbox_inches='tight')
plt.show()

