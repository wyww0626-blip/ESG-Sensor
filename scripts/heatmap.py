import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义模型与CSV文件路径
csv_files = [
    ("Baichuan4", "baichuan_all_folders_evaluation_results.csv"),
    ("Deepseek-reasoner", "deepseek_all_folders_evaluation_results.csv"),
    ("Glm-4-Plus", "zhipu_all_folders_evaluation_results.csv")
]

# 调整总图尺寸（保证每个子图宽度一致）
fig, axes = plt.subplots(
    nrows=1, ncols=3,
    figsize=(10, 8),  # 每个模型分配 4 英寸宽度
    gridspec_kw={'width_ratios': [1, 1, 1.2]}
)

# 色彩设置
vmin = 0.5
vmax = 1.0
cmap = "YlGnBu"

# 自定义字体加粗样式
bold_font = FontProperties(weight='bold')
normal_font = FontProperties(weight='normal')

# 遍历三个模型绘制热力图
for ax, (title, file) in zip(axes, csv_files):
    df = pd.read_csv(file)
    sampled_df = df.sample(n=25, random_state=42).sort_index()
    sampled_df["文件夹对"] = sampled_df["参考文件夹"] + " | " + sampled_df["生成文件夹"]
    sampled_df = sampled_df.set_index("文件夹对")

    metrics = sampled_df[["average_F1"]]

    # 绘图并获取热力图对象
    heat = sns.heatmap(
        metrics,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=(ax == axes[-1]),
        yticklabels=False
    )

    # 设置所有单元格数值：大于0.75的加粗
    for text in heat.texts:
        value = float(text.get_text())
        if value > 0.75:
            text.set_fontproperties(bold_font)
            text.set_color("black")
        else:
            text.set_fontproperties(normal_font)
            text.set_color("black")

    # 设置子图标题
    ax.set_title(f"{title} (F1 Score)", fontsize=12)
    # ax.set_xlabel("average_F1", fontsize=11)
    if ax == axes[0]:
        ax.set_ylabel("ESG Report", fontsize=11)
    else:
        ax.set_ylabel("")

    ax.set_aspect("auto")  # 保持每张图横向比例一致

# 总标题
fig.suptitle("Comparative Heatmaps of ESG average_F1 Scores Across Models", fontsize=16, y=1.03)

# 自动布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("ESG_average_F1_heatmaps_all_models_bolded.png", dpi=300)
plt.show()

