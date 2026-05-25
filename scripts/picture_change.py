import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_summary_diagram():
    """
    绘制“摘要”与“全文摘要”“局部摘要”“线索摘要”之间关系的示意图。
    使用matplotlib的patches.Rectangle绘制方框，并添加箭头连接。
    """

    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    # 方框统一设置
    box_width = 2.0
    box_height = 0.8
    facecolor = "#cce5ff"  # 浅蓝色填充
    edgecolor = "black"

    # ---- 1. 绘制“摘要”方框 ----
    # 位置：左侧，坐标(0, 0)
    summary_box = patches.Rectangle(
        (0, 0), box_width, box_height,
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2
    )
    ax.add_patch(summary_box)
    ax.text(
        0 + box_width / 2, 0 + box_height / 2,
        "摘要", ha="center", va="center", fontsize=10, zorder=3
    )

    # ---- 2. 绘制右侧三个方框 ----
    # 2.1 全文摘要
    full_box_y = 0  # y坐标
    full_box = patches.Rectangle(
        (4, full_box_y), box_width, box_height,
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2
    )
    ax.add_patch(full_box)
    ax.text(
        4 + box_width / 2, full_box_y + box_height / 2,
        "全文摘要", ha="center", va="center", fontsize=10, zorder=3
    )

    # 2.2 局部摘要
    partial_box_y = -1.5
    partial_box = patches.Rectangle(
        (4, partial_box_y), box_width, box_height,
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2
    )
    ax.add_patch(partial_box)
    ax.text(
        4 + box_width / 2, partial_box_y + box_height / 2,
        "局部摘要", ha="center", va="center", fontsize=10, zorder=3
    )

    # 2.3 线索摘要
    clue_box_y = -3.0
    clue_box = patches.Rectangle(
        (4, clue_box_y), box_width, box_height,
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2
    )
    ax.add_patch(clue_box)
    ax.text(
        4 + box_width / 2, clue_box_y + box_height / 2,
        "线索摘要", ha="center", va="center", fontsize=10, zorder=3
    )

    # ---- 3. 绘制箭头 ----
    # “摘要”方框中心右侧坐标：(2, 0.4)
    # 右侧方框中心左侧坐标分别为：
    #  - 全文摘要: (4, 0.4)
    #  - 局部摘要: (4, -1.1)
    #  - 线索摘要: (4, -2.6)
    arrow_start = (2, 0.4)

    ax.annotate(
        "", xy=(4, 0.4), xytext=arrow_start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )
    ax.annotate(
        "", xy=(4, -1.1), xytext=arrow_start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )
    ax.annotate(
        "", xy=(4, -2.6), xytext=arrow_start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )

    # ---- 4. 坐标范围与外观设置 ----
    ax.set_xlim(-0.5, 7.0)
    ax.set_ylim(-4.0, 2.0)
    ax.set_aspect('equal')   # 保持x/y比例一致
    ax.axis('off')           # 关闭坐标轴

    plt.show()

if __name__ == "__main__":
    draw_summary_diagram()

