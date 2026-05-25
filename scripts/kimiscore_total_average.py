import pandas as pd

# 读取 CSV 文件（请将文件名替换为实际文件路径）
df = pd.read_csv("deepseek_evaluation/zhipu_evaluation_results_average.csv")

# 根据“平均总体相似度评分”升序排序
df_sorted = df.sort_values("平均总体相似度评分", ascending=True)

# 去掉排序后最低的一行（即第一行）
df_filtered = df_sorted.iloc[1:]

# 计算剩余数据的各项平均值
avg_semantic = df_filtered["平均语义一致性"].mean()
avg_coverage = df_filtered["平均信息覆盖度"].mean()
avg_expression = df_filtered["平均表达相似性"].mean()
avg_conciseness = df_filtered["平均简洁性与冗余度"].mean()
avg_logic = df_filtered["平均逻辑结构"].mean()
avg_overall = df_filtered["平均总体相似度评分"].mean()

# 输出计算结果
print("剩余数据的总体平均结果：")
print(f"平均语义一致性：{avg_semantic:.4f}")
print(f"平均信息覆盖度：{avg_coverage:.4f}")
print(f"平均表达相似性：{avg_expression:.4f}")
print(f"平均简洁性与冗余度：{avg_conciseness:.4f}")
print(f"平均逻辑结构：{avg_logic:.4f}")
print(f"平均总体相似度评分：{avg_overall:.4f}")

