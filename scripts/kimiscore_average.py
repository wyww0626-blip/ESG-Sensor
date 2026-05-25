import pandas as pd

# 读取 CSV 文件（请将文件名替换为实际文件名）
input_csv = "deepseek_evaluation/baichuan_evaluation_results_dp.csv"  # 请替换成你的 CSV 文件路径
df = pd.read_csv(input_csv)

# 对列名去除前后空格（如果有）
df.columns = df.columns.str.strip()

# 检查必要的列是否存在
required_cols = ["参考摘要文件名", "生成摘要文件名", "语义一致性", "信息覆盖度",
                 "表达相似性", "简洁性与冗余度", "逻辑结构", "总体相似度评分", "来源CSV文件"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV 文件中缺少必要的列，请检查文件内容！")

# 按“来源文件”分组处理
grouped = df.groupby("来源CSV文件")
group_results = []  # 存放各组计算后的平均结果
filtered_rows_list = []  # 存放各组过滤后的记录

# 遍历每个分组
for source, group in grouped:
    # 按总体相似度评分升序排序
    group_sorted = group.sort_values("总体相似度评分", ascending=True)
    # 如果组内记录数大于1，则去掉最低的一行，否则保留全部记录
    if len(group_sorted) > 1:
        group_filtered = group_sorted.iloc[1:]
    else:
        group_filtered = group_sorted.copy()
    filtered_rows_list.append(group_filtered)

    # 计算当前组各项平均分（转换为 float 后计算平均）
    avg_semantic = group_filtered["语义一致性"].astype(float).mean()
    avg_coverage = group_filtered["信息覆盖度"].astype(float).mean()
    avg_expression = group_filtered["表达相似性"].astype(float).mean()
    avg_conciseness = group_filtered["简洁性与冗余度"].astype(float).mean()
    avg_logic = group_filtered["逻辑结构"].astype(float).mean()
    avg_overall = group_filtered["总体相似度评分"].astype(float).mean()

    group_results.append({
        "来源CSV文件": source,
        "记录数": len(group_filtered),
        "平均语义一致性": avg_semantic,
        "平均信息覆盖度": avg_coverage,
        "平均表达相似性": avg_expression,
        "平均简洁性与冗余度": avg_conciseness,
        "平均逻辑结构": avg_logic,
        "平均总体相似度评分": avg_overall
    })

# 将各组结果转换为 DataFrame
group_result_df = pd.DataFrame(group_results)

# 合并所有组过滤后的数据，用于计算总体平均
if filtered_rows_list:
    all_filtered = pd.concat(filtered_rows_list, ignore_index=True)
    overall_avg_semantic = all_filtered["语义一致性"].astype(float).mean()
    overall_avg_coverage = all_filtered["信息覆盖度"].astype(float).mean()
    overall_avg_expression = all_filtered["表达相似性"].astype(float).mean()
    overall_avg_conciseness = all_filtered["简洁性与冗余度"].astype(float).mean()
    overall_avg_logic = all_filtered["逻辑结构"].astype(float).mean()
    overall_avg_overall = all_filtered["总体相似度评分"].astype(float).mean()
else:
    overall_avg_semantic = overall_avg_coverage = overall_avg_expression = overall_avg_conciseness = overall_avg_logic = overall_avg_overall = None

# 构造总体平均结果的 DataFrame，并设置来源文件为“总体平均”
overall_result_df = pd.DataFrame({
    "来源CSV文件": ["总体平均"],
    "记录数": [""],
    "平均语义一致性": [overall_avg_semantic],
    "平均信息覆盖度": [overall_avg_coverage],
    "平均表达相似性": [overall_avg_expression],
    "平均简洁性与冗余度": [overall_avg_conciseness],
    "平均逻辑结构": [overall_avg_logic],
    "平均总体相似度评分": [overall_avg_overall]
})

# 将各组结果和总体平均结果合并到一起
final_df = pd.concat([group_result_df, overall_result_df], ignore_index=True)

# 输出最终结果，并保存到 CSV 文件
print(final_df)
final_df.to_csv("deepseek_evaluation/baichuan_evaluation_results_average.csv", index=False, encoding="utf-8-sig")
print("合并后的结果已保存到 combined_evaluation_results.csv")

