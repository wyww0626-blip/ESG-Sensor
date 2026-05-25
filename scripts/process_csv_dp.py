import pandas as pd
import re
import os


# 定义函数，利用正则表达式从评估结果中提取各项分数
def extract_scores(text):
    # 分别匹配各维度的分数
    semantic = re.search(r'-\s*语义一致性：([\d\.]+)\s*分', text)
    coverage = re.search(r'-\s*信息覆盖度：([\d\.]+)\s*分', text)
    expression = re.search(r'-\s*表达相似性：([\d\.]+)\s*分', text)
    conciseness = re.search(r'-\s*简洁性与冗余度：([\d\.]+)\s*分', text)
    logic = re.search(r'-\s*逻辑结构：([\d\.]+)\s*分', text)
    overall = re.search(r'-\s*总体相似度评分：([\d\.]+)\s*分', text)

    # 若匹配到则转换为 float，否则设为 None
    semantic_score = float(semantic.group(1)) if semantic else None
    coverage_score = float(coverage.group(1)) if coverage else None
    expression_score = float(expression.group(1)) if expression else None
    conciseness_score = float(conciseness.group(1)) if conciseness else None
    logic_score = float(logic.group(1)) if logic else None
    overall_score = float(overall.group(1)) if overall else None

    return semantic_score, coverage_score, expression_score, conciseness_score, logic_score, overall_score


# 指定存放 CSV 文件的文件夹（此处文件夹中包含多个 CSV 文件）
input_folder = "zhipu_evaluation_results_dp"

# 获取该文件夹中所有 CSV 文件（忽略大小写）
csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]

all_results = []  # 用于存放每个 CSV 文件处理后的 DataFrame

# 遍历每个 CSV 文件进行处理
for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(file_path)

    # 对“评估结果”列应用 extract_scores 函数，提取出各维度分数
    scores = df["评估结果"].apply(extract_scores)
    df[['语义一致性', '信息覆盖度', '表达相似性', '简洁性与冗余度', '逻辑结构', '总体相似度评分']] = pd.DataFrame(
        scores.tolist(), index=df.index)

    # 添加一列“来源CSV文件”以标记该记录来源于哪个 CSV 文件
    df["来源CSV文件"] = csv_file

    # 保留需要的列
    df_selected = df[["参考摘要文件名", "生成摘要文件名", "语义一致性", "信息覆盖度", "表达相似性",
                      "简洁性与冗余度", "逻辑结构", "总体相似度评分", "来源CSV文件"]]
    all_results.append(df_selected)

# 合并所有 CSV 文件的处理结果
combined_df = pd.concat(all_results, ignore_index=True)

# 输出并保存合并后的结果到新的 CSV 文件中
output_file = "deepseek_evaluation/zhipu_evaluation_results_dp.csv"
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"合并后的结果已保存到 {output_file}")

