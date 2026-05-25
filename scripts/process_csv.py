import pandas as pd
import re
import os


# 定义函数，利用正则表达式提取评估结果中的各项分数
def extract_scores(text):
    # 分别匹配各维度
    semantic = re.search(r'-\s*语义一致性：([\d\.]+)\s*分', text)
    coverage = re.search(r'-\s*信息覆盖度：([\d\.]+)\s*分', text)
    expression = re.search(r'-\s*表达相似性：([\d\.]+)\s*分', text)
    conciseness = re.search(r'-\s*简洁性与冗余度：([\d\.]+)\s*分', text)
    logic = re.search(r'-\s*逻辑结构：([\d\.]+)\s*分', text)
    overall = re.search(r'-\s*总体相似度评分：([\d\.]+)\s*分', text)

    # 将匹配到的分数字符串转换为 float（若匹配不到则返回 None）
    semantic_score = float(semantic.group(1)) if semantic else None
    coverage_score = float(coverage.group(1)) if coverage else None
    expression_score = float(expression.group(1)) if expression else None
    conciseness_score = float(conciseness.group(1)) if conciseness else None
    logic_score = float(logic.group(1)) if logic else None
    overall_score = float(overall.group(1)) if overall else None

    return semantic_score, coverage_score, expression_score, conciseness_score, logic_score, overall_score


# 输入文件夹路径（存放多个 CSV 文件）
input_folder = "baichuan_evaluation_results"
# 输出合并结果的 CSV 文件名称
output_file = "baichuan_kimi_evaluation/baichuan_kimi_evaluation_results.csv"

all_results = []  # 用于存放所有 CSV 文件处理后的 DataFrame

# 获取文件夹中所有 CSV 文件
csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]

# 遍历每个 CSV 文件进行处理
for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(file_path)

    # 对"评估结果"列应用 extract_scores 函数
    scores = df["评估结果"].apply(extract_scores)
    df[['语义一致性', '信息覆盖度', '表达相似性', '简洁性与冗余度', '逻辑结构', '总体相似度评分']] = pd.DataFrame(
        scores.tolist(), index=df.index)

    # 添加一列“来源文件”，标记当前数据来自哪个 CSV 文件
    df["来源文件"] = csv_file

    # 选取需要展示的列
    result_df = df[["参考摘要文件名", "生成摘要文件名", "语义一致性", "信息覆盖度", "表达相似性",
                    "简洁性与冗余度", "逻辑结构", "总体相似度评分", "来源文件"]]

    all_results.append(result_df)

# 合并所有 CSV 文件处理后的数据
combined_df = pd.concat(all_results, ignore_index=True)
print(combined_df)

# 将合并后的结果保存为 CSV 文件
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"合并后的结果已保存到 {output_file}")

