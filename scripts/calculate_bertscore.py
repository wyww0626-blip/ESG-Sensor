# -*- coding: utf-8 -*-
import os
import numpy as np
from dotenv import load_dotenv
from zhipuai import ZhipuAI
import chardet
import time
import pandas as pd

# 加载 .env 文件中的环境变量
# load_dotenv()

# zhipu_embedding 函数：获取文本的 embedding，并截断到最大长度（这里设置为3072个字符）
def zhipu_embedding(input_texts):
    max_length = 3072
    input_texts = [text[:max_length] for text in input_texts]

    client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")  # 请在此处填写你的 API Key
    response = client.embeddings.create(
        model="embedding-3",
        input=input_texts,
    )
    if isinstance(response.data, list):
        return [np.array(item.embedding) for item in response.data]
    else:
        return [np.array(response.data.embedding)]

# 计算两个向量列表之间的余弦相似度矩阵
def cosine_similarity_matrix(embeddings1, embeddings2):
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    matrix = np.dot(embeddings1, embeddings2.T)
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    return matrix / (norm1 * norm2.T)

# 根据两个向量计算 Precision, Recall, F1 Score
# 注意：如果每个文件只产生一个整体 embedding，则得到的 1×1 相似度矩阵使得 precision 与 recall 值相同，从而 F1 与它们一致。
def calculate_similarity_scores(embedding1, embedding2):
    cos_sim_matrix = cosine_similarity_matrix([embedding1], [embedding2])
    precision = np.mean(np.max(cos_sim_matrix, axis=1))
    recall = np.mean(np.max(cos_sim_matrix, axis=0))
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

# 自动检测文件编码并读取文本
def read_text(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        return file.read()

# 比较两个文本文件，返回 Precision, Recall, F1 Score
def compare_text_files(file1, file2):
    text1 = read_text(file1)
    text2 = read_text(file2)
    embeddings = zhipu_embedding([text1, text2])
    precision, recall, f1_score = calculate_similarity_scores(embeddings[0], embeddings[1])
    return precision, recall, f1_score

# 计算单个文件夹中所有文件对的平均相似度
# 先将所有文件对的 Precision、Recall、F1 保存到列表中，然后如果数量大于1，去掉各自最低的值再计算平均
def calculate_average_similarity(folder1, folder2):
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(".txt")])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith(".txt")])

    precision_list, recall_list, f1_score_list = [], [], []

    # 遍历时取最小文件数量对应的文件对
    for file1, file2 in zip(files1, files2):
        file1_path = os.path.join(folder1, file1)
        file2_path = os.path.join(folder2, file2)
        if os.path.isfile(file1_path) and os.path.isfile(file2_path):
            precision, recall, f1_score = compare_text_files(file1_path, file2_path)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            print(f"文件 {file1} 和 {file2} 的 Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    file_count = len(f1_score_list)
    if file_count == 0:
        return None, None, None, 0

    # 若文件对数量大于1，则去掉最低的一个值
    def remove_lowest(values):
        if len(values) > 1:
            values_copy = values.copy()
            values_copy.remove(min(values_copy))
            return values_copy
        else:
            return values

    precision_list_filtered = remove_lowest(precision_list)
    recall_list_filtered = remove_lowest(recall_list)
    f1_score_list_filtered = remove_lowest(f1_score_list)

    average_precision = np.mean(precision_list_filtered)
    average_recall = np.mean(recall_list_filtered)
    average_f1_score = np.mean(f1_score_list_filtered)
    return average_precision, average_recall, average_f1_score, file_count

# 主程序：遍历多个子文件夹对进行评估，并结构化输出每个子文件夹的 Precision、Recall、F1 以及总体平均指标
if __name__ == "__main__":
    # 父目录路径（根据实际情况修改）
    generated_parent = r"zhipu_result_dp"
    reference_parent = r"ESG基准v1-zhipu_dp"

    # 获取两个父目录下的所有子文件夹（排序保证顺序对应）
    generated_subfolders = sorted([d for d in os.listdir(generated_parent) if os.path.isdir(os.path.join(generated_parent, d))])
    reference_subfolders = sorted([d for d in os.listdir(reference_parent) if os.path.isdir(os.path.join(reference_parent, d))])

    if len(generated_subfolders) != len(reference_subfolders):
        print(f"⚠️ 子文件夹数量不一致：生成摘要文件夹有 {len(generated_subfolders)} 个，而参考摘要文件夹有 {len(reference_subfolders)} 个。将按最小数量进行处理。")

    num_pairs = min(len(generated_subfolders), len(reference_subfolders))

    results = []
    total_f1_sum = 0.0
    total_prec_sum = 0.0
    total_recall_sum = 0.0
    total_file_count = 0

    # 遍历每对子文件夹（按顺序对应）
    for i in range(num_pairs):
        gen_folder = os.path.join(generated_parent, generated_subfolders[i])
        ref_folder = os.path.join(reference_parent, reference_subfolders[i])
        print(f"\n正在评估子文件夹对：生成摘要文件夹【{generated_subfolders[i]}】 对 参考摘要文件夹【{reference_subfolders[i]}】")
        avg_prec, avg_recall, avg_f1, file_count = calculate_average_similarity(gen_folder, ref_folder)
        if file_count == 0:
            print("❌ 此对子文件夹中没有匹配的文件。跳过。")
            continue

        results.append({
            "参考文件夹": reference_subfolders[i],
            "生成文件夹": generated_subfolders[i],
            "文件数量": file_count,
            "平均Precision": avg_prec,
            "平均Recall": avg_recall,
            "平均F1": avg_f1
        })
        total_prec_sum += avg_prec * file_count
        total_recall_sum += avg_recall * file_count
        total_f1_sum += avg_f1 * file_count
        total_file_count += file_count

        # 固定延时，防止请求过于密集
        time.sleep(1)

    overall_avg_prec = total_prec_sum / total_file_count if total_file_count > 0 else 0
    overall_avg_recall = total_recall_sum / total_file_count if total_file_count > 0 else 0
    overall_avg_f1 = total_f1_sum / total_file_count if total_file_count > 0 else 0
    print(f"\n所有文件夹的总体平均 Precision: {overall_avg_prec:.4f}, Recall: {overall_avg_recall:.4f}, F1 Score: {overall_avg_f1:.4f}")

    # 将每个子文件夹评估结果保存为 CSV 文件
    output_file = "zhipu_all_folders_evaluation_results.csv"
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 所有子文件夹评估结果已保存至：{output_file}")
    else:
        print("❌ 没有生成任何评估结果。")

