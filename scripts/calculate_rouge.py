# -*- coding: utf-8 -*-
import os
import chardet
import time
import pandas as pd
from rouge_score import rouge_scorer
import numpy as np


# 自动检测文件编码并读取文本内容
def read_text(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()


# 比较两个文本文件，计算 ROUGE 指标
def compare_text_files(file1, file2):
    ref_text = read_text(file1)
    cand_text = read_text(file2)
    # 初始化 ROUGE scorer，计算 rouge1, rouge2, rougeL，使用词干化
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref_text, cand_text)
    # 返回每个指标的 (precision, recall, f1)
    rouge1 = (scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure)
    rouge2 = (scores['rouge2'].precision, scores['rouge2'].recall, scores['rouge2'].fmeasure)
    rougeL = (scores['rougeL'].precision, scores['rougeL'].recall, scores['rougeL'].fmeasure)
    return rouge1, rouge2, rougeL


# 去掉列表中最低的值（如果数量大于1）
def remove_lowest(values):
    if len(values) > 1:
        values_copy = values.copy()
        values_copy.remove(min(values_copy))
        return values_copy
    else:
        return values


# 计算单个文件夹中所有文件对的平均 ROUGE 指标
# 针对每个指标（rouge1、rouge2、rougeL），分别保存 Precision、Recall、F1 的列表，
# 若文件对数量大于1，则去掉各自最低值后再计算平均。
def calculate_average_similarity(folder1, folder2):
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(".txt")])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith(".txt")])

    # 初始化各指标的列表
    rouge1_prec_list, rouge1_rec_list, rouge1_f1_list = [], [], []
    rouge2_prec_list, rouge2_rec_list, rouge2_f1_list = [], [], []
    rougeL_prec_list, rougeL_rec_list, rougeL_f1_list = [], [], []

    # 遍历时按顺序对应两个文件夹中的文件
    for file1, file2 in zip(files1, files2):
        file1_path = os.path.join(folder1, file1)
        file2_path = os.path.join(folder2, file2)
        if os.path.isfile(file1_path) and os.path.isfile(file2_path):
            r1, r2, rL = compare_text_files(file1_path, file2_path)
            # r1, r2, rL 分别为 (precision, recall, f1)
            rouge1_prec_list.append(r1[0])
            rouge1_rec_list.append(r1[1])
            rouge1_f1_list.append(r1[2])

            rouge2_prec_list.append(r2[0])
            rouge2_rec_list.append(r2[1])
            rouge2_f1_list.append(r2[2])

            rougeL_prec_list.append(rL[0])
            rougeL_rec_list.append(rL[1])
            rougeL_f1_list.append(rL[2])

            print(f"文件 {file1} 与 {file2} 的 ROUGE 指标：")
            print(f"  ROUGE-1 -> Precision: {r1[0]:.4f}, Recall: {r1[1]:.4f}, F1: {r1[2]:.4f}")
            print(f"  ROUGE-2 -> Precision: {r2[0]:.4f}, Recall: {r2[1]:.4f}, F1: {r2[2]:.4f}")
            print(f"  ROUGE-L -> Precision: {rL[0]:.4f}, Recall: {rL[1]:.4f}, F1: {rL[2]:.4f}")

    file_count = len(rouge1_f1_list)
    if file_count == 0:
        return None, None, None, None, None, None, None, None, None, 0

    # 对每个指标列表去掉最低值（如果有多于1个数据）
    rouge1_prec_list = remove_lowest(rouge1_prec_list)
    rouge1_rec_list = remove_lowest(rouge1_rec_list)
    rouge1_f1_list = remove_lowest(rouge1_f1_list)

    rouge2_prec_list = remove_lowest(rouge2_prec_list)
    rouge2_rec_list = remove_lowest(rouge2_rec_list)
    rouge2_f1_list = remove_lowest(rouge2_f1_list)

    rougeL_prec_list = remove_lowest(rougeL_prec_list)
    rougeL_rec_list = remove_lowest(rougeL_rec_list)
    rougeL_f1_list = remove_lowest(rougeL_f1_list)

    avg_r1_prec = np.mean(rouge1_prec_list)
    avg_r1_rec = np.mean(rouge1_rec_list)
    avg_r1_f1 = np.mean(rouge1_f1_list)

    avg_r2_prec = np.mean(rouge2_prec_list)
    avg_r2_rec = np.mean(rouge2_rec_list)
    avg_r2_f1 = np.mean(rouge2_f1_list)

    avg_rL_prec = np.mean(rougeL_prec_list)
    avg_rL_rec = np.mean(rougeL_rec_list)
    avg_rL_f1 = np.mean(rougeL_f1_list)

    return (avg_r1_prec, avg_r1_rec, avg_r1_f1,
            avg_r2_prec, avg_r2_rec, avg_r2_f1,
            avg_rL_prec, avg_rL_rec, avg_rL_f1, file_count)


# 主程序：遍历多个子文件夹对进行评估，并结构化输出每个子文件夹的 ROUGE 指标及总体平均指标
if __name__ == "__main__":
    # 父目录路径（根据实际情况修改）
    generated_parent = r"deepseek_result_dp"
    reference_parent = r"ESG基准v1-deepseek_dp"

    # 获取两个父目录下的所有子文件夹（按排序后顺序对应）
    generated_subfolders = sorted(
        [d for d in os.listdir(generated_parent) if os.path.isdir(os.path.join(generated_parent, d))])
    reference_subfolders = sorted(
        [d for d in os.listdir(reference_parent) if os.path.isdir(os.path.join(reference_parent, d))])

    if len(generated_subfolders) != len(reference_subfolders):
        print(
            f"⚠️ 子文件夹数量不一致：生成摘要文件夹有 {len(generated_subfolders)} 个，而参考摘要文件夹有 {len(reference_subfolders)} 个。将按最小数量处理。")

    num_pairs = min(len(generated_subfolders), len(reference_subfolders))

    results = []
    total_r1_prec = total_r1_rec = total_r1_f1 = 0.0
    total_r2_prec = total_r2_rec = total_r2_f1 = 0.0
    total_rL_prec = total_rL_rec = total_rL_f1 = 0.0
    total_file_count = 0

    # 遍历每对子文件夹（按排序后顺序对应）
    for i in range(num_pairs):
        gen_folder = os.path.join(generated_parent, generated_subfolders[i])
        ref_folder = os.path.join(reference_parent, reference_subfolders[i])
        print(
            f"\n正在评估子文件夹对：生成摘要文件夹【{generated_subfolders[i]}】 对 参考摘要文件夹【{reference_subfolders[i]}】")
        (avg_r1_prec, avg_r1_rec, avg_r1_f1,
         avg_r2_prec, avg_r2_rec, avg_r2_f1,
         avg_rL_prec, avg_rL_rec, avg_rL_f1, file_count) = calculate_average_similarity(gen_folder, ref_folder)
        if file_count == 0:
            print("❌ 此对子文件夹中没有匹配的文件，跳过。")
            continue

        results.append({
            "参考文件夹": reference_subfolders[i],
            "生成文件夹": generated_subfolders[i],
            "文件数量": file_count,
            "平均Rouge1 Precision": avg_r1_prec,
            "平均Rouge1 Recall": avg_r1_rec,
            "平均Rouge1 F1": avg_r1_f1,
            "平均Rouge2 Precision": avg_r2_prec,
            "平均Rouge2 Recall": avg_r2_rec,
            "平均Rouge2 F1": avg_r2_f1,
            "平均RougeL Precision": avg_rL_prec,
            "平均RougeL Recall": avg_rL_rec,
            "平均RougeL F1": avg_rL_f1
        })
        total_r1_prec += avg_r1_prec * file_count
        total_r1_rec += avg_r1_rec * file_count
        total_r1_f1 += avg_r1_f1 * file_count

        total_r2_prec += avg_r2_prec * file_count
        total_r2_rec += avg_r2_rec * file_count
        total_r2_f1 += avg_r2_f1 * file_count

        total_rL_prec += avg_rL_prec * file_count
        total_rL_rec += avg_rL_rec * file_count
        total_rL_f1 += avg_rL_f1 * file_count

        total_file_count += file_count

        # 固定延时，防止请求过于密集
        time.sleep(1)

    overall_avg_r1_prec = total_r1_prec / total_file_count if total_file_count > 0 else 0
    overall_avg_r1_rec = total_r1_rec / total_file_count if total_file_count > 0 else 0
    overall_avg_r1_f1 = total_r1_f1 / total_file_count if total_file_count > 0 else 0

    overall_avg_r2_prec = total_r2_prec / total_file_count if total_file_count > 0 else 0
    overall_avg_r2_rec = total_r2_rec / total_file_count if total_file_count > 0 else 0
    overall_avg_r2_f1 = total_r2_f1 / total_file_count if total_file_count > 0 else 0

    overall_avg_rL_prec = total_rL_prec / total_file_count if total_file_count > 0 else 0
    overall_avg_rL_rec = total_rL_rec / total_file_count if total_file_count > 0 else 0
    overall_avg_rL_f1 = total_rL_f1 / total_file_count if total_file_count > 0 else 0

    print(
        f"\n所有文件夹的总体平均 ROUGE-1 -> Precision: {overall_avg_r1_prec:.4f}, Recall: {overall_avg_r1_rec:.4f}, F1: {overall_avg_r1_f1:.4f}")
    print(
        f"所有文件夹的总体平均 ROUGE-2 -> Precision: {overall_avg_r2_prec:.4f}, Recall: {overall_avg_r2_rec:.4f}, F1: {overall_avg_r2_f1:.4f}")
    print(
        f"所有文件夹的总体平均 ROUGE-L -> Precision: {overall_avg_rL_prec:.4f}, Recall: {overall_avg_rL_rec:.4f}, F1: {overall_avg_rL_f1:.4f}")

    # 将每个子文件夹的评估结果保存为 CSV 文件
    output_file = "deepseek_result_all_folders_evaluation_results_rouge.csv"
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 所有子文件夹评估结果已保存至：{output_file}")
    else:
        print("❌ 没有生成任何评估结果。")

