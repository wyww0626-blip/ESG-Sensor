import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import re
import time
import os
import csv
from zhipuai import ZhipuAI

# 初始化 API
client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")

# 缓存嵌入结果
cache_dict = {}

def zhipu_embedding(inputs, batch_size=64):
    embeddings = []
    new_inputs = []
    for text in inputs:
        if text in cache_dict:
            embeddings.append(cache_dict[text])
        else:
            new_inputs.append(text)
    if new_inputs:
        for i in range(0, len(new_inputs), batch_size):
            batch = new_inputs[i:i + batch_size]
            response = client.embeddings.create(
                model="embedding-3",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            for j, text in enumerate(batch):
                cache_dict[text] = batch_embeddings[j]
                embeddings.append(batch_embeddings[j])
    return embeddings

def cosine_similarity_parallel(embeddings1, embeddings2):
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    dot_products = np.dot(embeddings1, embeddings2.T)
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True).T
    return dot_products / (norms1 * norms2)

def calculate_similarity_ave(list1, list2):
    embeddings_list1 = zhipu_embedding(list1)
    embeddings_list2 = zhipu_embedding(list2)
    with ThreadPoolExecutor() as executor:
        similarities_matrix = executor.submit(
            cosine_similarity_parallel,
            embeddings_list1,
            embeddings_list2
        ).result()
    sorted_similarities = np.sort(similarities_matrix, axis=0)[1:]
    avg_similarities = np.mean(sorted_similarities, axis=0)
    sorted_indices = np.argsort(avg_similarities)[::-1]
    sorted_results = [(list2[i], avg_similarities[i]) for i in sorted_indices]
    return sorted_results

# ✅ 修复后的extract_titles_from_pdf函数
def extract_titles_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    titles = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if line["spans"]:  # 确保 spans 非空，防止IndexError
                        text = "".join(span["text"] for span in line["spans"])
                        font_size = line["spans"][0]["size"]
                        if font_size > 11:
                            titles.append(f"Page {page_num + 1}: {text.strip()}")
    return titles

def create_sorted_pdf(input_pdf_path, output_pdf_path, sorted_page_numbers):
    original_pdf = fitz.open(input_pdf_path)
    new_pdf = fitz.open()
    for page_num in sorted_page_numbers:
        page = original_pdf.load_page(page_num - 1)
        new_pdf.insert_pdf(original_pdf, from_page=page_num - 1, to_page=page_num - 1)
    new_pdf.save(output_pdf_path)
    print(f"✅ 新 PDF 已保存: {output_pdf_path}")

if __name__ == "__main__":
    folder_path = "含有利益相关方列表或者坐标系的ESG报告"
    output_folder = "处理后的_ESG报告"
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [
        f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() == ".pdf"
    ]

    biaozhun_list = [
        "实质性议题分析", "对利益相关方的重要性", "对公司发展的重要性", "实质性议题分析矩阵",
        "实质性议题识别与分析", "实质性议题矩阵", "实质性评估", "ESG重大性议题",
        "聚焦实质性议题", "实质性分析", "实质性议题", "实质性议题识别",
        "重要议题识别", "关键实质性议题", "重要议题评估"
    ]

    # 创建time.csv文件并写入表头
    with open('time.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["文件名", "处理时间(秒)"])

    for pdf_file in pdf_files:
        start_time = time.time()
        pdf_path = os.path.join(folder_path, pdf_file)
        output_pdf_path = os.path.join(output_folder, f"相关性排序后的_{pdf_file}")

        print(f"📄 处理中: {pdf_file} ...")

        titles = extract_titles_from_pdf(pdf_path)
        sorted_results = calculate_similarity_ave(biaozhun_list, titles)

        sorted_page_numbers = []
        for item, similarity in sorted_results[:5]:
            match = re.search(r'Page (\d+)', item)
            if match:
                page_number = int(match.group(1))
                if page_number not in sorted_page_numbers:
                    sorted_page_numbers.append(page_number)

        print(f"📌 按相似度排序的相关页面: {sorted_page_numbers}")

        create_sorted_pdf(pdf_path, output_pdf_path, sorted_page_numbers)

        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ {pdf_file} 处理完成！⏱ 用时: {duration:.2f}s\n")

        # 将处理时间追加到time.csv
        with open('time.csv', mode='a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([pdf_file, f"{duration:.2f}"])

    print("🎉 所有 PDF 文件处理完毕，处理时间已记录至 time.csv！")

