import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import re
import time
import os
import csv  # 新增导入csv模块
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
    """
    新的相似度计算策略：
    对每个页面标题，计算其与所有标准句子的余弦相似度，
    然后直接取最大相似度作为该标题的得分，再按得分降序排序。
    """
    embeddings_list1 = zhipu_embedding(list1)
    embeddings_list2 = zhipu_embedding(list2)
    with ThreadPoolExecutor() as executor:
        similarities_matrix = executor.submit(cosine_similarity_parallel, embeddings_list1, embeddings_list2).result()
    # 取每个标题与标准句子之间的最大相似度作为得分
    max_similarities = np.max(similarities_matrix, axis=0)
    sorted_indices = np.argsort(max_similarities)[::-1]
    sorted_results = [(list2[i], max_similarities[i]) for i in sorted_indices]
    return sorted_results

def extract_titles_from_pdf(pdf_path):
    """
    新的标题识别策略：
    对每一页中所有字号进行分组统计字符数，
    如果某个字号组的总字符数小于30，则认为该组为候选标题，
    如果有多个候选标题，则全部输出，并附带页码。
    """
    pdf_document = fitz.open(pdf_path)
    titles = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_dict = page.get_text("dict")
        # 按字号分组，存储文本出现顺序及累计字符数
        font_groups = {}  # key: font size, value: {"texts": [(order, text), ...], "total_chars": int}
        order_counter = 0
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    fs = span["size"]
                    if fs not in font_groups:
                        font_groups[fs] = {"texts": [], "total_chars": 0}
                    font_groups[fs]["texts"].append((order_counter, text))
                    font_groups[fs]["total_chars"] += len(text)
                    order_counter += 1
        # 遍历各字号组，若累计字符数小于30，则视为候选标题
        for fs, data in font_groups.items():
            if data["total_chars"] < 30:
                texts_sorted = sorted(data["texts"], key=lambda x: x[0])
                candidate_title = " ".join(text for _, text in texts_sorted)
                titles.append(f"Page {page_num + 1}: {candidate_title}")
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
    folder_path = "含有实质性议题"
    output_folder = "处理后的_ESG报告_优化"
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() == ".pdf"]

    biaozhun_list = [
        "实质性议题分析", "对利益相关方的重要性", "对公司发展的重要性", "实质性议题分析矩阵",
        "实质性议题识别与分析", "实质性议题矩阵", "实质性评估", "ESG重大性议题",
        "聚焦实质性议题", "实质性分析", "实质性议题", "实质性议题识别",
        "重要议题识别", "关键实质性议题", "重要议题评估"
    ]

    # 创建并初始化time.csv文件
    with open('time.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["文件名", "处理时间(秒)"])

    for pdf_file in pdf_files:
        start_time = time.time()
        pdf_path = os.path.join(folder_path, pdf_file)
        output_pdf_path = os.path.join(output_folder, f"{pdf_file}")

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

        # 处理完成后，将处理时间追加到CSV文件
        with open('time.csv', mode='a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([pdf_file, f"{duration:.2f}"])

    print("🎉 所有 PDF 文件处理完毕，处理时间已记录至 time.csv！")

