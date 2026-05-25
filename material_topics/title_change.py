import fitz  # PyMuPDF
import numpy as np
from zhipuai import ZhipuAI
import os
import time
import csv
from collections import defaultdict

# 初始化 zhipuai API 客户端（请填写你的 API_KEY）
client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")
cache_dict = {}


def zhipu_embedding(inputs, batch_size=64):
    embeddings = [None] * len(inputs)
    uncached_texts = []
    uncached_indices = []

    for idx, text in enumerate(inputs):
        if text in cache_dict:
            embeddings[idx] = cache_dict[text]
        else:
            uncached_texts.append(text)
            uncached_indices.append(idx)

    if uncached_texts:
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            response = client.embeddings.create(
                model="embedding-3",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            for j, emb in enumerate(batch_embeddings):
                original_idx = uncached_indices[i + j]
                embeddings[original_idx] = emb
                cache_dict[uncached_texts[i + j]] = emb

    return embeddings


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_titles_by_font_size(pdf_path, char_limit=40):
    pdf_document = fitz.open(pdf_path)
    titles_with_pages = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        page_dict = page.get_text("dict")
        font_groups = defaultdict(list)

        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    fs = span["size"]
                    font_groups[fs].append(text)

        font_size_char_count = {fs: sum(len(t) for t in texts) for fs, texts in font_groups.items()}

        for fs, total_chars in font_size_char_count.items():
            if total_chars < char_limit:
                for title_text in font_groups[fs]:
                    if title_text.strip():
                        titles_with_pages.append((page_num + 1, title_text.strip()))

    return titles_with_pages


if __name__ == "__main__":
    pdf_folder = "含有实质性议题"
    output_folder = "materials_topics_results_40"
    os.makedirs(output_folder, exist_ok=True)

    biaozhun_list = [
        "实质性议题分析", "对利益相关方的重要性", "对公司发展的重要性", "实质性议题分析矩阵",
        "实质性议题识别与分析", "实质性议题矩阵", "实质性评估", "ESG重大性议题",
        "聚焦实质性议题", "实质性分析", "实质性议题", "实质性议题识别",
        "重要议题识别", "关键实质性议题", "重要议题评估"
    ]
    biaozhun_embeddings = zhipu_embedding(biaozhun_list)

    # CSV文件路径
    csv_path = os.path.join(output_folder, "results.csv")
    with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["文件名", "页码", "标题", "得分", "处理耗时(秒)"])

        for filename in os.listdir(pdf_folder):
            if not filename.lower().endswith('.pdf'):
                continue

            pdf_path = os.path.join(pdf_folder, filename)
            print(f"\n开始处理文件: {filename}")
            start_time = time.time()

            candidate_titles = extract_titles_by_font_size(pdf_path, char_limit=40)
            candidate_texts = [text for _, text in candidate_titles]

            if not candidate_texts:
                print("未找到候选标题。")
                continue

            candidate_embeddings = zhipu_embedding(candidate_texts)

            candidate_scores = []
            for i, cand_emb in enumerate(candidate_embeddings):
                scores = [cosine_similarity(cand_emb, b_emb) for b_emb in biaozhun_embeddings]
                scores.sort()
                avg_score = np.mean(scores[1:]) if len(scores) > 1 else scores[0]
                page = candidate_titles[i][0]
                candidate_scores.append((page, candidate_texts[i], avg_score))

            candidate_scores.sort(key=lambda x: x[2], reverse=True)
            top_five = candidate_scores[:5]

            print("与标准列表相似度最高的前七个候选标题：")
            for page, title, score in top_five:
                print(f"Page {page}: {title}  得分: {score:.4f}")

            end_time = time.time()
            duration = end_time - start_time
            print(f"处理文件 '{filename}' 耗时: {duration:.2f}秒")

            # 将结果写入CSV
            for page, title, score in top_five:
                csv_writer.writerow([filename, page, title, f"{score:.4f}", f"{duration:.2f}"])

            # 创建按相似度排序的新PDF
            sorted_pages = [page for page, _, _ in top_five]
            original_doc = fitz.open(pdf_path)
            new_pdf = fitz.open()

            added_pages = set()
            for page_num in sorted_pages:
                if page_num - 1 not in added_pages:
                    new_pdf.insert_pdf(original_doc, from_page=page_num - 1, to_page=page_num - 1)
                    added_pages.add(page_num - 1)

            # 保存新PDF
            new_pdf_path = os.path.join(output_folder, filename)
            new_pdf.save(new_pdf_path)
            new_pdf.close()
            original_doc.close()

            print(f"重排序页面PDF文件已保存至: {new_pdf_path}")


