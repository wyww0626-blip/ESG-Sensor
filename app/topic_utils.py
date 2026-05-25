import base64
import fitz  # PyMuPDF
import json
import os
import time
from collections import defaultdict
from zhipuai import ZhipuAI
from dotenv import load_dotenv

import numpy as np

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY", "")) if os.getenv("ZHIPU_API_KEY") else None
cache_dict = {}

def zhipu_embedding(inputs, batch_size=64):
    if client is None:
        raise RuntimeError("Missing ZHIPU_API_KEY. Copy .env.example to .env and set your own key.")
    embeddings = [None] * len(inputs)
    uncached_texts, uncached_indices = [], []
    for idx, text in enumerate(inputs):
        if text in cache_dict:
            embeddings[idx] = cache_dict[text]
        else:
            uncached_texts.append(text)
            uncached_indices.append(idx)
    if uncached_texts:
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            response = client.embeddings.create(model="embedding-3", input=batch)
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
    pdf_document.close()
    return titles_with_pages

def similarity_top3_average(candidate_embeddings, biaozhun_embeddings):
    candidate_scores = []
    for cand_emb in candidate_embeddings:
        scores = [cosine_similarity(cand_emb, b_emb) for b_emb in biaozhun_embeddings]
        scores.sort(reverse=True)
        final_score = np.mean(scores[:3])
        candidate_scores.append(final_score)
    return candidate_scores

def get_pdf_bytes_from_pages(pdf_path, selected_pages):
    original_doc = fitz.open(pdf_path)
    new_pdf = fitz.open()
    for page in selected_pages:
        new_pdf.insert_pdf(original_doc, from_page=page - 1, to_page=page - 1)
    pdf_bytes = new_pdf.write()
    new_pdf.close()
    original_doc.close()
    return pdf_bytes

def pdf_page_to_base64_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_base64_list = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        image_bytes = pix.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_base64_list.append(image_base64)
    doc.close()
    return image_base64_list

def analyze_pdf(pdf_path, fewshot_path):
    # 准备基准关键词
    biaozhun_list = [
        "实质性议题分析", "对利益相关方的重要性", "对公司发展的重要性", "实质性议题分析矩阵",
        "实质性议题分析与识别", "实质性议题矩阵", "实质性评估", "ESG重大性议题",
        "聚焦实质性议题", "实质性分析", "实质性议题", "实质性议题识别",
        "重要议题识别", "关键实质性议题", "重要议题评估"
    ]
    biaozhun_embeddings = zhipu_embedding(biaozhun_list)

    candidate_titles = extract_titles_by_font_size(pdf_path)
    if not candidate_titles:
        return None, None, "❌ 未识别到疑似标题，跳过分析"

    texts = [t for _, t in candidate_titles]
    text_embeddings = zhipu_embedding(texts)
    scores = similarity_top3_average(text_embeddings, biaozhun_embeddings)

    candidate_scores = [(candidate_titles[i][0], texts[i], scores[i]) for i in range(len(scores))]
    candidate_scores.sort(key=lambda x: x[2], reverse=True)
    top_pages = [item[0] for item in candidate_scores[:3]]

    pdf_bytes = get_pdf_bytes_from_pages(pdf_path, top_pages)
    images = pdf_page_to_base64_images(pdf_bytes)

    # 加载 fewshot 示例
    with open(fewshot_path, "r", encoding="utf-8") as f:
        fewshot_data = json.load(f)

    # 构造对话消息体
    content_list = []
    for item in fewshot_data:
        with open(item['img'], 'rb') as img_f:
            few_shot_img_base64 = base64.b64encode(img_f.read()).decode('utf-8')
        content_list.append({
            "type": "image_url",
            "image_url": {"url": few_shot_img_base64}
        })
        content_list.append({
            "type": "text",
            "text": f"{item['prompt']}\n{item['label']}"
        })

    for img64 in images:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": img64}
        })

    content_list.append({
        "type": "text",
        "text": "请注意只需要提取界面中的议题矩阵中所包含的议题，并分析其重要性，如果界面中没有议题矩阵，则直接输出“无””。"
    })

    try:
        response = client.chat.completions.create(
            model="glm-4v-plus-0111",
            messages=[{"role": "user", "content": content_list}]
        )
        result = response.choices[0].message.content.strip()
        return images, result, "✅ 分析完成"
    except Exception as e:
        return images, f"API调用失败：{str(e)}", "❌ 调用失败"

