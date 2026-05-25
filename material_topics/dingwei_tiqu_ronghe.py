import base64
import json
import fitz  # PyMuPDF
import os
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from zhipuai import ZhipuAI

# -----------------------------
# 公共工具函数及初始化
# -----------------------------

# API密钥，请替换为你的真实 API_KEY
client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")
cache_dict = {}

def zhipu_embedding(inputs, batch_size=64):
    """调用芝麻 AI 进行文本嵌入，内置缓存"""
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
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extract_titles_by_font_size(pdf_path, char_limit=40):
    """
    根据字体大小和字符总数来提取疑似“标题”文本。
    若某一字体总共出现的字符数 < char_limit，则认为是可能的候选标题。
    返回列表：[(页码, 标题文本), ...]
    """
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
        # 统计每个字体大小对应的字符总数
        font_size_char_count = {fs: sum(len(t) for t in texts) for fs, texts in font_groups.items()}
        # 若某一字体的字符总数 < char_limit，则将其对应的文本作为候选标题
        for fs, total_chars in font_size_char_count.items():
            if total_chars < char_limit:
                for title_text in font_groups[fs]:
                    if title_text.strip():
                        titles_with_pages.append((page_num + 1, title_text.strip()))
    pdf_document.close()
    return titles_with_pages

def similarity_top3_average(candidate_embeddings, biaozhun_embeddings):
    """
    使用 Top‑3 平均相似度计算策略，对候选标题向量与基准向量计算相似度。
    返回每个候选标题的得分列表。
    """
    candidate_scores = []
    for cand_emb in candidate_embeddings:
        scores = [cosine_similarity(cand_emb, b_emb) for b_emb in biaozhun_embeddings]
        scores.sort(reverse=True)
        final_score = np.mean(scores[:3])
        candidate_scores.append(final_score)
    return candidate_scores

def image_bytes_to_base64(image_bytes):
    """图片二进制转base64字符串"""
    return base64.b64encode(image_bytes).decode('utf-8')

def pdf_page_to_image_bytes_from_doc(pdf_doc, page_number, zoom=2.0):
    """从已打开的 PDF 文档中，将指定页转换为图片二进制"""
    page = pdf_doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes()

# -----------------------------
# 第一部分：提取 Top-3 页面，生成新 PDF（内存中保存）
# -----------------------------

def extract_top3_pdf(pdf_path, biaozhun_embeddings):
    """
    从原始 PDF 中提取疑似标题页面（使用 Top‑3 平均相似度策略），
    选择相似度最高的 5 页，合并为新的 PDF。
    返回新 PDF 的二进制数据（bytes）；若未找到候选标题则返回 None。
    """
    # 1. 提取候选标题
    candidate_titles = extract_titles_by_font_size(pdf_path, char_limit=40)
    candidate_texts = [text for _, text in candidate_titles]
    if not candidate_texts:
        print(f"文件 {os.path.basename(pdf_path)} 未找到候选标题。")
        return None

    # 2. 获取候选标题的向量
    candidate_embeddings = zhipu_embedding(candidate_texts)
    # 3. 计算相似度（Top-3 平均）
    scores = similarity_top3_average(candidate_embeddings, biaozhun_embeddings)
    # 整理标题与得分，按相似度从高到低排序
    candidate_scores = [(candidate_titles[i][0], candidate_texts[i], scores[i])
                        for i in range(len(scores))]
    candidate_scores.sort(key=lambda x: x[2], reverse=True)
    top_five = candidate_scores[:5]
    print(f"文件 {os.path.basename(pdf_path)} 提取到相似度最高的页面：{[(p, s) for p, _, s in top_five]}")

    # 4. 从原始 PDF 中抽取上述页面，合并为新的 PDF
    original_doc = fitz.open(pdf_path)
    new_pdf = fitz.open()  # 新建空白 PDF
    added_pages = set()
    for page_num, _, _ in top_five:
        if page_num - 1 not in added_pages:
            new_pdf.insert_pdf(original_doc, from_page=page_num - 1, to_page=page_num - 1)
            added_pages.add(page_num - 1)
    # 获取新 PDF 的二进制数据
    pdf_bytes = new_pdf.write()
    new_pdf.close()
    original_doc.close()
    return pdf_bytes

# -----------------------------
# 第二部分：使用 glm-4v 对新 PDF 进行分析
# -----------------------------

def process_pdf_with_glm4v(pdf_bytes, file_name, fewshot_data, client):
    """
    对输入的 PDF（二进制数据）进行逐页分析：
    将每页转换为图片后，调用 glm-4v 接口判断是否存在有效议题矩阵。
    如果检测到有效内容，则返回结果；否则返回 “无”。
    """
    # 从内存中打开 PDF（需指定 filetype）
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)
    print(f"\n开始处理内存 PDF：{file_name}，共 {total_pages} 页")
    start_time = time.time()
    found_result = "无"
    processed_page_count = 0

    for page_number in range(total_pages):
        processed_page_count += 1
        image_bytes = pdf_page_to_image_bytes_from_doc(pdf_document, page_number)
        # 构造消息体，包含 few-shot 示例及当前页图片
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
        content_list.append({
            "type": "image_url",
            "image_url": {"url": image_bytes_to_base64(image_bytes)}
        })
        content_list.append({
            "type": "text",
            "text": "请你帮我判断这张图中是否有议题矩阵，如果有则分析这张ESG报告图中展示了什么内容？如果没有则只需要输出“无”。"
        })

        try:
            response = client.chat.completions.create(
                model="glm-4v-plus-0111",
                messages=[{"role": "user", "content": content_list}]
            )
            result = response.choices[0].message.content.strip()
            print(f"  文件【{file_name}】第 {page_number + 1}/{total_pages} 页结果：{result}")
            if result != "无":
                found_result = result
                print("  🟢 检测到有效议题矩阵，停止后续页面分析。\n")
                break
            # 延时，避免频率过高
            time.sleep(1)
        except Exception as e:
            print(f"  🔴 API调用错误，停止该文件处理：{e}")
            found_result = f"API调用错误：{e}"
            break

    elapsed_time = time.time() - start_time
    pdf_document.close()
    return {
        '文件名': file_name,
        '分析结果': found_result,
        '处理页数': processed_page_count,
        '处理耗时(秒)': round(elapsed_time, 2)
    }

# -----------------------------
# 主流程：只处理单个 PDF 文件
# -----------------------------

if __name__ == "__main__":
    # 加载 few-shot 示例数据（注意：确保路径正确）
    with open('img/dataset.json', 'r', encoding='utf-8') as f:
        fewshot_data = json.load(f)

    # 基准关键词列表（可根据实际需要调整）
    biaozhun_list = [
        "实质性议题分析", "对利益相关方的重要性", "对公司发展的重要性", "实质性议题分析矩阵",
        "实质性议题分析与识别", "实质性议题矩阵", "实质性评估", "ESG重大性议题",
        "聚焦实质性议题", "实质性分析", "实质性议题", "实质性议题识别",
        "重要议题识别", "关键实质性议题", "重要议题评估"
    ]
    # 获取基准关键词向量
    biaozhun_embeddings = zhipu_embedding(biaozhun_list)

    # 指定单个 PDF 文件所在目录及文件名（请根据实际情况修改）
    input_pdf_dir = "含有实质性议题"
    single_pdf_file = "1.电靓陇原2021—国网甘肃省电力公司服务幸福美好新甘肃履责行动书.pdf"  # 替换为你要处理的 PDF 文件名
    pdf_path = os.path.join(input_pdf_dir, single_pdf_file)

    print(f"\n===== 开始处理文件：{single_pdf_file} =====")
    # 第一步：提取页面，生成新 PDF（二进制）
    new_pdf_bytes = extract_top3_pdf(pdf_path, biaozhun_embeddings)
    if new_pdf_bytes is None:
        print(f"文件 {single_pdf_file} 未生成新 PDF，跳过 glm-4v 分析。")
    else:
        # 第二步：调用 glm-4v 模型处理内存中的新 PDF
        summary = process_pdf_with_glm4v(new_pdf_bytes, single_pdf_file, fewshot_data, client)
        # 保存汇总结果到 CSV
        df_results = pd.DataFrame([summary])
        output_csv_path = "top_3_results_summary_weiyouhua_single.csv"
        df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"文件【{single_pdf_file}】处理完成，结果已记录至 {output_csv_path}。")

    print(f"\n🎉 文件处理完成。")

