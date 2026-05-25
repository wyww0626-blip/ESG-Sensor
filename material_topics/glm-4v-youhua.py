from concurrent.futures import ThreadPoolExecutor
import base64
import json
import fitz
import os
import pandas as pd
import time
from zhipuai import ZhipuAI

def pdf_page_to_image_bytes(pdf_document, page_number, zoom=2.0):
    page = pdf_document.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes()

def image_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

with open('img/dataset.json', 'r', encoding='utf-8') as f:
    fewshot_data = json.load(f)

fewshot_preloaded = []
for item in fewshot_data:
    with open(item['img'], 'rb') as img_f:
        img_base64 = base64.b64encode(img_f.read()).decode('utf-8')
    fewshot_preloaded.extend([
        {"type": "image_url", "image_url": {"url": img_base64}},
        {"type": "text", "text": f"{item['prompt']}\n{item['label']}"}
    ])

client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")
pdf_dir = "materials_topics_results_40_topk_3"
output_csv_path = "top_3_results_summary.csv"
results_summary = []

def analyze_single_page(pdf_document, page_number):
    image_bytes = pdf_page_to_image_bytes(pdf_document, page_number)
    img_base64 = image_bytes_to_base64(image_bytes)
    content_list = fewshot_preloaded + [
        {"type": "image_url", "image_url": {"url": img_base64}},
        {"type": "text", "text": "请你帮我判断这张图中是否有议题矩阵，如果有则分析这张ESG报告图中展示了什么内容？如果没有则只需要输出“无”。"}
    ]
    try:
        response = client.chat.completions.create(
            model="glm-4v-plus-0111",
            messages=[{"role": "user", "content": content_list}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API调用错误：{e}"

# 逐文件处理
for pdf_file in os.listdir(pdf_dir):
    if not pdf_file.lower().endswith('.pdf'):
        continue

    pdf_path = os.path.join(pdf_dir, pdf_file)
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    start_time = time.time()

    print(f"\n🚀 开始处理文件：{pdf_file}，共 {total_pages} 页")

    found_result = "无"
    processed_page_count = 0
    futures_dict = {}

    max_workers = 3  # 并发线程数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 按页码顺序提交任务
        for page_number in range(total_pages):
            future = executor.submit(analyze_single_page, pdf_document, page_number)
            futures_dict[page_number] = future

        # 按页码顺序获取结果
        for page_number in range(total_pages):
            result = futures_dict[page_number].result()
            processed_page_count += 1
            print(f"  📄 文件【{pdf_file}】第 {page_number + 1}/{total_pages} 页结果：{result}")

            if result != "无":
                found_result = result
                print("  🟢 检测到有效议题矩阵，终止后续页面分析。\n")
                break

    elapsed_time = time.time() - start_time

    results_summary.append({
        '文件名': pdf_file,
        '分析结果': found_result,
        '处理页数': processed_page_count,
        '处理耗时(秒)': round(elapsed_time, 2)
    })

    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ 文件【{pdf_file}】处理完成，用时 {elapsed_time:.2f} 秒，处理 {processed_page_count} 页。\n")

print(f"\n🎉 所有文件处理完成，结果汇总至：{output_csv_path}")

