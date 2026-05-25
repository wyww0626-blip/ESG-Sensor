import base64
import json
import fitz  # PyMuPDF
import os
import pandas as pd
import time
from openai import OpenAI

# API密钥（请确保填写正确）
API_KEY = "<YOUR_STEPFUN_API_KEY>"
client = OpenAI(api_key=API_KEY, base_url="https://api.stepfun.com/v1")

# PDF单页转图片（不保存，直接返回二进制）
def pdf_page_to_image_bytes(pdf_path, page_number, zoom=2.0):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes()

# 图片二进制转base64（Data URL格式）
def image_bytes_to_base64(image_bytes):
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

# 加载few-shot示例（保持原方式）
with open('img/dataset_1.json', 'r', encoding='utf-8') as f:
    fewshot_data = json.load(f)

pdf_dir = "materials_topics_results_40_topk_3"
output_csv_path = "top_3_results_summary.csv"
results_summary = []

system_prompt = (
    "你是由阶跃星辰提供的AI助手，能够精准分析ESG议题矩阵图片内容。"
    "如果图片中有议题矩阵，请详细分析展示内容；否则，只需输出“无”。"
)

# 遍历所有PDF文件
for pdf_file in os.listdir(pdf_dir):
    if not pdf_file.lower().endswith('.pdf'):
        continue

    pdf_path = os.path.join(pdf_dir, pdf_file)
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)

    print(f"\n开始处理文件：{pdf_file}，共 {total_pages} 页")

    start_time = time.time()
    found_result = "无"
    processed_page_count = 0

    # 逐页处理PDF
    for page_number in range(total_pages):
        processed_page_count += 1

        # PDF页面转图片二进制
        image_bytes = pdf_page_to_image_bytes(pdf_path, page_number)

        # 构造消息体
        content_list = []

        # 添加few-shot示例
        for item in fewshot_data:
            with open(item['img'], 'rb') as img_f:
                few_shot_img_base64 = image_bytes_to_base64(img_f.read())
            content_list.append({
                "type": "image_url",
                "image_url": {"url": few_shot_img_base64}
            })
            content_list.append({
                "type": "text",
                "text": f"{item['prompt']}\n{item['label']}"
            })

        # 添加当前PDF页面图片及提示
        content_list.append({
            "type": "image_url",
            "image_url": {"url": image_bytes_to_base64(image_bytes)}
        })
        content_list.append({
            "type": "text",
            "text": "请你帮我判断这张图中是否有议题矩阵，如果有则分析这张ESG报告图中展示了什么内容？如果没有则只需要输出“无”。"
        })

        # API调用（仅修改此处API调用方式）
        try:
            response = client.chat.completions.create(
                model="step-1v-32k",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list}
                ]
            )
            result = response.choices[0].message.content.strip()
            print(f"  文件【{pdf_file}】第 {page_number + 1}/{total_pages} 页结果：{result}")

            # 如果有有效结果，则终止该文件处理
            if result != "无":
                found_result = result
                print("  🟢 检测到有效议题矩阵，停止后续页面分析。\n")
                break

            # API调用延时，避免频率过高
            time.sleep(1)

        except Exception as e:
            print(f"  🔴 API调用错误，停止该文件处理：{e}")
            found_result = f"API调用错误：{e}"
            break

    elapsed_time = time.time() - start_time

    # 汇总当前PDF的结果
    results_summary.append({
        '文件名': pdf_file,
        '分析结果': found_result,
        '处理页数': processed_page_count,
        '处理耗时(秒)': round(elapsed_time, 2)
    })

    # 每次处理完一个PDF后实时更新CSV
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"文件【{pdf_file}】处理完成，用时 {elapsed_time:.2f} 秒，共处理 {processed_page_count} 页，结果已记录。\n")

print(f"\n🎉 所有文件处理完成，结果汇总至：{output_csv_path}")

