import base64
import fitz  # PyMuPDF
import os
import pandas as pd
import time
from zhipuai import ZhipuAI

# PDF单页转图片（不保存，直接返回二进制）
def pdf_page_to_image_bytes(pdf_path, page_number, zoom=2.0):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes()

# 图片二进制转base64
def image_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# 初始化智谱API
client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")

# PDF文件目录
pdf_dir = "含有实质性议题"

# 输出CSV文件路径
output_csv_path = "top_3_results_summary_weiyouhua.csv"
results_summary = []

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

    for page_number in range(total_pages):
        processed_page_count += 1

        # 转换PDF页为图片并编码为base64
        image_bytes = pdf_page_to_image_bytes(pdf_path, page_number)
        image_base64 = image_bytes_to_base64(image_bytes)

        # 构造对话内容（无few-shot）
        content_list = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64}
            },
            {
                "type": "text",
                "text": "请你帮我判断这张图中是否有议题矩阵，如果有则分析这张ESG报告图中展示了什么内容？如果没有则只需要输出“无“"
            }
        ]

        try:
            response = client.chat.completions.create(
                model="glm-4v-plus-0111",
                messages=[{"role": "user", "content": content_list}]
            )
            result = response.choices[0].message.content.strip()
            print(f"  文件【{pdf_file}】第 {page_number + 1}/{total_pages} 页结果：{result}")

            if result != "无":
                found_result = result
                print(f"  🟢 检测到有效议题矩阵，停止后续页面分析。\n")
                break

            time.sleep(1)  # 防止API请求过于频繁

        except Exception as e:
            print(f"  🔴 API调用错误，停止该文件处理：{e}")
            found_result = f"API调用错误：{e}"
            break

    elapsed_time = time.time() - start_time

    # 汇总结果
    results_summary.append({
        '文件名': pdf_file,
        '分析结果': found_result,
        '处理页数': processed_page_count,
        '处理耗时(秒)': round(elapsed_time, 2)
    })

    # 实时更新CSV
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"文件【{pdf_file}】处理完成，用时 {elapsed_time:.2f} 秒，共处理 {processed_page_count} 页，结果已记录。\n")

print(f"\n🎉 所有文件处理完成，结果汇总至：{output_csv_path}")

