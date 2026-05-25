import os
import base64
import requests
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter

load_dotenv()

API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
PDF_PATH = "ESG_report/2021华为ESG报告/chapter_3.pdf"  # 你的 PDF 文件路径

def extract_pdf_page(pdf_path, page_number):
    """
    提取 PDF 指定页，并生成新的 PDF 数据（字节形式）
    :param pdf_path: 原始 PDF 路径
    :param page_number: 要提取的页码（从0开始，0 表示第一页）
    :return: 新 PDF 文件的字节数据
    """
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        if page_number >= len(reader.pages):
            raise ValueError("页码超出 PDF 页数范围")
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number])
        output_stream = BytesIO()
        writer.write(output_stream)
        return output_stream.getvalue()

def get_access_token():
    """
    使用 AK 和 SK 获取鉴权签名（Access Token）
    :return: access_token，或是 None（如果出错）
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY
    }
    res = requests.post(url, params=params)
    return str(res.json().get("access_token"))

def main():
    # 获取 PDF 文件的基本名称，用于生成输出 txt 文件名
    base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    output_filename = f"{base_name}.txt"

    # 获取 access_token 后构造基础请求 URL
    access_token = get_access_token()
    base_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_analysis_office?access_token=" + access_token

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    common_payload = {
        "detect_direction": "false",
        "line_probability": "false",
        "disp_line_poly": "false",
        "layout_analysis": "false",
        "recg_tables": "false",
        "recog_seal": "false",
        "recg_formula": "false",
        "erase_seal": "false",
        "disp_underline_analysis": "false",
        "char_probability": "false"
    }

    # 获取 PDF 总页数
    with open(PDF_PATH, "rb") as f:
        reader = PdfReader(f)
        total_pages = len(reader.pages)

    recognized_text = ""

    # 遍历 PDF 中的每一页
    for i in range(total_pages):
        try:
            page_data = extract_pdf_page(PDF_PATH, i)
        except Exception as e:
            recognized_text += f"第 {i+1} 页提取失败: {e}\n"
            continue

        pdf_base64 = base64.b64encode(page_data).decode('utf-8')

        # 上传的是单页 PDF，故 pdf_file_num 固定为 "1"
        payload = common_payload.copy()
        payload.update({
            "pdf_file": pdf_base64,
            "pdf_file_num": "1"
        })

        response = requests.post(base_url, headers=headers, data=payload)
        data = response.json()

        # 按结构顺序提取识别出的文字
        if data.get("error_code", 0) == 0 and "results" in data:
            page_text = ""
            for result in data["results"]:
                word = result.get("words", {}).get("word", "")
                page_text += word + "\n"
            recognized_text += f"第 {i+1} 页识别结果：\n{page_text}\n"
        else:
            error_msg = data.get("error_msg", "未知错误")
            recognized_text += f"第 {i+1} 页识别失败: {error_msg}\n"

    # 将所有识别结果写入到以 PDF 文件命名的 txt 文件中
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(recognized_text)

    print(f"处理完成，识别结果已保存到 {output_filename}")

if __name__ == '__main__':
    main()

