import fitz
import requests
import base64

API_KEY = "<YOUR_BAIDU_OCR_API_KEY>"
SECRET_KEY = "<YOUR_BAIDU_OCR_SECRET_KEY>"


def get_access_token():
    """
    使用 AK、SK 获取 Access Token
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    response = requests.post(url, params=params)
    return str(response.json().get("access_token"))


def extract_titles_from_pdf(pdf_path):
    """
    对 PDF 文件的每一页调用百度办公室文档 OCR 接口，
    提取返回结果中布局类型为 "figure_title" 的区域（候选标题），
    并打印出标题文本及其对应的页码。
    """
    # 读取 PDF 文件，并进行 base64 编码
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    encoded_pdf = base64.b64encode(pdf_data).decode("utf-8")

    # 使用 PyMuPDF 获取 PDF 文件的总页数
    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    access_token = get_access_token()
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_analysis_office?access_token=" + access_token

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    # 遍历 PDF 的每一页
    for page_num in range(1, num_pages + 1):
        payload = {
            "pdf_file": encoded_pdf,
            "pdf_file_num": str(page_num),  # 指定要识别的页码
            "layout_analysis": "true",  # 开启版面分析，返回包括标题在内的区域信息
            "detect_direction": "false",
            "line_probability": "false",
            "disp_line_poly": "false",
            "recg_tables": "false",
            "recog_seal": "false",
            "recg_formula": "false",
            "erase_seal": "false",
            "disp_underline_analysis": "false",
            "char_probability": "false"
        }

        response = requests.post(url, headers=headers, data=payload)
        result = response.json()

        # 判断返回结果是否包含必要的字段
        if "layouts" not in result or "results" not in result:
            print(f"Page {page_num}: 未找到布局信息")
            continue

        layouts = result.get("layouts", [])
        results_list = result.get("results", [])

        # 遍历所有布局区域，筛选出类型为 "figure_title" 的区域
        for layout in layouts:
            if layout.get("layout") == "figure_title":
                indices = layout.get("layout_idx", [])
                title_parts = []
                for idx in indices:
                    # 确保索引在返回结果范围内
                    if idx < len(results_list):
                        word_obj = results_list[idx]
                        # OCR 返回的文字内容通常在 word_obj["words"]["word"] 中
                        word_text = word_obj.get("words", {}).get("word", "")
                        if word_text:
                            title_parts.append(word_text)
                if title_parts:
                    title = " ".join(title_parts)
                    print(f"Page {page_num}: {title}")


if __name__ == '__main__':
    pdf_path = "含有实质性议题/1.电靓陇原2021—国网甘肃省电力公司服务幸福美好新甘肃履责行动书.pdf"  # 请替换为实际 PDF 文件路径
    extract_titles_from_pdf(pdf_path)
