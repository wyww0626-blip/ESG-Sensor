import fitz  # PyMuPDF
import concurrent.futures  # 并行处理
import os  # 用于文件操作


def extract_font_sizes_from_pdf(pdf_path):
    """提取PDF每一页的字体大小及对应的字符数"""
    doc = fitz.open(pdf_path)
    page_font_sizes = {}

    def extract_sizes_for_page(page_num):
        page = doc.load_page(page_num)
        font_sizes = {}

        text_instances = page.get_text("dict")["blocks"]
        for block in text_instances:
            if block.get('type', 1) == 0:  # 仅处理文本块
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_size = span.get('size', 0)
                        text = span.get('text', '')
                        if font_size not in font_sizes:
                            font_sizes[font_size] = 0
                        font_sizes[font_size] += len(text)

        return page_num, font_sizes

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_sizes_for_page, page_num) for page_num in range(len(doc))]

        for future in concurrent.futures.as_completed(futures):
            page_num, font_sizes = future.result()
            if font_sizes:
                page_font_sizes[page_num] = font_sizes

    doc.close()
    return page_font_sizes


def classify_font_sizes(page_font_sizes):
    """分类PDF每一页的字体信息，判断正文和标题"""
    classified_font_sizes = {}
    title_only_pages = set()  # 只包含标题的页码
    pages_with_body_text = set()  # 存储包含正文的页码

    for page_num, font_sizes in page_font_sizes.items():
        sorted_font_sizes = sorted(font_sizes.items(), key=lambda x: x[1], reverse=True)
        if not sorted_font_sizes:
            continue

        max_size, max_count = sorted_font_sizes[0]
        second_size, second_count = sorted_font_sizes[1] if len(sorted_font_sizes) > 1 else (None, 0)

        # 判断是否包含正文
        contains_body = max_count > 110  # 设定正文的基本字数阈值
        if contains_body:
            pages_with_body_text.add(page_num)
        else:
            title_only_pages.add(page_num)  # 只包含标题的页面

        # 分类
        if second_count and second_count >= max_count * 0.85:
            classified_font_sizes[page_num] = [
                (size, count, '正文') if count == max_count or count == second_count else (size, count, '标题')
                for size, count in sorted_font_sizes
            ]
        else:
            classified_font_sizes[page_num] = [
                (size, count, '正文' if count == max_count else '标题')
                for size, count in sorted_font_sizes
            ]

    return classified_font_sizes, title_only_pages


def split_pdf_into_chapters(pdf_path, title_only_pages):
    """根据标题页将PDF拆分成章节"""
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(os.path.dirname(pdf_path), base_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    title_only_pages = sorted(title_only_pages)  # 标题页排序
    chapters = []
    start_page = 0  # 章节起始页

    for title_page in title_only_pages:
        if start_page < title_page:  # 避免重复分割
            chapters.append((start_page, title_page - 1))  # 记录当前章节（标题页到正文部分）
        start_page = title_page  # 新的章节起点

    if start_page < len(doc):  # 记录最后一章
        chapters.append((start_page, len(doc) - 1))

    # 生成章节 PDF
    for i, (start, end) in enumerate(chapters):
        output_path = os.path.join(output_folder, f"chapter_{i+1}.pdf")
        new_doc = fitz.open()
        for page in range(start, end + 1):
            new_doc.insert_pdf(doc, from_page=page, to_page=page)
        new_doc.save(output_path)
        new_doc.close()
        print(f"章节 {i+1}: {output_path} 已保存")

    doc.close()


# 处理 ESG_report 目录下的所有 PDF 文件
pdf_folder = "ESG_report"

for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file_name)
        print(f"正在处理：{pdf_path}")
        page_font_sizes = extract_font_sizes_from_pdf(pdf_path)
        classified_font_sizes, title_only_pages = classify_font_sizes(page_font_sizes)
        split_pdf_into_chapters(pdf_path, title_only_pages)

