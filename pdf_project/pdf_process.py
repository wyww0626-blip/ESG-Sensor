import fitz  # PyMuPDF
import concurrent.futures  # 并行处理
import os  # 用于文件操作


def extract_font_sizes_from_pdf(pdf_path):
    """提取PDF每一页的字体大小及对应的字符数，并获取对应的文本内容"""
    doc = fitz.open(pdf_path)
    page_font_sizes = {}
    page_titles = {}  # 存储页码对应的最大字体文本（可能是标题）

    def extract_sizes_for_page(page_num):
        page = doc.load_page(page_num)
        font_sizes = {}
        font_texts = {}  # 记录不同字体的文本

        text_instances = page.get_text("dict")["blocks"]
        for block in text_instances:
            if block.get('type', 1) == 0:  # 仅处理文本块
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_size = span.get('size', 0)
                        text = span.get('text', '').strip()
                        if font_size not in font_sizes:
                            font_sizes[font_size] = 0
                            font_texts[font_size] = []
                        font_sizes[font_size] += len(text)
                        font_texts[font_size].append(text)

        # 获取最大字体的文本作为标题
        if font_sizes:
            max_font_size = max(font_sizes, key=lambda size: font_sizes[size])
            page_title = " ".join(font_texts[max_font_size])  # 合并标题文本
            return page_num, font_sizes, page_title
        return page_num, font_sizes, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_sizes_for_page, page_num) for page_num in range(len(doc))]

        for future in concurrent.futures.as_completed(futures):
            page_num, font_sizes, page_title = future.result()
            if font_sizes:
                page_font_sizes[page_num] = font_sizes
                if page_title:
                    page_titles[page_num] = page_title

    doc.close()
    return page_font_sizes, page_titles


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


def split_pdf_into_chapters(pdf_path, title_only_pages, page_titles):
    """根据标题页拆分PDF，并打印每章标题和页码"""
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(os.path.dirname(pdf_path), base_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    title_only_pages = sorted(title_only_pages)  # 按页码排序
    chapters = []
    start_page = 0  # 章节起始页

    for title_page in title_only_pages:
        if start_page < title_page:  # 避免重复分割
            chapters.append((start_page, title_page - 1))  # 记录当前章节
        start_page = title_page  # 新的章节起点

    if start_page < len(doc):  # 记录最后一章
        chapters.append((start_page, len(doc) - 1))

    # 生成章节 PDF，并输出每章标题
    for i, (start, end) in enumerate(chapters):
        output_path = os.path.join(output_folder, f"chapter_{i+1}.pdf")
        new_doc = fitz.open()
        for page in range(start, end + 1):
            new_doc.insert_pdf(doc, from_page=page, to_page=page)
        new_doc.save(output_path)
        new_doc.close()

        # 获取章节标题
        chapter_title = page_titles.get(start, f"章节 {i+1}")  # 若无标题，默认"章节 X"
        print(f"章节 {i+1}: {chapter_title} (页码 {start + 1} - {end + 1})")

    doc.close()


# 运行处理流程
pdf_path = "ESG_report/2021华为ESG报告.pdf"  # 替换为实际 PDF 路径
page_font_sizes, page_titles = extract_font_sizes_from_pdf(pdf_path)
classified_font_sizes, title_only_pages = classify_font_sizes(page_font_sizes)
split_pdf_into_chapters(pdf_path, title_only_pages, page_titles)

