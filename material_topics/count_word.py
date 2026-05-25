import fitz  # PyMuPDF
from collections import defaultdict

def extract_titles_by_font_size(pdf_path, char_limit=30):
    pdf_document = fitz.open(pdf_path)
    font_sizes_per_page = []

    # 存放标题的列表
    titles_with_pages = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        page_dict = page.get_text("dict")
        font_groups = defaultdict(list)

        # 提取每个页面所有字号的文本
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

        # 统计每个字号的总字数
        font_size_char_count = {fs: sum(len(t) for t in texts) for fs, texts in font_groups.items()}
        font_sizes_per_page.append((page_num + 1, font_size_char_count))

        # 选择字数小于char_limit的字号对应的文本作为标题
        for fs, total_chars in font_size_char_count.items():
            if total_chars < char_limit:
                # 该字号下可能有多个文本，都视作标题
                for title_text in font_groups[fs]:
                    titles_with_pages.append((page_num + 1, title_text.strip(), fs, total_chars))

    return titles_with_pages

# 使用示例
if __name__ == "__main__":
    pdf_path = "含有实质性议题/1.电靓陇原2021—国网甘肃省电力公司服务幸福美好新甘肃履责行动书.pdf"  # 你的PDF文件路径
    titles = extract_titles_by_font_size(pdf_path, char_limit=30)

    print("PDF文件中各个字号字数小于30的标题及对应页码如下：\n")
    for page_num, title, fs, chars in titles:
        print(f"页码: {page_num} | 字号: {fs:.2f} | 字数: {chars} | 标题: {title}")

