import os
import PyPDF2
import numpy as np

def count_pdf_pages(pdf_path):
    """计算单个PDF的页数"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return len(reader.pages)

def analyze_pdfs_in_folder(folder_path):
    """遍历文件夹，计算所有PDF页数的平均值、方差和标准差"""
    page_counts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            try:
                pages = count_pdf_pages(pdf_path)
                print(f"{filename} 页数：{pages}")
                page_counts.append(pages)
            except Exception as e:
                print(f"读取文件 {filename} 时出现错误：{e}")

    if not page_counts:
        print("未找到有效的PDF文件。")
        return

    mean_pages = np.mean(page_counts)
    variance_pages = np.var(page_counts)
    std_pages = np.std(page_counts)

    print(f"\nPDF平均页数：{mean_pages:.2f}")
    print(f"PDF页数方差：{variance_pages:.2f}")
    print(f"PDF页数标准差：{std_pages:.2f}")

if __name__ == "__main__":
    folder_path = "materials_topics_results_40"
    analyze_pdfs_in_folder(folder_path)

