import cv2
from PIL import Image

def convert_to_gray(input_image_path, output_image_path):
    """
    将彩色图片转换为高清灰度（黑白）图片，并保存
    :param input_image_path: 输入彩色图片路径
    :param output_image_path: 输出灰度图片路径
    """
    # 读取彩色图像
    image = cv2.imread(input_image_path)

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 PIL 以 300 DPI 高清格式保存
    pil_image = Image.fromarray(gray_image)
    pil_image.save(output_image_path, dpi=(300, 300))  # 设置 300 DPI 高清保存

    print(f"灰度图片已保存至 {output_image_path}")

# 示例调用
input_path = "wjz.png"  # 请替换为你的彩色图片路径
output_path = "wjz1.png"  # 输出灰度图路径
convert_to_gray(input_path, output_path)

