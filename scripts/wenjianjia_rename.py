import os
import re

# 文件夹路径
base_path = "ESG报告总拆分"

# 遍历文件夹
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path):
        # 使用正则表达式去除文件夹名前的数字和符号
        new_name = re.sub(r'^\d+[_\.-]*', '', folder_name)
        new_folder_path = os.path.join(base_path, new_name)

        # 重命名文件夹
        if folder_name != new_name:
            try:
                os.rename(folder_path, new_folder_path)
                print(f"重命名: {folder_name} -> {new_name}")
            except Exception as e:
                print(f"重命名失败: {folder_name}, 错误: {e}")

print("重命名完成！")
