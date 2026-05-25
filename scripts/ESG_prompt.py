import os
import json

# 定义输入和输出文件夹路径
esg_topics_folder = "ESG实质性议题"  # 存放 JSON 文件
esg_reports_folder = "ESG报告总拆分"  # 存放对应的 TXT 文件夹
output_folder = "baichuan_promptv1"  # 结果存放路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历 ESG实质性议题 文件夹下的所有 JSON 文件
for json_filename in os.listdir(esg_topics_folder):
    if json_filename.endswith(".json"):  # 只处理 JSON 文件
        json_path = os.path.join(esg_topics_folder, json_filename)

        # 读取 JSON 文件内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 解析 JSON 文件名（去掉扩展名）
        base_name = os.path.splitext(json_filename)[0]

        # 对应的 ESG 报告文件夹路径
        txt_folder_path = os.path.join(esg_reports_folder, base_name)

        if not os.path.isdir(txt_folder_path):  # 确保对应的 TXT 目录存在
            print(f"警告：未找到匹配的TXT文件夹: {txt_folder_path}")
            continue

        # 创建一个新的字典来保存拼接后的内容
        combined_data = {}

        # 遍历 JSON 数据
        for txt_filename, phrases in data.items():
            txt_path = os.path.join(txt_folder_path, txt_filename)

            # 确保 TXT 文件存在
            if not os.path.isfile(txt_path):
                print(f"警告：文件 {txt_path} 不存在，跳过...")
                continue

            # 读取 TXT 文件内容
            with open(txt_path, "r", encoding="utf-8") as f:
                file_content = f.read().strip()  # 去除文件内容前后空白字符

            # 拼接每个短语和文件内容
            combined_content = [
                f"你现在要做的是文本总结任务，仅限于我提供的文本内容，要求是梳理文本内容，提取出本章中与{phrase} 相关的内容，说明围绕{phrase} 都做了哪些工作。总结要全面，去除无关和一些常用字等，最后以段落的形式给出，不要分条列举,文本内容如下：{file_content}"
                for phrase in phrases
            ]

            # 保存拼接后的内容到字典
            combined_data[txt_filename] = combined_content

        # 生成输出 JSON 文件路径
        output_json_path = os.path.join(output_folder, json_filename)

        # 保存拼接后的内容到 JSON 文件
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

        print(f"处理完成：{output_json_path}")

print("所有处理完成！")

