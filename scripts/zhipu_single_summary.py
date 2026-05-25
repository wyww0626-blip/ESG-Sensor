import os
import json
from zhipuai import ZhipuAI
# from dotenv import load_dotenv  # 如果需要加载环境变量可以取消注释

def zhipu_api_abstract(json_file_path):
    client = ZhipuAI(api_key="<YOUR_ZHIPU_API_KEY>")
    base_folder = 'zhipu_result'
    os.makedirs(base_folder, exist_ok=True)

    # 读取单个 JSON 文件内容
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 以 JSON 文件名（去掉扩展名）作为输出文件夹名称
    json_filename = os.path.basename(json_file_path)
    output_folder = os.path.join(base_folder, os.path.splitext(json_filename)[0])
    os.makedirs(output_folder, exist_ok=True)

    for filename, content_list in data.items():
        output_file_path = os.path.join(output_folder, filename.replace('.txt', '_summary.txt'))

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for index, content in enumerate(content_list):
                try:
                    if not content.strip():
                        continue

                    response = client.chat.completions.create(
                        model="glm-4-plus",
                        messages=[
                            {"role": "system", "content": "你是一个文本总结助手"},
                            {"role": "user", "content": content},
                        ],
                    )
                    summary = response.choices[0].message.content
                    output_file.write(f"内容 {index + 1}:")
                    output_file.write(f"总结: {summary}")
                    output_file.write("\n---\n\n")
                except Exception as e:
                    error_log_path = os.path.join(base_folder, 'error_log.txt')
                    with open(error_log_path, 'a', encoding='utf-8') as error_log:
                        error_log.write(f"文件: {filename}, 内容索引: {index + 1}\n")
                        error_log.write(f"错误: {str(e)}\n")
                        error_log.write(f"原始内容: {content[:100]}...\n")
                        error_log.write("\n---\n\n")
                    print(f"处理 {filename} 中的第 {index + 1} 条内容时发生错误，已记录日志。")
        print(f"文件 {output_file_path} 已完成输出。")

# 指定单个 JSON 文件的路径
json_file_path = 'zhipu_prompt/文山电力_output.json'
zhipu_api_abstract(json_file_path)

