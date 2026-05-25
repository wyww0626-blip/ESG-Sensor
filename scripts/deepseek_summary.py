import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import requests

# # 加载 .env 文件中的环境变量
# load_dotenv()
#
# # 获取环境变量中的 API key
# api_key = os.getenv("deepseek_API")

# 初始化客户端，确保 base_url 包含协议头
client = OpenAI(api_key="<YOUR_API_KEY>", base_url="https://api.deepseek.com")

# 遍历 deepseek_prompt 目录下所有的 JSON 文件
json_folder_path = "deepseek_prompt"
json_files = [f for f in os.listdir(json_folder_path) if f.endswith(".json")]

# 遍历每个 JSON 文件
for json_file_name in json_files:
    try:
        # 获取 JSON 文件路径
        json_file_path = os.path.join(json_folder_path, json_file_name)

        # 读取 JSON 文件
        with open(json_file_path, "r", encoding="utf-8") as f:
            kk_data = json.load(f)

        # 为当前 JSON 文件创建一个文件夹来保存输出文件
        output_folder = os.path.join(json_folder_path, os.path.splitext(json_file_name)[0])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 对每个文本文件名及其内容列表进行批量推理和保存
        for idx, (file_name, text_list) in enumerate(kk_data.items(), start=1):
            try:
                # 调用 OpenAI 模型进行批量推理
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "你是一个帮助助理，提供中文回答"},
                        {"role": "user", "content": "请依次回答以下问题：\n" + "\n".join(text_list)}
                    ],
                    stream=False
                )

                # 获取批量推理结果
                result = response.choices[0].message.content

                # 生成文件名 c1.txt, c2.txt, ...
                output_file_name = f"c{idx}.txt"
                output_file_path = os.path.join(output_folder, output_file_name)

                # 保存批量推理结果到对应的 txt 文件中
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(result)

                print(f"推理结果已保存到 {output_file_path}")

                # 每处理完一个批次，等待 10 秒再发起下一批次处理
                if idx < len(kk_data):
                    time.sleep(10)

            except Exception as e:
                print(f"处理文件 {json_file_name} 中的 {file_name} 时出错: {e}")

        # 每处理完一个 JSON 文件，等待 5 分钟再去请求下一个文件
        print(f"处理完文件 {json_file_name}，等待 5 分钟后继续...")
        time.sleep(200)  # 5分钟 = 300秒

    except json.JSONDecodeError as e:
        print(f"JSON 文件 {json_file_name} 解析失败: {e}")
    except Exception as e:
        print(f"处理文件 {json_file_name} 时发生未知错误: {e}")

