import os
import json
import requests
# from dotenv import load_dotenv
# # 加载环境变量
# load_dotenv()
# apikey1 = os.getenv("BAICHUAN_API_KEY")
def baichuan_api_abstract(folder_path):
    url = "<YOUR_API_KEY>"
    api_key ="<YOUR_API_KEY>"
    base_folder = 'baichuan_result'
    os.makedirs(base_folder, exist_ok=True)

    for json_filename in os.listdir(folder_path):
        if json_filename.endswith('.json'):
            json_file_path = os.path.join(folder_path, json_filename)
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            output_folder = os.path.join(base_folder, os.path.splitext(json_filename)[0])
            os.makedirs(output_folder, exist_ok=True)

            for filename, content_list in data.items():
                output_file_path = os.path.join(output_folder, filename.replace('.txt', '_summary.txt'))

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    for index, content in enumerate(content_list):
                        try:
                            if not content.strip():
                                continue

                            payload = {
                                "model": "Baichuan4",
                                "messages": [
                                    {"role": "system", "content": "你是一个文本总结助手"},
                                    {"role": "user", "content": content}
                                ],
                                "stream": False
                            }

                            headers = {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {api_key}"
                            }

                            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)

                            if response.status_code == 200:
                                summary = response.json()["choices"][0]["message"]["content"]
                                output_file.write(f"内容 {index + 1}:")
                                output_file.write(f"总结: {summary}")
                                output_file.write("\n---\n\n")
                            else:
                                raise Exception(f"API 请求失败，状态码: {response.status_code}, 响应: {response.text}")

                        except Exception as e:
                            error_log_path = os.path.join(base_folder, 'error_log.txt')
                            with open(error_log_path, 'a', encoding='utf-8') as error_log:
                                error_log.write(f"文件: {filename}, 内容索引: {index + 1}\n")
                                error_log.write(f"错误: {str(e)}\n")
                                error_log.write(f"原始内容: {content[:100]}...\n")
                                error_log.write("\n---\n\n")
                            print(f"处理 {filename} 中的第 {index + 1} 条内容时发生错误，已记录日志。")
                print(f"文件 {output_file_path} 已完成输出。")

json_folder_path = 'baichuan_promptv1'
baichuan_api_abstract(json_folder_path)
