import json
import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY", "")) if os.getenv("ZHIPU_API_KEY") else None

def summarize_json_and_merge(json_file_path):
    if client is None:
        raise RuntimeError("Missing ZHIPU_API_KEY. Copy .env.example to .env and set your own key.")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    merged_summary = ""
    for filename, content_list in data.items():
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
                summary = response.choices[0].message.content.strip()
                merged_summary += f"[{filename} 内容 {index + 1} 摘要]：{summary}\n\n"
            except Exception as e:
                merged_summary += f"[{filename} 内容 {index + 1}] 摘要失败：{str(e)}\n\n"
    return merged_summary



