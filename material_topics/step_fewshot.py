import os
from openai import OpenAI
import json

# API_KEY = os.getenv("2LDAaqmJy6a7r4GZHaDzwHet6KOzwaZKzdcaKUxpbOmSnwVHzKHdQu3irufeFgQ2F")
client = OpenAI(api_key="<YOUR_STEPFUN_API_KEY>", base_url="https://api.stepfun.com/v1")

# 加载few-shot数据
with open('img/dataset.json', 'r', encoding='utf-8') as f:
    fewshot_data = json.load(f)

# 上传图片获取File IDs
for item in fewshot_data:
    img_path = item['img']
    with open(img_path, 'rb') as img_file:
        uploaded_file = client.files.create(
            file=img_file,
            purpose='storage'
        )
    file_id = uploaded_file.id
    # 更新JSON内容，保存stepfile链接
    item['file_id'] = f"stepfile://{file_id}"
    print(f"已上传 {img_path}，File ID: {file_id}")

# 更新dataset.json保存file_id
with open('img/dataset_uploaded.json', 'w', encoding='utf-8') as f:
    json.dump(fewshot_data, f, ensure_ascii=False, indent=2)

print("上传完成，已生成 dataset_uploaded.json 文件。")

