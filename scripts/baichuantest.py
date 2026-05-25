import requests
import json

url = "https://api.baichuan-ai.com/v1/chat/completions"
api_key = "<YOUR_API_KEY>"

data = {
    "model": "Baichuan2-Turbo",
    "messages": [
        {"role": "user", "content": "测试 API Key 是否有效"}
    ],
    "stream": False
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + api_key
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("API Key 有效")
    print(response.json())
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(f"响应: {response.text}")

