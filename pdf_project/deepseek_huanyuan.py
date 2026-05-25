# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

# 读取 txt 文件内容
with open("output.txt", "r", encoding="utf-8") as f:
    txt_content = f.read()

# 定义提示词
prompt = "请严格按照下述要求进行操作：1.仅使用我提供的文本内容 2.对文本进行完整的还原，请注意不是摘要，是对文本内容的还原，如标题、正文、表格等；3.不添加任何修饰、补充或扩展 4.严格尊重原文内容"

# 拼接提示词与 txt 文件中的内容
user_message = prompt + "\n" + txt_content

client = OpenAI(api_key="<YOUR_API_KEY>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": user_message}
    ],
    max_tokens=8192,
    stream=False
)

# 获取 API 返回内容
output = response.choices[0].message.content

# 使用输入 txt 文件同名，但扩展名为 .md 作为输出文件名
base_name = "input1"  # 如果 input.txt 文件在其他位置或有不同名称，可通过 os.path.splitext(os.path.basename(...)) 获取
output_filename = f"{base_name}.md"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(output)

print(f"处理完成，输出已写入到 {output_filename}")

