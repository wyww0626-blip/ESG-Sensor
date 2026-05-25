# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import os

# 读取 txt 文件内容
with open("chapter_3.txt", "r", encoding="utf-8") as f:
    txt_content = f.read()

# 定义提示词
prompt = "请严格按照下述要求进行操作：1.仅使用我提供的文本内容，不进行任何猜测，可以进行一定的语句完善，但是要尊重原文本 2.基于我提供的文本思考原文的结构应该是什么样的，并最大限度地还原文本的原始结构，包括标题、正文、段落、表格、列表等各个部分；3.输出的文本应尽可能清晰、条理分明，严格尊重原文内容。"

# 拼接提示词与 txt 文件中的内容
user_message = prompt + "\n" + txt_content

# 调用 Kimi 的 API
client = OpenAI(api_key="<YOUR_API_KEY>", base_url="https://api.moonshot.cn/v1")

response = client.chat.completions.create(
    model="moonshot-v1-32k",
    messages=[
        {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
        {"role": "user", "content": user_message}
    ],
    max_tokens=16384,
    temperature=0.3,
)

# 获取 API 返回内容
output = response.choices[0].message.content

# 使用输入 txt 文件同名，但扩展名为 .md 作为输出文件名
base_name = os.path.splitext(os.path.basename("chapter_33.txt"))[0]
output_filename = f"{base_name}.md"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(output)

print(f"处理完成，输出已写入到 {output_filename}")

