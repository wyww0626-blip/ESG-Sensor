import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import pandas as pd

# # 加载 .env 文件
# load_dotenv()
#
# # 获取 API Key
# api_key = os.getenv("MOONSHOT_API_KEY1")

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="<YOUR_API_KEY>",
    base_url="https://api.moonshot.cn/v1"
)

# 读取文本文件
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}：{e}")
        return None

# 构造评估提示词（Prompt）
def get_evaluation_prompt(reference_summary, generated_summary):
    return f"""
你是一位专业的文本分析和摘要评估专家。现在需要对生成摘要与参考摘要之间的相似性进行全面评估。请从以下维度进行打分（评分范围为 1 到 10 分，其中 10 分表示完全一致）：

1️⃣ **语义一致性（Semantic Consistency）**：生成摘要是否准确反映参考摘要中的核心观点和关键信息？
2️⃣ **信息覆盖度（Information Coverage）**：生成摘要是否完整涵盖参考摘要中的重要内容？是否存在信息遗漏或偏差？
3️⃣ **表达相似性（Linguistic Similarity）**：生成摘要与参考摘要在语言风格、用词和句式结构上是否接近？
4️⃣ **简洁性与冗余度（Conciseness and Redundancy）**：生成摘要是否与参考摘要一样简明扼要，避免冗余信息？
5️⃣ **逻辑结构（Logical Structure）**：生成摘要是否与参考摘要保持相同的内容组织方式和叙述顺序？

请针对每个维度分别给出 1 到 10 的评分，并提供一个整体相似度评分。同时，请简要说明你的评分依据。

📄 **参考摘要**：
{reference_summary}

📝 **生成摘要**：
{generated_summary}

🎯 **请按以下格式输出评估结果**：
- 语义一致性：X 分
- 信息覆盖度：X 分
- 表达相似性：X 分
- 简洁性与冗余度：X 分
- 逻辑结构：X 分
- 总体相似度评分：X 分

💡 **评分依据**：
（请简要说明每个评分的理由）
"""

# 调用 API，对单个摘要对进行评估
def evaluate_summaries(reference_summary, generated_summary):
    retries = 5  # 最大重试次数
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=[{
                    "role": "system",
                    "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手。你会为用户提供安全、准确、有帮助的回答。"
                }, {
                    "role": "user",
                    "content": get_evaluation_prompt(reference_summary, generated_summary)
                }],
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ API 调用失败（尝试 {attempt + 1}/{retries}）：{e}")
            wait_time = (attempt + 1) * 5
            print(f"⏳ 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    return "评估失败"

# 批量评估（按文件夹顺序一一对应）
def batch_evaluate(reference_root, generated_root, output_folder):
    reference_dirs = sorted(os.listdir(reference_root))
    generated_dirs = sorted(os.listdir(generated_root))

    # 确认文件夹数量一致
    if len(reference_dirs) != len(generated_dirs):
        print(f"⚠️ 文件夹数量不一致：参考摘要文件夹 {len(reference_dirs)} 个，生成摘要文件夹 {len(generated_dirs)} 个")
        return

    # 依次遍历文件夹
    for ref_dir, gen_dir in zip(reference_dirs, generated_dirs):
        ref_path = os.path.join(reference_root, ref_dir)
        gen_path = os.path.join(generated_root, gen_dir)

        if not os.path.isdir(ref_path) or not os.path.isdir(gen_path):
            print(f"⚠️ 跳过非文件夹项：{ref_dir} 或 {gen_dir}")
            continue

        results = []
        reference_files = sorted([f for f in os.listdir(ref_path) if f.endswith(".txt")])
        generated_files = sorted([f for f in os.listdir(gen_path) if f.endswith(".txt")])

        if len(reference_files) != len(generated_files):
            print(f"⚠️ 子文件夹中文件数量不一致：{ref_dir} 和 {gen_dir}")
            continue

        for ref_file, gen_file in zip(reference_files, generated_files):
            reference_summary = read_file(os.path.join(ref_path, ref_file))
            generated_summary = read_file(os.path.join(gen_path, gen_file))

            if reference_summary and generated_summary:
                print(f"✅ 正在评估文件对：{ref_file} 和 {gen_file}")
                evaluation_result = evaluate_summaries(reference_summary, generated_summary)

                results.append({
                    "参考摘要文件名": ref_file,
                    "生成摘要文件名": gen_file,
                    "评估结果": evaluation_result
                })

                time.sleep(5)  # 防止 API 请求过于密集

        if results:
            output_file = os.path.join(output_folder, f"{ref_dir}_evaluation_results.csv")
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ 评估结果已保存至：{output_file}")

if __name__ == "__main__":
    reference_root = "ESG基准v1-deepseek"
    generated_root = "deepseek_result"
    output_folder = "deepseek_evaluation_results"

    os.makedirs(output_folder, exist_ok=True)
    batch_evaluate(reference_root, generated_root, output_folder)

