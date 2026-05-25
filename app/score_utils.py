import os
import json
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY", "")
MAX_WORKERS = 5
client = ZhipuAI(api_key=API_KEY) if API_KEY else None

def load_score_points(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        score_criteria = json.load(f)
    return [{"category": cat, "score_point": point} for cat, points in score_criteria.items() for point in points]

def evaluate_score(chapter_name, chapter_text, category, score_point):
    if client is None:
        raise RuntimeError("Missing ZHIPU_API_KEY. Copy .env.example to .env and set your own key.")
    try:
        system_prompt = "你是ESG报告评估专家，请你仔细阅读以下报告内容，并判断它是否披露了评分点所述内容，并给出披露分析与打分。"
        user_prompt = f"""请你判断该报告是否披露了以下内容：

[评分点]：
{score_point}

[报告内容]：
{chapter_text}

请按如下 JSON 格式输出：
{{
  "score": 0-1之间的分数,
  "disclosed": "是"或"否",
  "explanation": "一句简要理由"
}}

请你务必仅输出符合上述格式的 JSON，不要添加多余解释。
"""
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content.strip().strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        result_json = json.loads(content)
        return float(result_json["score"])
    except:
        return 0.0

def run_score_for_dimension(score_json_path, name, folder_path, progress_bar=None):
    score_points = load_score_points(score_json_path)
    total_score = 0.0
    valid_count = 0
    chapters = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    total_tasks = len(chapters) * len(score_points)
    completed_tasks = 0

    for chapter_file in chapters:
        file_path = os.path.join(folder_path, chapter_file)
        with open(file_path, "r", encoding="utf-8") as f:
            chapter_text = f.read()[:10000]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(evaluate_score, chapter_file, chapter_text, item["category"], item["score_point"])
                for item in score_points
            ]
            for future in as_completed(futures):
                score = future.result()
                completed_tasks += 1
                if score > 0:
                    total_score += score
                    valid_count += 1
                if progress_bar:
                    progress_bar.progress(min(completed_tasks / total_tasks, 1.0))

    avg_score = total_score / valid_count if valid_count > 0 else 0.0
    return total_score, avg_score

