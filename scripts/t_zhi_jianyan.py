import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# 读取两个 CSV 文件（请替换为你的实际路径）
df1 = pd.read_csv('zhipu_kimi_evaluation/zhipu_kimi_evaluation_results.csv')
df2 = pd.read_csv('deepseek_evaluation/zhipu_evaluation_results_dp.csv')

# 公共评分列
score_cols = ['语义一致性', '信息覆盖度', '表达相似性', '简洁性与冗余度', '逻辑结构', '总体相似度评分']

# 来源字段自适应
source_col = '来源CSV文件' if '来源CSV文件' in df1.columns else '来源文件'

# 结果列表
results = []

# 全部来源总体统计
all_t_list = []
all_p_list = []
all_similar_count = 0
all_different_count = 0

# 获取所有来源文件
all_sources = df1[source_col].unique()

for source in all_sources:
    print(f"\n📄 {source} 的 t 值和 p 值如下：")

    indices = df1[df1[source_col] == source].index
    t_list, p_list = [], []
    similar_count, different_count = 0, 0

    for i in indices:
        scores1 = df1.loc[i, score_cols].astype(float).values
        scores2 = df2.loc[i, score_cols].astype(float).values
        chapter = df1.loc[i, '参考摘要文件名']

        if np.array_equal(scores1, scores2):
            t_stat, p_val = 0.0, 1.0
            judgment = '完全一致，相似'
            similar_count += 1
            all_similar_count += 1
        else:
            t_stat, p_val = ttest_rel(scores1, scores2)
            if (
                np.isnan(t_stat) or np.isnan(p_val) or
                np.isinf(t_stat) or np.isinf(p_val)
            ):
                t_stat, p_val = np.nan, np.nan
                judgment = '检验异常'
            else:
                if p_val >= 0.01:
                    judgment = '相似'
                    similar_count += 1
                    all_similar_count += 1
                else:
                    judgment = '有差异'
                    different_count += 1
                    all_different_count += 1

        # 收集有效值
        if not np.isnan(t_stat) and not np.isinf(t_stat) and not np.isnan(p_val) and not np.isinf(p_val):
            t_list.append(t_stat)
            p_list.append(p_val)
            all_t_list.append(t_stat)
            all_p_list.append(p_val)

        print(f"{chapter}：t = {t_stat:.4f}, p = {p_val:.4f} {'✅ ' + judgment if '相似' in judgment else '❌ ' + judgment}")

        results.append({
            '来源文件': source,
            '章节': chapter,
            't 值': round(t_stat, 4) if not np.isnan(t_stat) else 'nan',
            'p 值': round(p_val, 4) if not np.isnan(p_val) else 'nan',
            '显著性判断': judgment
        })

    # 来源级统计
    if t_list and p_list:
        t_avg = round(np.mean(t_list), 4)
        p_avg = round(np.mean(p_list), 4)
        t_std = round(np.std(t_list, ddof=1), 4)
        p_std = round(np.std(p_list, ddof=1), 4)
        similarity_rate = round(similar_count / (similar_count + different_count), 4) if (similar_count + different_count) > 0 else 'N/A'

        print(f"\n📈 来源文件【{source}】\n平均 t = {t_avg}, 标准差 = {t_std}\n平均 p = {p_avg}, 标准差 = {p_std}\n✅ 相似率 = {similarity_rate}\n")

        results.append({
            '来源文件': source,
            '章节': '来源平均值',
            't 值': t_avg,
            'p 值': p_avg,
            't 值标准差': t_std,
            'p 值标准差': p_std,
            '相似率': similarity_rate,
            '显著性判断': '——'
        })

# 总体统计
if all_t_list and all_p_list:
    total_t_avg = round(np.mean(all_t_list), 4)
    total_p_avg = round(np.mean(all_p_list), 4)
    total_t_std = round(np.std(all_t_list, ddof=1), 4)
    total_p_std = round(np.std(all_p_list, ddof=1), 4)
    total_similarity_rate = round(all_similar_count / (all_similar_count + all_different_count), 4) if (all_similar_count + all_different_count) > 0 else 'N/A'

    print(f"\n📊 所有来源章节总体平均 t = {total_t_avg}, 标准差 = {total_t_std}")
    print(f"📊 所有来源章节总体平均 p = {total_p_avg}, 标准差 = {total_p_std}")
    print(f"✅ 所有来源章节相似率 = {total_similarity_rate}\n")

    results.append({
        '来源文件': '全部来源',
        '章节': '总体平均值',
        't 值': total_t_avg,
        'p 值': total_p_avg,
        't 值标准差': total_t_std,
        'p 值标准差': total_p_std,
        '相似率': total_similarity_rate,
        '显著性判断': '——'
    })

# 转换为 DataFrame，并保存
results_df = pd.DataFrame(results)

# 确保列存在
for col in ['t 值标准差', 'p 值标准差', '相似率']:
    if col not in results_df.columns:
        results_df[col] = ''

results_df.to_csv('t_zhi_jianyan/zhipu_评分_t检验结果输出.csv', index=False, encoding='utf-8-sig')
print("\n✅ 所有结果已保存到文件：t_zhi_jianyan/zhipu_评分_t检验结果输出.csv")

