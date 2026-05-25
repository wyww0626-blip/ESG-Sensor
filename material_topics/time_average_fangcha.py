import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv("materials_topics_results_40_topk_7/results.csv")

# 去除文件名重复项，只保留每个文件名对应的一次处理耗时
unique_times = df.drop_duplicates(subset='文件名')[['文件名', '处理耗时(秒)']]

# 提取处理耗时的数组
times = unique_times['处理耗时(秒)'].values

# 去掉处理耗时中的最低值和最高值
if len(times) > 2:
    trimmed_times = np.delete(times, [np.argmin(times), np.argmax(times)])
else:
    print("数据不足以去掉最高值和最低值。")
    trimmed_times = times

# 计算统计值（去掉极值后）
mean_time = np.mean(trimmed_times)
variance_time = np.var(trimmed_times)
std_time = np.std(trimmed_times)

# 输出结果
print("去除极值后不同文件处理耗时的统计信息：")
print(f"文件数：{len(trimmed_times)}")
print(f"平均处理耗时：{mean_time:.2f} 秒")
print(f"处理耗时方差：{variance_time:.4f}")
print(f"处理耗时标准差：{std_time:.4f}")

