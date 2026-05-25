import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('top_3_results_summary.csv')  # 请将 '你的文件名.csv' 替换为你的实际文件路径

# 计算“处理页数”的平均值和标准差
page_mean = df['处理页数'].mean()
page_std = df['处理页数'].std()

# 计算“处理耗时(秒)”的平均值和标准差
time_mean = df['处理耗时(秒)'].mean()
time_std = df['处理耗时(秒)'].std()

# 输出结果
print(f"处理页数 - 平均值: {page_mean:.2f}, 标准差: {page_std:.2f}")
print(f"处理耗时(秒) - 平均值: {time_mean:.2f}, 标准差: {time_std:.2f}")

