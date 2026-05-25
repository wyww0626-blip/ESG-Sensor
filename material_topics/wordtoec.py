import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 设置日志，方便观察训练过程
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 假设语料文件 'your_corpus.txt' 每行是一个句子，单词以空格分隔
corpus_file = "corpus.txt"

# 使用 LineSentence 逐行读取语料
sentences = LineSentence(corpus_file)

# 训练 Word2Vec 模型
model = Word2Vec(
    sentences,         # 输入语料
    vector_size=100,   # 词向量维度
    window=5,          # 上下文窗口大小
    min_count=1,       # 词频低于5的单词会被忽略
    workers=4          # 使用的线程数
)

# 保存模型以便后续使用
model.save("word2vec.model")
print("模型训练完毕并已保存。")

# 示例：获取某个词的向量及查找最相似词
vector = model.wv["实质性议题"]  # 替换为你语料中存在的词
print("向量：", vector)

similar_words = model.wv.most_similar("实质性议题", topn=5)
print("与'示例词'最相似的5个词：")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

