# ESG Report Intelligence Pipeline

本项目是一个面向企业 ESG/可持续发展报告的自动化分析原型。它把 PDF 报告解析、实质性议题识别、章节级摘要生成、ESG 三维度评分，以及多模型结果评估串成一条实验流程，用来比较不同大模型在 ESG 报告理解任务上的表现。

> 说明：本仓库是作品集公开版，已移除真实 API Key、原始批量 PDF、完整中间产物和个人本地路径，仅保留核心代码、示例输入、部分可视化结果与评估样例。

## 项目做了什么

项目围绕 ESG 报告的“读、抽、评、比”四个环节展开：

1. PDF 解析与章节拆分  
   通过 PyMuPDF/OCR 提取报告文本，并根据字体大小、标题页等特征将长报告拆分为章节。

2. 实质性议题识别  
   使用报告目录页、议题页和文本相似度方法定位 ESG 实质性议题，辅助找到企业重点披露的环境、社会、治理主题。

3. 多模型摘要生成  
   构造面向 ESG 议题的提示词，调用不同模型生成章节摘要和议题相关总结，覆盖 DeepSeek、Kimi、智谱、百川等模型实验。

4. ESG 三维度评分  
   将环境、社会、治理维度拆解为结构化评分点，使用大模型判断报告是否披露相应内容，并输出分数和解释。

5. 结果评估与可视化  
   使用 ROUGE、BERTScore、平均分、热力图、雷达图等方法对不同模型的摘要质量和评分表现进行比较。

## 目录结构

```text
.
├── app/                     # Streamlit 本地演示工具：评分、议题识别、摘要生成
├── scripts/                 # 摘要生成、评分、ROUGE/BERTScore、图表绘制等脚本
├── material_topics/         # 实质性议题定位与多方法融合实验代码
├── pdf_project/             # PDF/OCR/章节拆分相关代码与示例 markdown
├── examples/                # 少量脱敏示例输入
├── results/                 # 部分评估 CSV 和可视化图片
├── .env.example             # 环境变量模板
├── requirements.txt         # 主要依赖
└── README.md
```

## 技术栈

- Python
- Streamlit
- PyMuPDF
- pandas / numpy
- rouge-score / bert-score
- matplotlib / seaborn
- zhipuai / OpenAI-compatible SDK
- OCR 与多模型 API 调用

## 快速开始

创建环境并安装依赖：

```bash
pip install -r requirements.txt
```

复制环境变量模板，并填入你自己的 API Key：

```bash
cp .env.example .env
```

运行本地演示工具：

```bash
streamlit run app/main.py
```

如果只想看代码结构和实验流程，不配置 API Key 也可以浏览 `scripts/`、`material_topics/`、`pdf_project/` 和 `results/`。

## 核心模块说明

- `app/main.py`：Streamlit 入口，提供 ESG 评分、实质性议题识别、摘要生成三个交互入口。
- `app/score_utils.py`：加载 ESG 评分点，并调用模型对章节内容进行披露判断与打分。
- `app/topic_utils.py`：结合 PDF 页面提取、向量相似度和少样本样例，定位实质性议题相关页面。
- `app/summary_utils.py`：读取结构化 JSON 内容，生成并合并章节摘要。
- `scripts/calculate_rouge.py`：对模型摘要和参考摘要计算 ROUGE 指标。
- `scripts/calculate_bertscore.py`：计算 BERTScore，用于衡量语义相似度。
- `scripts/heatmap.py`、`scripts/leidatu.py`：生成模型评估热力图和雷达图。
- `material_topics/`：实质性议题抽取、OCR、标题页定位、多方法融合与实验统计。
- `pdf_project/`：PDF 章节拆分、OCR 识别、摘要还原相关实验。

## 公开版删减内容

为了适合放到 GitHub，本版本做了以下处理：

- 删除真实 `.env` 文件和所有真实 API Key。
- 将代码里的硬编码密钥替换为环境变量或占位符。
- 删除 IDE 配置、缓存、`.pyc`、模型二进制文件和本地数据库。
- 不包含 10GB 级别的原始 ESG PDF、完整批量拆分文本和全部中间输出。
- 仅保留少量示例输入、评估 CSV 与可视化图片，便于面试官理解实验结果。

## 项目特点

- 端到端搭建了 ESG 报告自动分析流程，而不是只做单点调用。
- 将长文档任务拆成章节、议题、摘要、评分和评估多个可控环节。
- 对比了多个大模型在同一 ESG 任务上的表现，并用 ROUGE/BERTScore/可视化图表进行量化分析。
- 做了一个 Streamlit 工具，把实验流程包装成可交互的原型系统。

