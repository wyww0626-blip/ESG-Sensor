from pathlib import Path
import os
import tempfile

import pandas as pd
import streamlit as st

from score_utils import run_score_for_dimension
from summary_utils import summarize_json_and_merge
from topic_utils import analyze_pdf


BASE_DIR = Path(__file__).resolve().parent
SCORE_CONFIG_DIR = BASE_DIR / "score_configs"
FEWSHOT_JSON_PATH = BASE_DIR / "img" / "dataset.json"
PAGES = ["总览", "ESG 评分", "议题识别", "摘要生成"]

APP_CSS = """
<style>
    :root {
        --bg: #f4f6f8;
        --surface: #ffffff;
        --surface-soft: #f8fafb;
        --ink: #18232b;
        --muted: #667684;
        --line: #dde5ea;
        --navy: #111821;
        --navy-2: #172433;
        --emerald: #1c8b68;
        --cyan: #0f7f91;
        --amber: #c58a25;
        --rose: #b85d70;
    }

    .stApp {
        background:
            radial-gradient(circle at 72% 8%, rgba(15,127,145,0.10), transparent 28%),
            linear-gradient(180deg, #f7f9fb 0%, var(--bg) 44%, #eef3f5 100%);
        color: var(--ink);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.86);
    }

    [data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 1.55rem;
        margin-top: 0.35rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.35rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.11);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    .hero {
        background: linear-gradient(135deg, #111821 0%, #173244 50%, #1b6d68 100%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 1.6rem 1.65rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 18px 45px rgba(24,35,43,0.12);
    }

    .eyebrow {
        color: #8be0c6;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.18rem;
        line-height: 1.14;
        margin: 0;
    }

    .hero p {
        color: rgba(255,255,255,0.78);
        font-size: 1rem;
        line-height: 1.68;
        max-width: 880px;
        margin: 0.8rem 0 0;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--ink);
        margin: 1.3rem 0 0.65rem;
    }

    .metric-wrap div[data-testid="stMetric"],
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.88);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.88rem 0.95rem;
        box-shadow: 0 10px 26px rgba(24,35,43,0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--cyan);
        font-size: 1.6rem;
        font-weight: 760;
    }

    .flow-card,
    .task-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        min-height: 132px;
        box-shadow: 0 10px 30px rgba(24,35,43,0.055);
    }

    .flow-card {
        border-top: 4px solid var(--emerald);
    }

    .flow-card strong,
    .task-card strong {
        display: block;
        color: var(--ink);
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }

    .flow-card span,
    .task-card span,
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.62;
    }

    .task-card {
        min-height: 150px;
        position: relative;
        overflow: hidden;
    }

    .task-card:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--emerald), var(--cyan), var(--amber));
    }

    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        border: 1px solid #183346;
        background: linear-gradient(135deg, #183346 0%, #116b72 100%);
        color: white;
        font-weight: 760;
        min-height: 2.65rem;
        box-shadow: 0 12px 22px rgba(17,107,114,0.15);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color: var(--emerald);
        background: linear-gradient(135deg, #123044 0%, #1c8b68 100%);
        color: white;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="select"] {
        border-radius: 8px;
    }

    .status-line {
        border-left: 4px solid var(--amber);
        background: rgba(255,255,255,0.76);
        border-radius: 8px;
        margin-top: 1rem;
        padding: 0.75rem 0.9rem;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .result-shell {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(24,35,43,0.055);
    }
</style>
"""


DIMENSIONS = {
    "环境": SCORE_CONFIG_DIR / "环境维度_评分点结构化.json",
    "社会": SCORE_CONFIG_DIR / "社会维度_评分点结构化.json",
    "治理": SCORE_CONFIG_DIR / "治理维度_评分点结构化.json",
}


def apply_page_config() -> None:
    st.set_page_config(
        page_title="ESG Report Intelligence",
        page_icon="ESG",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


def ensure_page_state() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "总览"


def go_to_page(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">ESG Report Intelligence</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    ensure_page_state()
    with st.sidebar:
        st.title("ESG Intelligence")
        st.caption("面向 ESG 报告的多模型分析工作台")
        current_index = PAGES.index(st.session_state["page"])
        selected_page = st.radio(
            "导航",
            PAGES,
            index=current_index,
            label_visibility="collapsed",
        )
        st.session_state["page"] = selected_page

        st.divider()
        st.caption("运行状态")
        st.write("公开版已移除真实密钥和完整原始数据。")
        st.write("需要调用模型时，请在 `.env` 中配置自己的 API Key。")
        return selected_page


def render_overview() -> None:
    render_header(
        "ESG 报告智能分析工作台",
        "从 PDF 解析、实质性议题识别、结构化摘要到 ESG 三维度评分，把长报告处理成可复核、可比较、可展示的分析结果。",
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("核心模块", "4")
    metric_cols[1].metric("评分维度", "E / S / G")
    metric_cols[2].metric("评估方法", "ROUGE + BERTScore")
    metric_cols[3].metric("公开版", "脱敏轻量化")

    st.markdown('<div class="section-title">分析流程</div>', unsafe_allow_html=True)
    flow_cols = st.columns(4)
    steps = [
        ("01 PDF 解析", "抽取报告文本，按标题和章节结构拆分长文档。"),
        ("02 议题定位", "识别实质性议题页，提取企业重点披露主题。"),
        ("03 摘要生成", "按章节和议题构造提示词，生成结构化摘要。"),
        ("04 模型评估", "用评分点、ROUGE、BERTScore 和图表比较结果。"),
    ]
    for col, (name, desc) in zip(flow_cols, steps):
        col.markdown(
            f'<div class="flow-card"><strong>{name}</strong><span>{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">演示入口</div>', unsafe_allow_html=True)
    entry_cols = st.columns(3)
    entries = [
        ("ESG 评分", "按环境、社会、治理评分点，对章节内容进行披露判断和量化评分。", "进入评分模块"),
        ("议题识别", "上传 ESG 报告 PDF，定位疑似实质性议题页面并生成分析。", "进入识别模块"),
        ("摘要生成", "上传结构化 JSON，生成合并摘要，并支持下载为文本。", "进入摘要模块"),
    ]
    for col, (page, desc, button_label) in zip(entry_cols, entries):
        with col:
            st.markdown(
                f'<div class="task-card"><strong>{page}</strong><span>{desc}</span></div>',
                unsafe_allow_html=True,
            )
            if st.button(button_label, key=f"go-{page}", use_container_width=True):
                go_to_page(page)


def render_score_page() -> None:
    render_header(
        "ESG 三维度评分",
        "基于环境、社会、治理评分点，对报告章节内容进行披露判断和量化评分。",
    )

    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        st.markdown('<div class="section-title">评分任务</div>', unsafe_allow_html=True)
        with st.form("score_form"):
            folder_path = st.text_input("章节文本目录", placeholder="例如 examples/report_chapters/京东健康_output")
            selected_dim = st.selectbox("评分维度", ["全部", "环境", "社会", "治理"])
            submitted = st.form_submit_button("开始评分", use_container_width=True)

        st.markdown(
            '<div class="status-line">输入目录应包含按章节拆分的 .txt 文件。公开版保留了评分配置，完整批量数据未包含在仓库中。</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">评分结果</div>', unsafe_allow_html=True)
        if not submitted:
            st.info("等待提交评分任务。")
            return

        if not folder_path or not os.path.isdir(folder_path):
            st.error("章节文本目录无效，请检查路径。")
            return

        target_dimensions = DIMENSIONS if selected_dim == "全部" else {selected_dim: DIMENSIONS[selected_dim]}
        results = []
        for name, config_path in target_dimensions.items():
            st.write(f"正在计算：{name}")
            progress_bar = st.progress(0)
            total, avg = run_score_for_dimension(str(config_path), name, folder_path, progress_bar)
            results.append({"维度": name, "总分": round(total, 2), "平均分": round(avg, 3)})

        result_df = pd.DataFrame(results)
        score_cols = st.columns(len(results))
        for col, item in zip(score_cols, results):
            col.metric(item["维度"], item["总分"], f"平均 {item['平均分']}")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.caption("模型评分适合用于初筛和横向比较，最终结论仍应结合人工复核。")


def render_topic_page() -> None:
    render_header(
        "实质性议题识别",
        "上传 ESG 报告 PDF，定位与实质性议题相关的页面，并生成视觉模型分析结果。",
    )

    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        st.markdown('<div class="section-title">上传报告</div>', unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader("PDF 文件", type=["pdf"])
        st.markdown(
            '<div class="status-line">公开版保留少样本配置。运行该模块需要配置模型 API Key。</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">识别结果</div>', unsafe_allow_html=True)
        if not uploaded_pdf:
            st.info("等待上传 PDF 文件。")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        with st.spinner("正在识别实质性议题页面..."):
            images, result, status = analyze_pdf(tmp_path, str(FEWSHOT_JSON_PATH))

        st.success(status)
        if images:
            st.markdown("定位页面")
            image_cols = st.columns(min(len(images), 3))
            for index, image_data in enumerate(images):
                image_cols[index % len(image_cols)].image(
                    f"data:image/png;base64,{image_data}",
                    caption=f"页面 {index + 1}",
                    use_container_width=True,
                )
        if result:
            st.markdown("模型分析")
            st.markdown(f'<div class="result-shell">{result}</div>', unsafe_allow_html=True)


def render_summary_page() -> None:
    render_header(
        "结构化摘要生成",
        "读取结构化 ESG 报告 JSON，按章节生成摘要并合并为可下载文本。",
    )

    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        st.markdown('<div class="section-title">上传 JSON</div>', unsafe_allow_html=True)
        uploaded_json = st.file_uploader("结构化报告文件", type=["json"])
        st.markdown(
            '<div class="status-line">适合上传由章节内容和议题提示词组成的 JSON 文件。</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">摘要结果</div>', unsafe_allow_html=True)
        if not uploaded_json:
            st.info("等待上传 JSON 文件。")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
            tmp_json.write(uploaded_json.read())
            tmp_json_path = tmp_json.name

        with st.spinner("正在生成摘要..."):
            summary_text = summarize_json_and_merge(tmp_json_path)

        st.text_area("合并摘要", summary_text, height=360)
        summary_txt_name = Path(uploaded_json.name).stem + "_summary.txt"
        st.download_button("下载 TXT", summary_text, file_name=summary_txt_name, use_container_width=True)


def main() -> None:
    apply_page_config()
    page = render_sidebar()

    if page == "总览":
        render_overview()
    elif page == "ESG 评分":
        render_score_page()
    elif page == "议题识别":
        render_topic_page()
    else:
        render_summary_page()


if __name__ == "__main__":
    main()
