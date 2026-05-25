from pathlib import Path
import unittest


APP_FILE = Path(__file__).resolve().parents[1] / "app" / "main.py"


class StreamlitDashboardContractTest(unittest.TestCase):
    def test_dashboard_uses_stateful_sidebar_navigation(self):
        source = APP_FILE.read_text(encoding="utf-8")

        self.assertIn("st.sidebar", source)
        self.assertIn("st.radio", source)
        self.assertIn("st.session_state", source)
        self.assertIn("PAGES", source)
        self.assertIn("总览", source)
        self.assertIn("ESG 评分", source)
        self.assertIn("议题识别", source)
        self.assertIn("摘要生成", source)

    def test_overview_entries_are_real_navigation_buttons(self):
        source = APP_FILE.read_text(encoding="utf-8")

        self.assertIn("go_to_page", source)
        self.assertIn("st.button", source)
        self.assertIn('("ESG 评分",', source)
        self.assertIn('("议题识别",', source)
        self.assertIn('("摘要生成",', source)
        self.assertIn("go_to_page(page)", source)

    def test_dashboard_has_polished_layout_contract(self):
        source = APP_FILE.read_text(encoding="utf-8")

        self.assertIn("APP_CSS", source)
        self.assertIn("render_overview", source)
        self.assertIn("render_score_page", source)
        self.assertIn("render_topic_page", source)
        self.assertIn("render_summary_page", source)
        self.assertIn("metric(", source)
        self.assertIn("st.columns", source)


if __name__ == "__main__":
    unittest.main()
