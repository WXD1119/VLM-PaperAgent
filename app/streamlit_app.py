"""Thin UI shell. Business logic belongs in src/paper_agent."""

try:
    import streamlit as st
except ImportError:
    st = None


def main() -> None:
    if st is None:
        raise RuntimeError("Install the ui extra: pip install -e .[ui]")
    st.set_page_config(page_title="VLM-PaperAgent", layout="wide")
    st.title("VLM-PaperAgent · 证据可追溯论文精读")
    st.file_uploader("上传 PDF", type=["pdf"])
    st.info("骨架已就绪：下一步接入 Parser、检索与 Reviewer 工作流。")


if __name__ == "__main__":
    main()

