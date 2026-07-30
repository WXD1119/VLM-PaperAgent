"""VLM-PaperAgent 的 Streamlit 阅读与运行追踪界面。"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import streamlit as st
except ImportError:
    st = None


DEFAULT_API_URL = os.getenv("PAPER_AGENT_API_URL", "http://127.0.0.1:8000")


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    """调用 FastAPI，并将 HTTP 异常转换为适合页面展示的错误。"""

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API 返回 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 API: {exc.reason}") from exc


def ask_stream(base_url: str, payload: dict[str, Any]):
    """消费 /ask/stream 的 SSE 事件，仅展示受控执行状态。"""

    request = Request(
        f"{base_url.rstrip('/')}/ask/stream",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            event_type = "progress"
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("event: "):
                    event_type = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    yield event_type, json.loads(line.removeprefix("data: "))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API 返回 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 API: {exc.reason}") from exc


_STAGE_LABELS = {
    "resolve_context": "正在判断论文范围与多轮上下文",
    "clarification": "问题需要澄清，正在生成澄清提示",
    "retrieve_evidence": "正在检索候选论文证据",
    "build_evidence": "正在整理可引用证据包",
    "generate_answer": "正在生成带引用的回答",
    "validate_citations": "正在校验 Claim 与引用完整性",
    "semantic_judge": "独立 Judge 正在审查 Claim 是否被证据支持",
    "refuse": "证据不足或审查未通过，正在安全拒答",
    "persist_memory": "正在保存受控会话记忆",
    "memory_degraded": "记忆服务不可用，已降级且不影响当前回答",
}


def upload_pdf(base_url: str, filename: str, content: bytes) -> dict[str, Any]:
    """构造最小 multipart 请求上传 PDF，不在浏览器线程解析论文。"""

    boundary = f"----VLMpaperagent{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    body = prefix + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/ingestion/tasks",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API 返回 HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 API: {exc.reason}") from exc


def render_answer(response: dict[str, Any]) -> None:
    """展示回答、claim、证据和本次请求的运行追踪入口。"""

    bundle = response["bundle"]
    answer = bundle["answer"]
    if answer["abstained"]:
        st.warning(answer["answer"])
        if answer.get("abstention_reason"):
            st.caption(f"原因：{answer['abstention_reason']}")
    else:
        st.markdown("### 回答")
        st.write(answer["answer"])

    left, right = st.columns(2)
    with left:
        citation = bundle["citation_validation"]
        st.metric("引用完整性", "通过" if citation["valid"] else "未通过")
    with right:
        gate = response.get("semantic_gate_passed")
        st.metric("语义审查", "未启用" if gate is None else ("通过" if gate else "未通过"))

    if response.get("trace_id"):
        st.info(f"本次运行 Trace ID：`{response['trace_id']}`。可在“运行追踪”页面查看节点耗时与决策。")

    context = response.get("context")
    if context:
        with st.expander("上下文与消歧", expanded=False):
            st.json(context)

    st.markdown("### 结论与证据")
    evidence_by_id = {item["evidence_id"]: item for item in bundle["evidence_pack"]["items"]}
    for index, claim in enumerate(answer.get("claims", []), start=1):
        evidence_ids = ", ".join(claim["evidence_ids"])
        st.markdown(f"**{index}. {claim['text']}**")
        st.caption(f"引用：{evidence_ids or '无'}")
        for evidence_id in claim["evidence_ids"]:
            item = evidence_by_id.get(evidence_id)
            if item:
                with st.expander(f"{evidence_id} · {item['paper_id']} · 页码 {item['pages']}"):
                    st.caption(" > ".join(item["section_path"]))
                    st.write(item["content"])

    memory = response.get("memory")
    if memory:
        with st.expander("记忆写入策略", expanded=False):
            st.write(memory.get("status", ""))
            st.caption(memory.get("recommendation", ""))


def render_graph(base_url: str) -> None:
    st.subheader("论文知识图谱")
    st.caption("图谱只保存论文内容；对话、用户偏好和生成回答保存在独立 Agent Memory。")
    keyword = st.text_input("搜索概念", placeholder="例如：Q-Former、cross attention、alignment")
    if keyword:
        data = request_json(base_url, f"/graph/concepts?{urlencode({'q': keyword, 'limit': 8})}")
        if data["requires_disambiguation"]:
            st.info("该关键词可能对应多个概念，请从候选概念中补充更具体的名称。")
        if data["resolved_concept"]:
            st.success(f"已定位概念：{data['resolved_concept']}")
        if data["candidates"]:
            st.write("候选概念：", "、".join(data["candidates"]))
        if data["papers"]:
            st.markdown("#### 涉及论文")
            st.dataframe(data["papers"], use_container_width=True, hide_index=True)
        if data["chunks"]:
            st.markdown("#### 相关证据块")
            for chunk in data["chunks"]:
                with st.expander(f"{chunk['paper_id']} · 页码 {chunk['pages']} · {chunk['chunk_id']}"):
                    st.caption(" > ".join(chunk["section"]))
                    st.write(chunk["preview"])
        if data["related_concepts"]:
            st.caption("相关概念：" + "、".join(data["related_concepts"]))

    if st.button("加载论文列表"):
        st.dataframe(request_json(base_url, "/papers"), use_container_width=True, hide_index=True)


def render_ingestion(base_url: str) -> None:
    st.subheader("导入论文")
    st.caption("上传只创建任务；请在服务器独立 Worker 中执行解析，避免网页被 OCR 长任务阻塞。")
    uploaded = st.file_uploader("选择 PDF", type=["pdf"])
    if uploaded is not None and st.button("创建导入任务", type="primary"):
        with st.spinner("正在上传并创建任务…"):
            task = upload_pdf(base_url, uploaded.name, uploaded.getvalue())
        st.success(f"已创建任务：{task['task_id']}")
        st.code("python scripts/run_ingestion_worker.py --once", language="bash")

    if st.button("刷新任务状态"):
        tasks = request_json(base_url, "/ingestion/tasks?limit=20")
        if not tasks:
            st.info("还没有导入任务。")
            return
        rows = [
            {
                "任务": task["task_id"],
                "文件": task["filename"],
                "状态": task["status"],
                "阶段": task["stage"],
                "论文": task.get("paper_id") or "-",
                "更新时间": task["updated_at"],
            }
            for task in tasks
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        for task in tasks:
            if task["status"] == "succeeded" and not task.get("workspace_commit_id"):
                if st.button("提交到个人图谱分支", key=f"promote-{task['task_id']}"):
                    result = request_json(
                        base_url,
                        f"/ingestion/tasks/{task['task_id']}/promote",
                        method="POST",
                        payload={},
                    )
                    st.success(f"图谱提交状态：{result['status']}；Commit：{result.get('commit_id') or '-'}")
            if task["status"] == "failed":
                with st.expander(f"失败详情：{task['filename']}"):
                    st.error(task.get("error") or "未知错误")
                    st.code("\n".join(task.get("logs", [])[-20:]))


def render_traces(base_url: str) -> None:
    """展示脱敏 Trace，帮助定位慢节点、拒答和重试。"""

    st.subheader("运行追踪")
    st.caption("Trace 不保存原始问题或论文正文，只保存哈希、节点耗时和受控决策元数据。")
    if st.button("刷新 Trace"):
        st.session_state["traces"] = request_json(base_url, "/traces?limit=30")
    traces = st.session_state.get("traces", [])
    if not traces:
        st.info("暂无 Trace。先在“证据问答”页完成一次提问，再刷新此页面。")
        return
    rows = [
        {
            "Trace ID": trace["trace_id"],
            "状态": trace["status"],
            "运行时": trace["runtime"],
            "开始时间": trace["started_at"],
            "节点数": len(trace["events"]),
            "证据数": trace.get("attributes", {}).get("evidence_count", "-"),
        }
        for trace in traces
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    selected = st.selectbox("选择 Trace", options=[trace["trace_id"] for trace in traces])
    if selected:
        trace = request_json(base_url, f"/traces/{selected}")
        st.markdown("#### 节点耗时")
        st.dataframe(trace["events"], use_container_width=True, hide_index=True)
        with st.expander("完整脱敏 Trace", expanded=False):
            st.json(trace)


def render_research_plans(base_url: str) -> None:
    """展示可确认、可回看的复杂调研计划，避免把规划过程混入普通问答。"""

    st.subheader("多论文调研计划")
    st.caption("先创建草案并检查论文范围；只有点击确认执行后，系统才会逐篇运行证据问答。")
    with st.form("research-plan-create"):
        goal = st.text_area("调研目标", placeholder="例如：比较这些方法如何连接视觉编码器与语言模型，以及各自的训练策略。")
        paper_ids_raw = st.text_area("论文 ID（每行一个，至少两篇）", placeholder="paper_...\npaper_...")
        submitted = st.form_submit_button("创建调研草案")
    if submitted:
        paper_ids = [line.strip() for line in paper_ids_raw.splitlines() if line.strip()]
        plan = request_json(
            base_url,
            "/research/plans",
            method="POST",
            payload={"goal": goal.strip(), "paper_ids": paper_ids},
        )
        st.success(f"已创建草案：{plan['plan_id']}。请检查任务后再确认执行。")
        st.session_state["selected_plan"] = plan["plan_id"]

    if st.button("刷新调研计划"):
        st.session_state["research_plans"] = request_json(base_url, "/research/plans?limit=20")
    plans = st.session_state.get("research_plans", [])
    if not plans:
        st.info("暂无调研计划。创建草案后会显示在这里。")
        return
    st.dataframe(
        [
            {"计划": plan["plan_id"], "目标": plan["goal"], "状态": plan["status"], "更新时间": plan["updated_at"]}
            for plan in plans
        ],
        use_container_width=True,
        hide_index=True,
    )
    selected = st.selectbox(
        "选择计划",
        options=[plan["plan_id"] for plan in plans],
        index=0,
    )
    plan = request_json(base_url, f"/research/plans/{selected}")
    st.markdown("#### 任务清单")
    st.dataframe(
        [
            {
                "任务": task["title"],
                "类型": task["task_type"],
                "论文": task.get("paper_id") or "-",
                "状态": task["status"],
                "证据数": len(task.get("evidence_chunk_ids", [])),
            }
            for task in plan["tasks"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    if plan["status"] != "completed":
        confirmed = st.checkbox("我已检查论文范围与任务清单，确认执行", key=f"confirm-{selected}")
        if st.button("确认并执行计划", type="primary", disabled=not confirmed):
            with st.status("正在逐篇执行可追溯问答，请勿关闭页面…", expanded=True) as status:
                result = request_json(
                    base_url,
                    f"/research/plans/{selected}/execute",
                    method="POST",
                    payload={"confirm": True},
                )
                status.update(label=f"计划执行完成：{result['status']}", state="complete", expanded=False)
            plan = result
    if plan.get("final_report"):
        with st.expander("查看按论文保留证据来源的汇总", expanded=True):
            st.markdown(plan["final_report"])


def main() -> None:
    if st is None:
        raise RuntimeError("请安装 ui extra：pip install -e '.[ui]'")
    st.set_page_config(page_title="VLM-PaperAgent", page_icon="📚", layout="wide")

    with st.sidebar:
        st.header("连接")
        api_url = st.text_input("API 地址", value=DEFAULT_API_URL)
        if st.button("检查服务"):
            try:
                st.success(f"API 正常：{request_json(api_url, '/health')['status']}")
            except RuntimeError as exc:
                st.error(str(exc))
        st.divider()
        page = st.radio("功能", ["证据问答", "多论文调研", "导入论文", "知识图谱", "运行追踪"], label_visibility="collapsed")

    st.title("VLM-PaperAgent 阅读台")
    st.caption("可追溯检索 · 引用约束回答 · 语义审查 · LangGraph 运行追踪 · 论文知识图谱")
    try:
        if page == "多论文调研":
            render_research_plans(api_url)
        elif page == "知识图谱":
            render_graph(api_url)
        elif page == "导入论文":
            render_ingestion(api_url)
        elif page == "运行追踪":
            render_traces(api_url)
        else:
            query = st.text_area(
                "你的论文问题",
                placeholder="例如：Q-Former 如何连接冻结的图像编码器和语言模型？",
                height=100,
            )
            left, middle, right = st.columns(3)
            with left:
                top_k = st.slider("证据数量", 1, 10, 5)
            with middle:
                paper_id = st.text_input("限定论文 ID（可选）")
            with right:
                rerank = st.checkbox("启用精排", value=True)
            if st.button("开始检索并回答", type="primary", disabled=not query.strip()):
                payload = {
                    "query": query.strip(),
                    "top_k": top_k,
                    "paper_id": paper_id.strip() or None,
                    "use_rerank": rerank,
                    "use_context_guard": True,
                }
                status_box = st.status("正在执行受控研究工作流…", expanded=True)
                answer_response = None
                try:
                    for event_type, event in ask_stream(api_url, payload):
                        if event_type == "started":
                            status_box.write(f"⏳ {event['message']}")
                        elif event_type == "node":
                            label = _STAGE_LABELS.get(str(event.get("name")), str(event.get("name")))
                            if event.get("status") == "running":
                                status_box.write(f"⏳ {label}")
                            elif event.get("status") == "success":
                                status_box.write(f"✅ {label}（{event.get('elapsed_ms', 0)} ms）")
                            else:
                                status_box.write(f"⚠️ {label}：{event.get('error', '失败')}")
                        elif event_type == "answer":
                            answer_response = event["response"]
                        elif event_type == "error":
                            raise RuntimeError(str(event.get("error", "问答执行失败")))
                    if answer_response is None:
                        raise RuntimeError("服务未返回最终回答")
                    status_box.update(label="回答已完成；可展开查看执行过程", state="complete", expanded=False)
                    render_answer(answer_response)
                except RuntimeError:
                    status_box.update(label="问答执行失败", state="error", expanded=True)
                    raise
    except RuntimeError as exc:
        st.error(str(exc))
        st.info("请确认 FastAPI 服务已启动，并检查侧边栏中的 API 地址。")


if __name__ == "__main__":
    main()
