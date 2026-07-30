"""Render the resume-facing VLM-PaperAgent functional architecture PNG."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2200, 1500
BG = "#F8FAFC"
TEXT = "#172033"
MUTED = "#526070"
LINE = "#667085"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE = font(46, True)
SUBTITLE = font(23)
SECTION = font(25, True)
NODE = font(22, True)
SMALL = font(18)


def center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=TEXT):
    left, top, right, bottom = box
    lines = text.split("\n")
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        heights.append(bbox[3] - bbox[1])
    total = sum(heights) + (len(lines) - 1) * 6
    y = (top + bottom - total) / 2
    for line, height in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = (left + right - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += height + 6


def node(draw: ImageDraw.ImageDraw, box, label: str, fill: str, outline: str, *, small=False):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    center_text(draw, box, label, SMALL if small else NODE)


def arrow(draw: ImageDraw.ImageDraw, start, end, *, dashed=False, color=LINE):
    if dashed:
        x1, y1 = start
        x2, y2 = end
        steps = 18
        for index in range(0, steps, 2):
            a = index / steps
            b = min(index + 1, steps) / steps
            draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=4)
    else:
        draw.line((*start, *end), fill=color, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    back = 18
    spread = 8
    points = [(end[0], end[1])]
    for offset in (math.pi - 0.45, math.pi + 0.45):
        points.append(
            (
                end[0] + back * math.cos(angle + offset),
                end[1] + back * math.sin(angle + offset),
            )
        )
    draw.polygon(points, fill=color)


def section(draw: ImageDraw.ImageDraw, box, title: str, accent: str):
    draw.rounded_rectangle(box, radius=22, fill="#FFFFFF", outline=accent, width=3)
    draw.rounded_rectangle((box[0] + 14, box[1] + 14, box[0] + 230, box[1] + 52), radius=12, fill=accent)
    center_text(draw, (box[0] + 22, box[1] + 17, box[0] + 222, box[1] + 49), title, SMALL, "#FFFFFF")


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((70, 45), "VLM-PaperAgent 功能架构", font=TITLE, fill=TEXT)
    draw.text(
        (72, 112),
        "Evidence-Grounded Research Copilot · 面向算法研发团队的可追溯论文调研与知识沉淀系统",
        font=SUBTITLE,
        fill=MUTED,
    )

    # 交互层
    section(draw, (65, 180, 2135, 365), "产品交互层", "#4769D8")
    user = (180, 245, 470, 330)
    web = (760, 245, 1070, 330)
    api = (1300, 245, 1600, 330)
    node(draw, web, "Web UI\nStreamlit", "#E8EDFF", "#4769D8")
    node(draw, api, "FastAPI API\n问答 / 图谱 / 导入", "#E8EDFF", "#4769D8")
    node(draw, user, "研发用户\n算法 / 工程团队", "#E8EDFF", "#4769D8")
    arrow(draw, (470, 287), (760, 287))
    arrow(draw, (1070, 287), (1300, 287))

    # 解析与知识层
    section(draw, (65, 405, 1040, 840), "论文导入与知识构建", "#1A8E75")
    upload = (115, 490, 315, 585)
    worker = (380, 490, 625, 585)
    normalize = (695, 490, 940, 585)
    index = (695, 675, 940, 770)
    artifacts = (380, 675, 625, 770)
    node(draw, upload, "PDF Upload Task\npending / running", "#E4F7F1", "#1A8E75", small=True)
    node(draw, worker, "Worker + MinerU\nOCR / Layout Parse", "#E4F7F1", "#1A8E75", small=True)
    node(draw, normalize, "Normalize + Chunk\npaper.json / chunks.json", "#E4F7F1", "#1A8E75", small=True)
    node(draw, artifacts, "Paper Artifacts\nPDF / MinerU / JSON", "#E4F7F1", "#1A8E75", small=True)
    node(draw, index, "ChromaDB + BM25\nDense / Sparse Index", "#E4F7F1", "#1A8E75", small=True)
    arrow(draw, (315, 537), (380, 537))
    arrow(draw, (625, 537), (695, 537))
    arrow(draw, (817, 585), (817, 675))
    arrow(draw, (625, 722), (695, 722))

    section(draw, (1090, 405, 2135, 840), "知识沉淀", "#9A5DC8")
    kg = (1160, 490, 1435, 585)
    workspace = (1530, 490, 1810, 585)
    neo = (1880, 490, 2080, 585)
    memory = (1370, 675, 1665, 770)
    node(draw, kg, "Paper KG\nJSONL Graph", "#F2E9FB", "#9A5DC8")
    node(draw, workspace, "Graph Workspace\n个人论文分支", "#F2E9FB", "#9A5DC8", small=True)
    node(draw, neo, "Neo4j\n可选物化视图", "#F2E9FB", "#9A5DC8", small=True)
    node(draw, memory, "Agent Memory\nSession / Episodic / Summary / Profile", "#F2E9FB", "#9A5DC8", small=True)
    arrow(draw, (1435, 537), (1530, 537))
    arrow(draw, (1810, 537), (1880, 537), dashed=True, color="#9A5DC8")

    # Agent 层
    section(draw, (65, 890, 2135, 1245), "可信回答与工具编排", "#D95B48")
    context = (120, 980, 365, 1085)
    orchestrator = (440, 980, 700, 1085)
    retrieval = (780, 945, 1030, 1030)
    graph = (780, 1080, 1030, 1165)
    research = (1120, 980, 1385, 1085)
    citation = (1470, 980, 1705, 1085)
    judge = (1785, 980, 2050, 1085)
    node(draw, context, "Context Guard\n实体消歧 + 多轮约束", "#FDECE8", "#D95B48", small=True)
    node(draw, orchestrator, "Tool Orchestrator\n依赖 / 并行 / 超时 / 重试", "#FDECE8", "#D95B48", small=True)
    node(draw, retrieval, "Hybrid Retrieval\nBM25 + Dense + RRF + Rerank", "#FDECE8", "#D95B48", small=True)
    node(draw, graph, "Paper Graph Tool\n概念 / 章节 / 证据", "#FDECE8", "#D95B48", small=True)
    node(draw, research, "Research Agent\nQwen · Answer + Claim + Citation", "#FDECE8", "#D95B48", small=True)
    node(draw, citation, "Citation Validator\n确定性引用校验", "#FDECE8", "#D95B48", small=True)
    node(draw, judge, "Judge Agent\nGLM · 语义支持审查", "#FDECE8", "#D95B48", small=True)
    arrow(draw, (365, 1032), (440, 1032))
    arrow(draw, (700, 1017), (780, 987))
    arrow(draw, (700, 1048), (780, 1122))
    arrow(draw, (1030, 987), (1120, 1017))
    arrow(draw, (1030, 1122), (1120, 1048))
    arrow(draw, (1385, 1032), (1470, 1032))
    arrow(draw, (1705, 1032), (1785, 1032))
    arrow(draw, (1918, 1085), (1250, 1108), dashed=True, color="#D95B48")
    draw.text((1480, 1100), "不支持 → 反馈重写 / 拒答", font=SMALL, fill="#B94739")

    # 评测与后续演进层
    section(draw, (65, 1290, 1020, 1440), "评测与质量回归", "#C78216")
    node(draw, (125, 1340, 360, 1412), "人工 Golden Set", "#FFF5DB", "#C78216", small=True)
    node(draw, (410, 1340, 665, 1412), "Retrieval Eval\nHit / Recall / MRR / nDCG", "#FFF5DB", "#C78216", small=True)
    node(draw, (715, 1340, 960, 1412), "Answer Eval\nCitation / Semantic Support", "#FFF5DB", "#C78216", small=True)
    arrow(draw, (360, 1376), (410, 1376))
    arrow(draw, (665, 1376), (715, 1376))

    section(draw, (1090, 1290, 2135, 1440), "P1：全链路可观测性", "#475569")
    node(draw, (1160, 1340, 1435, 1412), "Trace / Span\nAPI + Tool + Agent + Worker", "#EEF2F7", "#475569", small=True)
    node(draw, (1510, 1340, 1790, 1412), "Trace Detail\n耗时 / 错误 / 重试 / 降级", "#EEF2F7", "#475569", small=True)
    node(draw, (1855, 1340, 2075, 1412), "Metrics\nP50 / P95 / 失败率", "#EEF2F7", "#475569", small=True)
    arrow(draw, (1435, 1376), (1510, 1376))
    arrow(draw, (1790, 1376), (1855, 1376))

    output = Path("docs/system-architecture.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)
    print(output)


if __name__ == "__main__":
    main()
