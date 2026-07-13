# VLM-PaperAgent 简历与面试说明

这份文档用于把项目讲清楚：它不是论文，也不是功能堆砌清单，而是面试时可以直接复述的项目总结。

## 项目定位

VLM-PaperAgent 是一个面向视觉语言模型论文的证据可追溯阅读 Agent。用户可以上传或指定论文，系统自动解析 PDF，构建结构化 chunk、混合检索证据，并生成带引用的回答。每条回答 claim 都需要绑定 evidence ID，并通过引用完整性校验和独立语义 judge，降低幻觉风险。

一句话版本：

> 做了一个 evidence-grounded 论文阅读 Agent，把 PDF 论文解析、混合检索、精排、带引用回答、语义审查、轻量知识图谱和记忆系统串成了可复现闭环。

## 为什么这个项目值得写进简历

很多 RAG 项目只展示“能回答”，但没有证明答案是否可信。这个项目的核心价值是把“可信回答”拆成可验证的工程链路：

- 检索证据是否足够；
- 证据能否覆盖多个相关 chunk；
- 回答中的 claim 是否都引用了 evidence；
- 引用是否真的来自检索结果；
- 独立 judge 是否认为 claim 被证据支持；
- 模糊追问是否被 context guard 约束，而不是全库乱搜；
- 用户记忆和论文事实是否隔离，避免污染 Paper KG。

这比单纯说“用了多 Agent”更容易体现工程深度。

## 技术栈

- Python 3.11
- MinerU：PDF 解析与 OCR/结构化抽取
- Pydantic：领域模型和结构化输出校验
- BM25：关键词检索基线
- BGE-M3：dense embedding
- Chroma：本地向量库
- RRF：BM25 + dense 融合
- BGE-reranker-v2-m3：cross-encoder 精排
- Qwen3-VL-8B-Instruct：本地 answer model
- GLM-4.1V-9B-Thinking：本地 semantic judge
- FastAPI：服务化接口
- JSONL Paper KG：轻量图谱与 workspace 分支
- pytest：核心链路测试

## 核心模块

### 1. 论文解析与切片

输入 PDF 后，系统读取 MinerU 的结构化结果，统一转换成内部 `paper.json`，再构建 `chunks.json`。chunk 保留页码、章节路径、类型和原始 element ID。

重点设计：

- 过滤页眉、页码、脚注等噪声；
- 公式、表格、图注独立成 chunk；
- 超长 chunk 自动切分；
- 公式 chunk 绑定上下文解释；
- chunk ID 稳定可复现。

### 2. 检索与精排

支持四种检索方案：

| 方案 | 作用 |
|---|---|
| BM25 | 术语、表格名、数据集名等精确匹配强 |
| BGE-M3 dense | 语义匹配和跨语言查询更强 |
| BM25 + BGE-M3 RRF | 融合关键词和语义结果 |
| RRF + BGE reranker | 提升前 5 条证据的排序质量 |

最新 48-case VLM benchmark 结果：

| 方法 | Hit@1 | Hit@5 | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6250 | 1.0000 | 0.1679 | 0.6144 | 0.7733 | 0.6042 |
| BGE-M3 dense | 0.7083 | 1.0000 | 0.2030 | 0.6005 | 0.8420 | 0.6377 |
| BM25 + BGE-M3 RRF | 0.7917 | 1.0000 | 0.2252 | 0.6606 | 0.8889 | 0.6906 |
| RRF + BGE reranker | 0.7500 | 1.0000 | 0.2099 | 0.6944 | 0.8611 | 0.6943 |

结论：

- Hybrid 的 Hit@1 和 MRR 最好，适合作为默认在线检索方案。
- Reranker 的 Recall@5 和 nDCG@5 最好，适合高质量回答模式。
- Hit@5 全部达到 1.0，说明前 5 条总能找到至少一个可用证据。

### 3. 证据可追溯回答

AnswerAgent 不直接“自由发挥”，而是在一个 bounded evidence pack 内回答。输出结构包括：

- answer；
- claims；
- 每条 claim 的 evidence IDs；
- abstention 状态；
- citation validation 结果。

如果引用了不存在的 evidence ID，或者 claim 没有引用证据，validator 会判失败。

### 4. 独立语义 Judge

Citation validation 只能检查“有没有引用”和“引用 ID 是否存在”，不能判断 claim 是否真的被证据支持。所以项目加入独立 semantic judge：

- answer model：负责生成回答；
- judge model：逐条 claim 判断是否 supported / unsupported；
- 如果 judge 失败，反馈给 answer model 重写；
- 多次失败后安全 abstain。

这形成一个 generate-then-verify loop。

### 5. Memory 和 Paper KG 分离

项目明确区分：

- Paper KG：只保存论文事实、章节、chunk、concept；
- Session memory：当前论文、上一轮问题；
- Episodic memory：运行事件、答案 artifact；
- Profile memory：用户偏好，例如中文回答；
- Artifact memory：answer JSON、评测报告。

这样做是为了避免把用户偏好、模型回答、论文事实混在同一个图谱里。

### 6. Graph workspace

轻量图谱支持 workspace 分支，类似 Git branch：

- base graph 保存公共论文图谱；
- 用户可以 fork workspace；
- 新论文、新 concept、新私有分支增量写入 workspace；
- base graph 不被污染。

Neo4j 可以作为后续 backend，但当前 JSONL 图谱更轻、更容易测试。

## 面试时怎么讲

建议按这个顺序讲：

1. 先说痛点：普通 RAG 回答论文问题容易幻觉，且引用不可验证。
2. 再说方案：我把论文问答拆成解析、检索、回答、引用校验、语义审查五个可测模块。
3. 展示数据：10 篇 VLM 论文、48 个手工标注问题，做 BM25 / dense / hybrid / reranker 消融。
4. 讲工程坑：Chroma 空库导致 dense 结果异常，后来加了 collection_count 检查和 `--reset`。
5. 讲设计取舍：Recall@1 低不是坏事，因为任务是多证据检索，所以加了 Hit@K 来辅助解释。
6. 最后讲扩展：Neo4j、Web UI、多用户 workspace、更多 benchmark。

## 简历 bullet 版本

可以压缩成 3 条：

- 构建面向 VLM 论文的 evidence-grounded reading agent，支持 MinerU 结构化解析、公式/表格/图注 chunking、BM25+BGE-M3+RRF+BGE-reranker 混合检索，以及 Qwen3-VL 本地回答生成。
- 设计 claim-level citation validation 与独立 GLM semantic judge，形成 generate-then-verify 闭环；答案默认保存为 artifact，不污染 Paper KG，降低幻觉和记忆污染风险。
- 自建 10 篇 VLM 论文、48 个问题的多证据检索 benchmark，完成 BM25 / dense / hybrid / reranker 消融；Hybrid 达到 Hit@1=0.7917、MRR=0.8889，Reranker 达到 Recall@5=0.6944、nDCG@5=0.6943。

## 可演示命令

端到端回答：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/ask.py \
  --query "How does Q-Former bridge the frozen image encoder and frozen language model?" \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --embedding-model artifacts/models/bge-m3 \
  --reranker-model artifacts/models/bge-reranker-v2-m3 \
  --model artifacts/models/Qwen3-VL-8B-Instruct \
  --offline \
  --device cuda:0 \
  --llm-device cuda:1 \
  --output artifacts/answers/demo.qformer.answer.json
```

带 judge gate：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/ask.py \
  --query "How does Q-Former bridge the frozen image encoder and frozen language model?" \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --embedding-model artifacts/models/bge-m3 \
  --reranker-model artifacts/models/bge-reranker-v2-m3 \
  --model artifacts/models/Qwen3-VL-8B-Instruct \
  --offline \
  --device cuda:0 \
  --llm-device cuda:1 \
  --judge-url http://127.0.0.1:8765 \
  --max-answer-attempts 2 \
  --output artifacts/answers/demo.qformer.gated.answer.json
```

重建索引：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/index_dense.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0 \
  --reset
```

检索消融：

```bash
python scripts/evaluate_bm25.py \
  --chunks artifacts/papers \
  --golden evals/retrieval_golden.v2.json \
  --output artifacts/evals/bm25.v2.json
```

## 后续规划

优先级建议：

1. 做 Web UI，把 ask / evidence / judge / graph query 展示出来。
2. 接 Neo4j backend，但保留 JSONL graph 作为测试和离线模式。
3. 扩展 benchmark 到更多论文和 held-out split。
4. 加入 answer-level 人工评分或 RAGAS 类指标，但仍以 evidence support 为主。
