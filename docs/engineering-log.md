# VLM-PaperAgent 工程日志

本文件持续记录真实实现、测试证据、失败案例与工程取舍。所有简历指标必须能追溯到这里或 `evals/` 中的固定评测结果。

消融和工程对照的集中汇总见 `docs/ablation-results.md`，机器可读注册表见 `evals/experiment_registry.json`。

## 2026-07-07：项目骨架与证据契约

### 完成内容

- 建立 `src` 布局和领域层、工作流层、基础设施端口、API/UI 骨架。
- 手写有限状态机调度器，加入合法转换、节点重试、最大步骤数和 checkpoint。
- `ReviewFinding` 增加约束：`SUPPORTED` 判定必须携带 `evidence_ids`。
- 实现 Reciprocal Rank Fusion 基础函数。

### 测试

- 服务器 Python 3.11.15 环境运行 3 个基线测试，全部通过。
- 首轮覆盖率 76%。未覆盖区域主要是尚未实现的 API、Parser、LLM 和 Storage 端口，以及 FSM 异常分支。

### 工程判断

- MVP 以证据可追溯为核心价值，多 Agent、Redis、Neo4j 和 Celery 按真实职责逐步接入。
- 第一版使用手写 FSM，LangGraph 保留为后续对照实现。

## 2026-07-07：MinerU 接入与 CUDA 兼容

### 完成内容

- 独立创建 MinerU 环境，避免 PyTorch/vLLM 依赖污染业务环境。
- 使用 MinerU 3.4.2 解析 BLIP-2、CLIP 和 Full-Atom Peptide Design 三篇论文。
- 实现 `MinerUAdapter`，将 `content_list.json` 转换为 `Paper/PaperElement`。
- 统一处理 0-based 页码、bbox、章节路径、稳定元素 ID、父级与相邻元素链接。

### 踩坑

1. **最新 hybrid-engine 无法在旧驱动运行**
   - 现象：RTX 4090、驱动 535.309.01、CUDA兼容上限 12.2；最新 vLLM/PyTorch 报驱动过旧。
   - 排除项：升级 Transformers 只能处理弃用警告，不能解决 CUDA驱动不兼容。
   - 处理：采用 `MinerU 3.4.2 + pipeline`；兼容方案为 PyTorch 2.6.0 cu118，长期方案是升级 NVIDIA驱动后再评估 hybrid-engine。
2. **页面噪声进入正文**
   - 现象：CLIP 初次转换 375 个元素，其中 47 个 header、47 个 page_number，未映射类型达 126。
   - 处理：过滤 header/page_number；page_footnote 保留但标记 `retrieval_excluded`；映射 aside_text 和 chart。
   - 结果：CLIP 降至 281 个有效元素，BLIP-2 降至 84 个，未映射类型为 0。
3. **表格 Caption 与正文割裂**
   - 现象：表格 HTML 在 `table_body`，语义说明在 `table_caption` metadata，直接检索表格缺少语境。
   - 处理：表格元素检索文本合并 Caption 与 HTML，同时保留原始 metadata。

### 真实解析结果

| 论文 | 页数 | 有效元素 | 特点 |
|---|---:|---:|---|
| BLIP-2 | 13 | 84 | 6 figure、9 table |
| CLIP | 48 | 281 | 5 image、20 chart、20 table |
| Full-Atom Peptide Design | - | 258 | 46 equation、12 figure、6 table；56 个正文块含行内 LaTeX |

### 测试

- Adapter 覆盖页码转换、章节路径、类型映射、稳定 ID、无效页码、图片 Caption、表格 Caption 合并和噪声过滤。
- 本地 bundled Python 完成编译检查和 Adapter 回归烟雾测试。
- 服务器正式 pytest 结果需在同步最新代码后补录。

## 2026-07-07：公式感知父子切片

### 完成内容

- 新增 `ParentChunk`、`RetrievalChunk` 和 `ChunkBundle` 领域模型。
- 普通段落按章节与字符预算合并，支持元素级重叠。
- 公式、表格、图片保留为独立证据块。
- 独立公式自动绑定同章节前后解释段落，行内 LaTeX 保留在原文本块并添加标记。
- 切片保留 `element_ids`、页码、章节路径、稳定 chunk ID 和 parent ID。
- `retrieval_excluded` 元素不进入父块或子块。

### 设计取舍

- 当前使用字符预算而不是模型 tokenizer，避免切片层绑定某个 Embedding 模型；接入具体 Embedding 后再增加 token-aware 策略。
- 单个超长元素暂不硬切，以避免在 LaTeX、HTML 表格或引用中间截断；通过 `oversized` metadata 暴露，后续专项处理。
- ParentChunk 保存完整章节上下文，检索命中 ChildChunk 后可以按需补回父级内容。

### 测试

- 多模态元素保持独立。
- 公式上下文绑定。
- 行内 LaTeX 标记。
- 页脚排除。
- Parent/Child 引用完整。
- Chunk ID 稳定性。

### 服务器回归结果

- 环境：Linux、Python 3.11.15、pytest 9.1.1、pytest-cov 7.1.0。
- 结果：9/9 测试通过，耗时 0.42 秒。
- 总覆盖率：84%（首次骨架测试为 76%）。
- 关键模块覆盖率：`chunk.py` 100%、`chunker.py` 91%、`mineru_adapter.py` 84%、`scheduler.py` 72%。
- 当前低覆盖区域主要是尚未进入实现阶段的 API、LLM、Storage、Parser Protocol，以及 FSM 异常分支。
- 真实论文的 `chunks.json` 尚未生成；自动化单测通过只证明切片逻辑，不能替代真实数据验收。

### 真实论文切片验收

三篇论文已生成 `chunks.json`，Parent/Child 引用全部有效：

| 论文 | Parent | Child | 类型分布 | 长度 min/median/max | 超长块 |
|---|---:|---:|---|---|---:|
| CLIP | 40 | 132 | text 87 / figure 25 / table 20 | 24 / 1499 / 23283 | 4 |
| Full-Atom Peptide Design | 34 | 148 | text 84 / equation 46 / figure 12 / table 6 | 21 / 356 / 3337 | 3 |
| BLIP-2 | 19 | 39 | text 22 / table 9 / figure 8 | 82 / 864 / 3519 | 1 |

- Full-Atom Peptide Design 的 46 个独立公式块全部绑定到解释上下文。
- 行内 LaTeX 文本块：CLIP 5、Full-Atom 46、BLIP-2 1。
- 当前主要风险是 8 个超长块；CLIP 最大块达到 23,283 字符，不能直接送入常见 Embedding 模型。
- 下一步先按块类型审计超长内容，再分别使用段落安全拆分、HTML 表格按行拆分或图表描述压缩策略，禁止统一硬截断。

超长块审计确认：5 个为普通文本（2,867-3,670 字符），3 个为 HTML 表格（3,060、3,519、23,283 字符）；没有超长公式或图片块。由此采用两种独立策略：

- 文本在段落/句末安全边界拆分，并避免在未闭合的 `$...$` 行内公式内断开。
- HTML 表格按 `<tr>` 分组，每个子块重复 Caption 与首行表头，保持可独立检索。
- 分片 Chunk ID 加入内容摘要，避免同一 PaperElement 拆分后发生 ID 冲突。

重新生成后，三篇论文均满足：超长块 0、重复 Chunk ID 0、无效 Parent 引用 0、公式缺少上下文 0。最大块长度分别为 CLIP 2,766、Full-Atom 2,777、BLIP-2 2,728 字符。数据摄取与切片阶段验收完成。

## 2026-07-07：BM25 稀疏检索基线

### 实现内容

- 实现依赖无关的 BM25Okapi，显式保留 `k1=1.5`、`b=0.75` 参数，便于面试解释和消融实验。
- 学术 tokenizer 支持英文术语、数字、中文字符和 LaTeX 命令，并在分词前移除 HTML 标签，避免 `table/tr/td` 污染词频。
- 支持按 `paper_id` 和块类型过滤，可单独检索 equation/table/figure/text。
- 检索结果返回分数、论文、页码、章节、Chunk ID、正文和公式上下文。
- 新增命令行搜索入口，能够直接加载多个 `chunks.json` 建立内存索引。

### 测试设计

- HTML 标签清理与 LaTeX 命令保留。
- 稀有精确术语排序。
- 公式检索及论文/类型过滤。
- 空查询和未登录词返回空结果。

### 服务器结果

- 15/15 测试通过，总覆盖率 85%。
- `Q-Former frozen image encoder`：Top-5 全部来自 BLIP-2，Top-1 精确命中模型架构段落。
- `flow matching objective theta vector field`：Top-5 同时召回定义文本、正文公式(1)、相关目标公式和附录重参数化公式。
- equation + paper 过滤正确工作。
- 暴露局限：仅查询 `theta vector field` 时，附录公式(39)/(40)因关键词更密集排在正文公式(1)前，说明 BM25不能充分理解“正文核心公式”的语义与篇章重要性。

## 2026-07-07：检索 Golden Set v0

### 完成内容

- 从真实查询结果中人工标注3个种子案例，保存 query、过滤条件和 relevant chunk IDs。
- 实现 Recall@1、Recall@5 和 MRR 自动评测及逐题结果输出。
- 保留一个 BM25 困难样例，避免评测集只包含“必然命中”的演示查询。

### 口径约束

- 3个 seed case 只验证评测工具链，不得写入简历作为性能指标。
- Dense/Hybrid 对比前至少扩展到30个问题，并记录数据集版本、标注人和 commit SHA。

### Seed 基线实测

- Cases：3。
- Macro Recall@1：0.3333。
- Macro Recall@5：1.0000。
- MRR：0.8333。
- `blip2_qformer_bridge`：R@1 0.5、R@5 1.0、RR 1.0。
- `flow_matching_objective`：R@1 0.5、R@5 1.0、RR 1.0。
- `flow_matching_equation_1`：R@1 0、R@5 1.0、RR 0.5；在限定 paper 和 equation 类型后，正文公式(1)排第2。
- 评测产物：`artifacts/evals/bm25.seed.json`。该文件是运行产物，不提交仓库；固定输入为 `evals/retrieval_golden.seed.json`。

## 2026-07-07：人工检索标注工具

### 实现内容

- 准备15题标注队列，三篇论文各5题，覆盖 method、equation、table、figure 和 evaluation。
- 交互式展示 BM25 Top-10 候选及分数、页码、章节、Chunk ID、正文和公式上下文。
- 支持序号列表与范围选择、跳过、安全退出、断点续标。
- 每题保存 expected answer、notes、annotator 和 category。
- 使用临时文件替换方式原子保存，避免标注过程中断损坏 Golden JSON。
- 检测重复 query ID 及同论文重复问题。

### 防止评测污染

- BM25只用于召回候选，不自动决定 relevant label。
- 标注人必须阅读候选内容后确认；无法判断的问题应跳过，不为凑数量强制标注。
- 后续应抽样复核，Dense/Hybrid 评测不得反向修改标签迎合结果。

## 2026-07-07：Dense Retrieval 与 Chroma

### 技术选型

- 默认 Embedding：`BAAI/bge-m3`，支持多语言、长文本和1024维归一化Dense向量，适合中文问题检索英文论文。
- 向量存储：Chroma PersistentClient；MVP规模仅数百个Chunk，嵌入式持久化比独立向量数据库服务更轻。
- 业务侧显式生成Embedding并传入Chroma，禁止依赖Chroma默认Embedding函数，确保模型版本可追溯。

### 实现内容

- `EmbeddingEncoder` Protocol与懒加载Sentence Transformers实现。
- 参考内存余弦索引，用Fake Encoder完成不下载模型的单元测试。
- Chroma幂等upsert、cosine查询、paper/kind元数据过滤。
- Collection记录模型名和向量维度；重新打开时配置不一致直接失败，防止混用Embedding空间。
- Dense建库与查询CLI，结果保留paper、页码、章节、Chunk ID和上下文。

### 风险与验证计划

- 服务器驱动只兼容CUDA 12.2，需先安装兼容的PyTorch cu118，再安装Sentence Transformers/Chroma。
- 首次运行需下载约GB级模型；必须记录模型revision、索引数量和建库耗时。
- Chroma真实集成测试将在服务器安装retrieval extras后执行；本地单元测试只验证检索契约和余弦排序。

### 踩坑：服务器无法访问Hugging Face

- 现象：Sentence Transformers初始化时请求 `huggingface.co/BAAI/bge-m3`，服务器报 `[Errno 101] Network is unreachable`，重试后失败。
- 根因：模型权重尚未下载，服务器无Hugging Face出口；与CUDA和Chroma无关。
- 处理：通过可访问的ModelScope下载完整模型仓库到项目外部/ignored模型目录；Dense CLI新增 `--offline`，Sentence Transformers以 `local_files_only=True` 从本地路径加载，禁止隐式联网。

### 首次真实Dense建库

- 本地模型路径：`artifacts/models/bge-m3`，离线加载成功。
- Embedding维度：1024。
- 写入Chunk：328；Chroma collection count：328。
- 数据库目录：`artifacts/chroma`。
- Sentence Transformers将维度接口重命名为 `get_embedding_dimension`；代码现优先使用新接口并兼容旧版本，消除FutureWarning。
- Dense评测复用BM25的同一Golden Case与Recall/MRR实现，避免两套评测口径漂移。

### Dense Seed结果与Hybrid RRF

- Dense Seed：Recall@1=0.1667、Recall@5=0.6667、MRR=0.5000。
- 中文Q-Former查询Top-1正确；公式精确定位不如BM25，验证了稀疏与稠密检索的互补性。
- 新增HybridRetriever，以RRF融合BM25与Dense候选，并保留sparse/dense rank用于结果解释。
- 新增混合查询与评测脚本，待服务器实测后登记Hybrid指标。
- Hybrid Seed实测：Recall@1=0.3333、Recall@5=1.0000、MRR=0.7333；优于Dense，但因公式目标降至第5名，MRR暂低于BM25。
- Q-Former结果保留来源排名：正确架构块sparse_rank=3、dense_rank=1，融合后Top-1，直观展示两路检索互补性。
- 本地Codex Python为Pydantic v1，而项目和服务器使用v2；本地导入测试会被旧版字段约束拦截。本地已完成compileall，完整pytest以服务器锁定环境为准。

## 待办

## 2026-07-08: Evidence-grounded Answer MVP

- Added request-local evidence packs with stable `E1...En` identifiers.
- Added structured claim-level answers, explicit abstention, and citation integrity validation.
- Added an OpenAI-compatible client so DeepSeek and GLM can be switched by configuration.
- Added the end-to-end `answer_question.py` CLI on top of RRF plus BGE reranking.
- Model policy: use a low-cost text model for normalized MinerU evidence; reserve a vision model
  for questions whose selected figure cannot be answered from caption or OCR.
- Added three unit tests covering traceable claims, unknown citations, and invalid abstention.
- Local `compileall` and `git diff --check` passed. Full pytest remains a server-side check because
  the local desktop environment contains Pydantic v1 while the project requires Pydantic v2.7+.
- Selected the locally downloaded `Qwen3-VL-8B-Instruct` as the free default model. Added a
  Transformers structured-output client and retained the paid OpenAI-compatible adapter only as
  an optional experiment baseline. Direct Transformers inference was chosen to reduce the CUDA
  compatibility risk previously encountered with vLLM on the NVIDIA 535 driver.
- First server smoke test succeeded with local Qwen3-VL-8B-Instruct. For the Q-Former bridge
  question, the model returned four structured claims, cited only supplied IDs `E1`-`E5`, and
  passed citation-integrity validation. Manual review flagged claim 3 ("reducing the burden on the
  LLM") as a useful hard case: its citation is syntactically valid, but semantic support may be
  weaker than the generated wording. This confirms the next metric must distinguish citation
  validity from claim-evidence entailment.
- Abstention smoke test succeeded: an out-of-scope GPT-4 training question produced no claims and
  did not attach irrelevant retrieved chunks as citations. The CLI now prints `abstained` and the
  abstention reason explicitly instead of exposing only an empty claim list.
- Equation smoke test succeeded: the conditional flow matching objective retrieved the definition
  and equation from page 3 and bound the formula and conditioning explanation to separate evidence.
- Added an optional batched semantic citation judge with `supported`, `partially_supported`, and
  `unsupported` verdicts. It validates assessment cardinality and evidence IDs. Since generation
  and judging use the same Qwen checkpoint, this guard is not treated as an independent metric.
- Prepared an isolated `judge-vlm` environment for the independent GLM judge. Server verification:
  PyTorch 2.7.1+cu118, Transformers 4.57.6, CUDA available on RTX 4090. The locally downloaded
  `GLM-4.1V-9B-Thinking` checkpoint occupies about 20 GB and all four safetensors shards are
  present. Keeping this dependency stack separate prevents changes to the retrieval/Qwen runtime.
- GLM smoke inference succeeded with the checkpoint split across two RTX 4090 GPUs and classified
  the controlled Q-Former claim as supported. The NVIDIA 535 driver emitted a P2P warning, but the
  generation completed correctly. Added a localhost-only FastAPI judge service and a dependency-
  free remote structured client, enabling heterogeneous Qwen generation and GLM judging across
  isolated Conda environments. The service dynamically supports one or more visible GPUs.
- Real heterogeneous execution exposed a resource bottleneck: Qwen occupied GPUs 2/3 while GLM on
  GPU 0 exceeded the 19 GiB budget and offloaded parameters to CPU, making semantic generation
  exceed practical request latency. The preferred path is now sequential offline judging: persist
  an immutable `AnswerBundle`, terminate Qwen, then load GLM across GPUs 2/3 with `judge_answer.py`.
- Independent GLM judging succeeded for the conditional flow matching answer. Claim 1's objective
  equation was judged supported by E2, and claim 2's conditioning explanation was judged supported
  by E1; `all_supported=true`. The structured report was persisted to
  `artifacts/evals/cfm.glm-judge.json`. This is an integration result, not an accuracy metric; a
  manually labeled multi-case citation set is required before reporting Judge precision/recall.
- Added resumable human citation annotation over immutable answer bundles. Labels use the same
  three-class contract as the Judge but are entered independently while the claim and cited
  evidence are displayed. Added automatic Claim Accuracy, Macro-F1, Supported Precision/Recall,
  and Abstention Accuracy evaluation with strict case/claim alignment checks.

## 2026-07-08：远程VS Code自动评测

- 通过VS Code Remote SSH连接服务器 `122.207.108.8`，在 `/workspace/guest/wxd/VLM-PaperAgent` 的 Conda `paper-agent` 环境执行。
- 回归测试：26 passed in 0.11s。
- 冻结Golden Set：16题，空证据0、重复query_id 0；SHA256为 `6a723400e056c919d3d7fa23da85b614b9f5fe46d00290ee6cc565299cbb1bb4`。
- BM25：Recall@1=0.1875、Recall@5=0.8047、MRR=0.4917。
- BGE-M3 Dense：Recall@1=0.3828、Recall@5=0.7734、MRR=0.6406。
- BM25 + BGE-M3 + RRF：Recall@1=0.4766、Recall@5=0.8438、MRR=0.7208。
- 远程项目目录缺少 `.git`，无法通过git pull可靠同步；当前依赖WinSCP，后续应改成Git仓库作为唯一代码源，数据与模型继续留在服务器且由.gitignore排除。
- 新增二元相关性nDCG@5：对相关证据按排名对数折损并以理想DCG归一化；三种评测CLI均输出逐题和宏平均nDCG@5。
- nDCG@5实测：BM25=0.5637、BGE-M3 Dense=0.6552、RRF=0.7358；RRF相对BM25绝对提升0.1721，相对Dense提升0.0806。
- 代表性困难样例：`clip_architectures`在BM25命中但融合后跌出Top-5；`peptide_evaluation_metrics`的相关表格位于BM25第10名，三种Top-5均未命中。二者将作为加权RRF与Top-20 Reranker的回归用例。

## 2026-07-08：Cross-Encoder Reranker骨架

- 新增可替换的`Reranker`协议与`RerankedRetriever`，将候选生成和精排解耦。
- 默认采用多语言`BAAI/bge-reranker-v2-m3`，支持本地模型路径、离线加载、GPU和批处理。
- RRF先生成Top-20，Cross-Encoder联合编码query与候选证据，再输出Top-5。
- 精排结果保留原始retrieval rank、RRF score、sparse rank和dense rank，便于解释排序变化。
- 新增查询、评测CLI和Fake Reranker单元测试；真实消融结果待服务器模型准备后填写。
- Reranker实测：Recall@1=0.5938、Recall@5=0.8984、MRR=0.7969、nDCG@5=0.8019，四项均优于未精排RRF。
- `clip_architectures`恢复Top-5命中；`peptide_evaluation_metrics`仍为0，后续需审计RRF Top-20候选覆盖与表格序列化质量。

- 对三篇真实论文生成 `chunks.json`，统计块类型、长度分布、超长块和公式上下文质量。
- 补齐 FSM 重试、非法转换、节点缺失与最大步骤异常测试。
- 实现 BM25、Dense Retrieval、RRF 与 Reranker 消融评测。

## 2026-07-09：AnswerAgent 与独立 Citation Judge 闭环

- 生成并保存三个不可变 `AnswerBundle`：`cfm`、`blip2_qformer_bridge` 和 `gpt4_learning_rate`。其中前两个为可回答样例，共 6 条 claim；第三个为拒答案例，用于验证 abstention。
- 人工标注 `evals/citation_golden.v1.json`，标注协议与 Judge 一致：claim 级别 `supported`、`partially_supported`、`unsupported`，并单独记录 abstention decision 是否正确。
- 使用 `GLM-4.1V-9B-Thinking` 在独立 `judge-vlm` 环境中顺序评审，避免 Qwen 与 GLM 同时驻留导致显存和 CPU offload 问题。
- 暴露并修复三类本地 thinking 模型结构化输出问题：
  - JSON 后追加 `<think>` 或额外对象导致 `Extra data`，通过首个括号配平 JSON 提取解决。
  - 批量评审遗漏或重复 claim，自动 fallback 到逐 claim 评审。
  - 模型复读 JSON Schema 而不是生成 JSON instance，改用具体 JSON 模板并显式拒绝 schema 对象。
- 回归测试入口：`python -m pytest tests/test_answer_agent.py -v`；服务器 `paper-agent` 环境使用 Pydantic v2，作为最终测试环境。
- Citation Judge smoke evaluation：
  - cases=3，claims=6。
  - Claim Accuracy=1.0000。
  - Macro F1=1.0000。
  - Supported Precision=1.0000。
  - Supported Recall=1.0000。
  - Abstention Accuracy=1.0000。
- 评测产物：`artifacts/evals/citation_judge.v1.json`；机器可读实验登记：`evals/experiment_registry.json` 的 `citation_judge_smoke_v1_3`。
- 限制：该结果仅验证 AnswerAgent → immutable answer bundle → GLM semantic judge → human golden evaluation 的闭环和鲁棒性；样本量只有 3，不作为最终泛化指标。

## 2026-07-09：一条命令端到端 Demo

- 新增 `scripts/ask.py`，作为展示友好的端到端入口：RRF 混合检索、可选 BGE Cross-Encoder 精排、Qwen 本地回答、citation integrity validation 和 `AnswerBundle` 保存。
- 默认使用 `artifacts/papers`、`artifacts/chroma`、`artifacts/models/bge-m3`、`artifacts/models/bge-reranker-v2-m3` 和 `artifacts/models/Qwen3-VL-8B-Instruct`，与服务器离线部署路径保持一致。
- 输出分为 `Answer`、`Claims`、`Citation integrity`、`Evidence` 和 `Run` 五段，便于面试演示和日志截图。
- 支持 `--no-rerank` 快速 smoke test，`--paper-id` 限定单篇论文，`--kind equation|table|figure|text` 检查特定模态证据。
- 新增 `tests/test_ask_cli.py` 覆盖渲染函数，确保回答、claim citation、证据卡片和 validation 状态稳定展示。
- 本地 Windows 环境仍受 Pydantic v1 限制，已完成 `py_compile` 与 `git diff --check`；完整 pytest 以服务器 `paper-agent` 环境为准。
