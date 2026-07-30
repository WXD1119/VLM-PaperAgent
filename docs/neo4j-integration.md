# Neo4j 图数据库集成

Neo4j 是 Paper KG 的可选派生存储，不替代 JSONL 图谱，也不保存用户 profile、
session、普通对话或生成答案。JSONL/workspace 继续负责离线可复现与分支语义；Neo4j
用于 Cypher 查询、可视化和后续网站的图探索。

## 启动

```bash
docker compose up -d neo4j
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=paper-agent-dev
export NEO4J_DATABASE=neo4j
```

浏览器界面地址为 `http://127.0.0.1:7474`。首次启动可能需要几十秒。

## 导出图谱

导出基础图谱：

```bash
python scripts/export_neo4j.py --graph artifacts/graph --reset
```

导出个人 workspace 的有效图谱：

```bash
python scripts/export_neo4j.py \
  --workspace artifacts/graph_workspaces/ws_wxd_demo \
  --reset
```

`--reset` 只会删除 Neo4j 中已有的 `:GraphNode` 物化视图，不会修改 JSONL、PDF、
Chroma 或 workspace commit。

导出时会过滤旧版 workspace 中可能残留的 `Answer`、`Claim`、`Query` 节点，只将
`Paper`、`Section`、`Chunk`、`Concept` 及其关系写入数据库，保持 Paper KG 的论文内容边界。

## 远程服务器生成、本地 Neo4j 查询

如果远程服务器不能运行 Docker，不要复制 workspace 原目录后直接在本地加载：其中的
`workspace.json` 指向服务器绝对路径。先在服务器物化可携带图谱：

```bash
python scripts/materialize_workspace_graph.py \
  --workspace artifacts/graph_workspaces/ws_wxd_demo \
  --output artifacts/neo4j_export/ws_wxd_demo
```

将输出目录中的 `nodes.jsonl` 与 `edges.jsonl` 下载到本地同一路径，然后在本地导入：

```powershell
python scripts/export_neo4j.py `
  --graph artifacts/neo4j_export/ws_wxd_demo `
  --reset
```

## 查询

```bash
python scripts/query_neo4j.py --papers
python scripts/query_neo4j.py --concept "Q-Former"
```

节点使用稳定的 `node_id`，边使用稳定的 `edge_id`。Neo4j 内部采用统一标签
`:GraphNode` 和关系类型 `:GRAPH_EDGE`，业务类型通过 `node_type`、`edge_type` 属性
表达；这样避免动态拼接 Cypher 标签，并能保留 JSONL 中完整的领域模型。

## 验证原则

导出命令会打印写入节点/边数和数据库节点/边数。两组数在 `--reset` 导出后应一致。
若不一致，先检查 workspace 是否包含预期论文，再重新导出；不要直接把 Neo4j 当作
版本源修改。
