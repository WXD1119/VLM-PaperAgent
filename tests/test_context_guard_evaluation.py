import json
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("context guard evaluation tests require pydantic v2", allow_module_level=True)

from pydantic import TypeAdapter

from paper_agent.evaluation import ContextGuardCase, evaluate_context_guard
from paper_agent.graph import EdgeType, GraphDocument, GraphEdge, GraphNode, NodeType, write_graph_jsonl
from paper_agent.memory import ContextGuard, GraphEntityResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "evals" / "context_guard.seed.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_context_guard.py"


def build_entity_graph() -> GraphDocument:
    nodes = [
        GraphNode(node_id="paper:paper_6bc5d399d6127a64", node_type=NodeType.PAPER, label="BLIP-2", properties={"paper_id": "paper_6bc5d399d6127a64", "title": "BLIP-2: Bootstrapping Language-Image Pre-training"}),
        GraphNode(node_id="paper:paper_dc8bace378a282ea", node_type=NodeType.PAPER, label="Visual Instruction Tuning", properties={"paper_id": "paper_dc8bace378a282ea", "title": "Visual Instruction Tuning"}),
        GraphNode(node_id="paper:paper_6478b6e571a7d6fc", node_type=NodeType.PAPER, label="Learning Transferable Visual Models From Natural Language Supervision", properties={"paper_id": "paper_6478b6e571a7d6fc", "title": "Learning Transferable Visual Models From Natural Language Supervision", "source_path": "data/raw/CLIP.pdf"}),
        GraphNode(node_id="concept:q-former", node_type=NodeType.CONCEPT, label="Q-Former"),
    ]
    for index, paper_id in enumerate(
        ["paper_6bc5d399d6127a64"] * 3 + ["paper_dc8bace378a282ea"], start=1
    ):
        nodes.append(GraphNode(node_id=f"chunk:{index}", node_type=NodeType.CHUNK, label=f"chunk {index}", properties={"paper_id": paper_id}))
    edges = [
        GraphEdge(edge_id=f"mention:{index}", source_id=f"chunk:{index}", target_id="concept:q-former", edge_type=EdgeType.MENTIONS)
        for index in range(1, 5)
    ]
    return GraphDocument(nodes=nodes, edges=edges)


def test_context_guard_seed_golden_scores_perfectly():
    cases = TypeAdapter(list[ContextGuardCase]).validate_json(
        GOLDEN_PATH.read_text(encoding="utf-8")
    )

    result = evaluate_context_guard(
        cases,
        guard=ContextGuard(GraphEntityResolver(build_entity_graph())),
    )

    assert result.case_count == 7
    assert result.accuracy == 1.0
    assert result.constraint_recall == 1.0
    assert result.clarification_recall == 1.0
    assert result.wrong_constraint_rate == 0.0


def test_evaluate_context_guard_script_writes_json(tmp_path):
    output = tmp_path / "context_guard_eval.json"
    graph_path = tmp_path / "graph"
    write_graph_jsonl(build_entity_graph(), graph_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--golden",
            str(GOLDEN_PATH),
            "--output",
            str(output),
            "--graph",
            str(graph_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "Accuracy: 1.0000" in completed.stdout
    assert payload["accuracy"] == 1.0
    assert payload["wrong_constraint_rate"] == 0.0
