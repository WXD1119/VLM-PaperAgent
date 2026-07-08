from paper_agent.domain.answer import (
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    SemanticCitationReport,
)
from paper_agent.llm.client import LLMClient
from paper_agent.retrieval.reranker import RerankedHit


SYSTEM_PROMPT = """You are an evidence-grounded paper reading assistant.
Answer only from the supplied evidence. Every independently verifiable claim must cite one or
more evidence IDs. If the evidence is insufficient, set abstained=true and explain why. Never
invent a citation. Return JSON matching the supplied schema."""


def build_evidence_pack(query: str, hits: list[RerankedHit]) -> EvidencePack:
    if not hits:
        raise ValueError("cannot build an evidence pack without retrieval hits")
    return EvidencePack(
        query=query,
        items=[
            EvidenceItem(
                evidence_id=f"E{rank}",
                chunk_id=hit.chunk_id,
                paper_id=hit.paper_id,
                kind=hit.kind,
                pages=hit.pages,
                section_path=hit.section_path,
                content=hit.content,
            )
            for rank, hit in enumerate(hits, start=1)
        ],
    )


def render_evidence_prompt(pack: EvidencePack) -> str:
    blocks = [f"Question: {pack.query}"]
    for item in pack.items:
        section = " > ".join(item.section_path) or "(unknown section)"
        blocks.append(
            f"[{item.evidence_id}] paper={item.paper_id}; pages={item.pages}; "
            f"kind={item.kind.value}; section={section}\n{item.content}"
        )
    return "\n\n".join(blocks)


class CitationValidator:
    def validate(self, answer: GroundedAnswer, pack: EvidencePack) -> CitationValidation:
        allowed = {item.evidence_id for item in pack.items}
        errors: list[str] = []
        cited = 0
        for index, claim in enumerate(answer.claims, start=1):
            unknown = sorted(set(claim.evidence_ids) - allowed)
            if unknown:
                errors.append(f"claim {index} cites unknown evidence: {', '.join(unknown)}")
            else:
                cited += 1
        if answer.abstained and answer.claims:
            errors.append("abstained answer must not contain factual claims")
        return CitationValidation(
            valid=not errors,
            claim_count=len(answer.claims),
            cited_claim_count=cited,
            errors=errors,
        )


class AnswerAgent:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.validator = CitationValidator()

    def answer(self, pack: EvidencePack) -> tuple[GroundedAnswer, CitationValidation]:
        prompt = f"{SYSTEM_PROMPT}\n\n{render_evidence_prompt(pack)}"
        answer = self.client.generate_structured(prompt, GroundedAnswer)
        report = self.validator.validate(answer, pack)
        if not report.valid:
            raise ValueError("citation validation failed: " + "; ".join(report.errors))
        return answer, report


class SemanticCitationJudge:
    """Judge whether each claim is entailed by its cited evidence in one batched LLM call."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def evaluate(
        self, answer: GroundedAnswer, pack: EvidencePack
    ) -> SemanticCitationReport:
        if answer.abstained:
            return SemanticCitationReport()
        evidence = {item.evidence_id: item for item in pack.items}
        blocks = [
            "Evaluate whether each claim is supported by its cited evidence only. "
            "Use supported only when all material details are directly supported; use "
            "partially_supported for overstatement, and unsupported for contradiction or absence."
        ]
        for index, claim in enumerate(answer.claims, start=1):
            blocks.append(f"Claim {index}: {claim.text}")
            for evidence_id in claim.evidence_ids:
                item = evidence[evidence_id]
                blocks.append(f"[{evidence_id}] {item.content}")
        report = self.client.generate_structured("\n\n".join(blocks), SemanticCitationReport)
        expected = set(range(1, len(answer.claims) + 1))
        actual = {item.claim_index for item in report.assessments}
        if actual != expected or len(report.assessments) != len(answer.claims):
            raise ValueError("semantic judge must assess every claim exactly once")
        allowed = set(evidence)
        for assessment in report.assessments:
            if not set(assessment.evidence_ids).issubset(allowed):
                raise ValueError("semantic judge returned an unknown evidence ID")
        return report
