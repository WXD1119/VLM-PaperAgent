from paper_agent.domain.answer import (
    AnswerClaim,
    CitationValidation,
    ClaimSupportAssessment,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)
from paper_agent.llm.client import LLMClient
from paper_agent.retrieval.reranker import RerankedHit
from paper_agent.security import render_untrusted_evidence
from paper_agent.security.evidence import UNTRUSTED_EVIDENCE_NOTICE


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
    return render_untrusted_evidence(pack)


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

    def answer(
        self,
        pack: EvidencePack,
        feedback: str | None = None,
    ) -> tuple[GroundedAnswer, CitationValidation]:
        prompt = f"{SYSTEM_PROMPT}\n\n{render_evidence_prompt(pack)}"
        if feedback:
            prompt += (
                "\n\nRevision feedback from the citation judge:\n"
                f"{feedback}\n"
                "Revise the answer using only the supplied evidence. Remove or soften any claim "
                "that is not directly supported. If the evidence is insufficient, abstain."
            )
        answer = self.client.generate_structured(prompt, GroundedAnswer)
        report = self.validator.validate(answer, pack)
        if not report.valid:
            raise ValueError("citation validation failed: " + "; ".join(report.errors))
        return answer, report


class SemanticCitationJudge:
    """审查每条 Claim 是否由其引用证据直接蕴含。

    Judge 先尝试一次批量 LLM 调用以提升效率；如果模型漏项、重复、编号错误或
    返回格式错误，则降级为逐 Claim 调用。这能兼容 JSON 契约不稳定的本地推理模型。
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def evaluate(
        self, answer: GroundedAnswer, pack: EvidencePack
    ) -> SemanticCitationReport:
        if answer.abstained:
            return SemanticCitationReport()
        evidence = {item.evidence_id: item for item in pack.items}
        blocks = [
            (
                "Evaluate whether each claim is supported by its cited evidence only. "
                "Use supported only when all material details are directly supported; use "
                "partially_supported for overstatement, and unsupported for contradiction or absence."
            ),
            UNTRUSTED_EVIDENCE_NOTICE,
        ]
        for index, claim in enumerate(answer.claims, start=1):
            blocks.append(f"Claim {index}: {claim.text}")
            for evidence_id in claim.evidence_ids:
                item = evidence[evidence_id]
                blocks.append(f"[{evidence_id}] {item.content}")
        try:
            report = self.client.generate_structured("\n\n".join(blocks), SemanticCitationReport)
            if self._is_complete(report, len(answer.claims)):
                return self._validate_evidence_ids(report, set(evidence))
        except Exception:
            # 批量审查是优化而非强依赖。本地推理模型可能无法遵守批量 JSON 契约，
            # 但能处理更小的单 Claim 提示，因此这里降级而非让整条回答失败。
            pass

        assessments = [
            self._evaluate_single_claim(index, claim, evidence)
            for index, claim in enumerate(answer.claims, start=1)
        ]
        return self._validate_evidence_ids(
            SemanticCitationReport(assessments=assessments), set(evidence)
        )

    @staticmethod
    def _is_complete(report: SemanticCitationReport, claim_count: int) -> bool:
        expected = set(range(1, claim_count + 1))
        actual = {item.claim_index for item in report.assessments}
        return actual == expected and len(report.assessments) == claim_count

    @staticmethod
    def _validate_evidence_ids(
        report: SemanticCitationReport, allowed: set[str]
    ) -> SemanticCitationReport:
        for assessment in report.assessments:
            if not set(assessment.evidence_ids).issubset(allowed):
                raise ValueError("semantic judge returned an unknown evidence ID")
        return report

    def _evaluate_single_claim(
        self,
        index: int,
        claim: AnswerClaim,
        evidence: dict[str, EvidenceItem],
    ) -> ClaimSupportAssessment:
        blocks = [
            "Evaluate exactly one claim using only its cited evidence. "
            f"Return exactly one assessment with claim_index={index}. "
            "Use supported only when all material details are directly supported; use "
            "partially_supported for overstatement, and unsupported for contradiction or absence.",
            UNTRUSTED_EVIDENCE_NOTICE,
            f"Claim {index}: {claim.text}",
        ]
        for evidence_id in claim.evidence_ids:
            item = evidence[evidence_id]
            blocks.append(f"[{evidence_id}] {item.content}")

        assessment = self.client.generate_structured("\n\n".join(blocks), ClaimSupportAssessment)
        return assessment.model_copy(update={"claim_index": index})
