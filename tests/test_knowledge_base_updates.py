import pytest

from paper_agent.knowledge_base import LibraryUpdate, LibraryUpdatePhase, LibraryUpdateStatus, LocalKnowledgeBaseStore, RebuildJob, RebuildKind, RebuildQueue, RebuildStatus
from paper_agent.knowledge_base.coordinator import LibraryUpdateCoordinator


def test_update_coordinator_persists_phase_progress_and_resumes(tmp_path):
    store = LocalKnowledgeBaseStore(tmp_path)
    calls: list[str] = []
    failed_once = {"value": True}

    def action(phase):
        def run(_update):
            calls.append(phase.value)
            if phase == LibraryUpdatePhase.GRAPH and failed_once["value"]:
                failed_once["value"] = False
                raise RuntimeError("graph temporarily unavailable")
            return "commit-1" if phase == LibraryUpdatePhase.GRAPH else None
        return run

    coordinator = LibraryUpdateCoordinator(store, {phase: action(phase) for phase in LibraryUpdateCoordinator.PHASES})
    update = LibraryUpdate(paper_id="paper-a", content_sha256="sha", parser_version="mineru-1", embedding_model="bge")
    with pytest.raises(RuntimeError):
        coordinator.run(update)
    failed = store.load_update(update.update_id)
    assert failed.status == LibraryUpdateStatus.FAILED
    assert failed.completed_phases == [LibraryUpdatePhase.VALIDATE, LibraryUpdatePhase.INDEX]
    completed = coordinator.run(failed)
    assert completed.status == LibraryUpdateStatus.COMPLETED
    assert completed.graph_commit_id == "commit-1"
    assert calls.count("validate") == 1
    assert calls.count("index") == 1


def test_update_store_deduplicates_same_content_and_configuration(tmp_path):
    store = LocalKnowledgeBaseStore(tmp_path)
    first = store.create_or_load_update(LibraryUpdate(paper_id="paper-a", content_sha256="sha", parser_version="mineru-1", embedding_model="bge"))
    second = store.create_or_load_update(LibraryUpdate(paper_id="paper-a", content_sha256="sha", parser_version="mineru-1", embedding_model="bge"))
    assert first.update_id == second.update_id


def test_rebuild_queue_claims_and_persists_completion(tmp_path):
    store = LocalKnowledgeBaseStore(tmp_path)
    job = store.enqueue_rebuild(RebuildJob(paper_id="paper-a", kind=RebuildKind.REEMBED, target_embedding_model="bge-v2"))
    calls: list[str] = []
    queue = RebuildQueue(store, reparse=lambda _job: None, reembed=lambda item: calls.append(item.paper_id))
    completed = queue.run_next()
    assert completed is not None and completed.status == RebuildStatus.COMPLETED
    assert calls == ["paper-a"]
    assert queue.run_next() is None
