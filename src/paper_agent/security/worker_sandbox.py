"""Declarative safety contract for PDF/OCR parser workers.

The contract is validated before a container/job runner is invoked. It is
usable locally for tests and maps directly to Kubernetes/Docker restrictions;
it intentionally does not claim to sandbox a Windows subprocess by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class NetworkMode(StrEnum):
    DENY = "deny"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True)
class WorkerResourceLimits:
    cpu_cores: float = 2.0
    memory_mb: int = 4_096
    pids: int = 128
    timeout_seconds: int = 1_800
    max_input_bytes: int = 100 * 1024 * 1024
    max_pages: int = 300

    def validate(self) -> None:
        if not 0 < self.cpu_cores <= 16:
            raise ValueError("cpu_cores must be in (0, 16]")
        if not 256 <= self.memory_mb <= 65_536:
            raise ValueError("memory_mb must be in [256, 65536]")
        if not 16 <= self.pids <= 4_096:
            raise ValueError("pids must be in [16, 4096]")
        if not 1 <= self.timeout_seconds <= 7_200:
            raise ValueError("timeout_seconds must be in [1, 7200]")
        if not 1 <= self.max_input_bytes <= 1024 * 1024 * 1024:
            raise ValueError("max_input_bytes must be in [1, 1073741824]")
        if not 1 <= self.max_pages <= 5_000:
            raise ValueError("max_pages must be in [1, 5000]")


@dataclass(frozen=True)
class ParserWorkerSandbox:
    """Required properties of an untrusted-document parsing runtime."""

    input_root: Path
    output_root: Path
    scratch_root: Path
    network_mode: NetworkMode = NetworkMode.DENY
    network_allowlist: tuple[str, ...] = ()
    read_only_root_filesystem: bool = True
    run_as_non_root: bool = True
    drop_all_capabilities: bool = True
    privileged: bool = False
    limits: WorkerResourceLimits = WorkerResourceLimits()

    def validate(self) -> None:
        self.limits.validate()
        roots = (self.input_root.resolve(), self.output_root.resolve(), self.scratch_root.resolve())
        if len(set(roots)) != len(roots):
            raise ValueError("input, output and scratch roots must be separate")
        if self.network_mode == NetworkMode.DENY and self.network_allowlist:
            raise ValueError("network deny mode cannot define an allowlist")
        if self.network_mode == NetworkMode.ALLOWLIST and not self.network_allowlist:
            raise ValueError("allowlist network mode requires explicit destinations")
        if not self.read_only_root_filesystem or not self.run_as_non_root or not self.drop_all_capabilities:
            raise ValueError("parser worker must use read-only root, non-root user and dropped capabilities")
        if self.privileged:
            raise ValueError("parser worker must never run privileged")

    def validate_job_paths(self, *, source: Path, output: Path) -> None:
        self.validate()
        if not _within(source.resolve(), self.input_root.resolve()):
            raise PermissionError("parser source must be staged below sandbox input_root")
        if not _within(output.resolve(), self.output_root.resolve()):
            raise PermissionError("parser output must be below sandbox output_root")


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True
