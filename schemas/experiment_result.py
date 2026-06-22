from __future__ import annotations

from dataclasses import dataclass, field

from schemas.base import new_id


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    result_id: str = field(default_factory=lambda: new_id("expresult"))
    status: str = "skipped"
    smoke_passed: bool = False
    eval_passed: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    log_tail: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: str = ""
    run_command: str = ""
    notes: list[str] = field(default_factory=list)
