from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.agent_base import Agent, AgentContext, AgentResult
from core.state import ResearchState
from schemas.code_patch import CodePatch


class CodeWriterAgent(Agent):
    name = "code_writer"

    def run(self, state: ResearchState, context: AgentContext) -> AgentResult:
        plans = state.values.get("experiment_plans", []) or []
        if not plans:
            return AgentResult(notes=["code_writer: no experiment plans"])

        plan = plans[0] if isinstance(plans[0], dict) else {}
        experiment_id = plan.get("experiment_id", "unknown")
        enable_code_writes = bool(context.settings.get("enable_code_writes"))

        if not enable_code_writes:
            patch = CodePatch(
                experiment_id=experiment_id,
                status="skipped",
                reason="code writes disabled; set --enable-code-writes",
            )
            return self._persist(patch, state, context, experiment_id)

        return AgentResult(notes=["code_writer: not implemented yet"])

    def _validate_paths(self, relative_paths: list[str], work_dir: Path, allowed_paths: list[str], protected_paths: list[str]) -> tuple[bool, str]:
        for rel in relative_paths:
            if not rel or rel != rel.strip():
                return False, f"empty or whitespace-padded path: {rel!r}"
            if Path(rel).is_absolute() or rel.startswith("/") or (len(rel) >= 3 and rel[1] == ":"):
                return False, f"absolute or drive-letter path rejected: {rel}"
            if ".." in Path(rel).parts:
                return False, f"parent traversal rejected: {rel}"
            resolved = (work_dir / rel).resolve()
            work_resolved = work_dir.resolve()
            try:
                resolved.relative_to(work_resolved)
            except ValueError:
                return False, f"resolved path {resolved} is outside work_dir {work_resolved}"
            for protected in protected_paths:
                if resolved == work_resolved / protected:
                    return False, f"protected file: {rel}"
            in_allowed = False
            if not allowed_paths:
                in_allowed = True
            else:
                for a in allowed_paths:
                    try:
                        resolved.relative_to(work_resolved / a)
                        in_allowed = True
                        break
                    except ValueError:
                        continue
            if not in_allowed:
                return False, f"path {rel} not in allowed_paths"
        return True, ""

    def _persist(self, patch: CodePatch, state: ResearchState, context: AgentContext, experiment_id: str) -> AgentResult:
        patch_dict = asdict(patch)
        patches = state.values.setdefault("code_patches_by_experiment_id", {})
        patches[experiment_id] = patch_dict
        state.values["code_patch"] = patch_dict
        context.artifact_store.save_json(state.run_id, "code_patches", patch.patch_id, patch_dict)
        return AgentResult(
            notes=[f"code_writer: status={patch.status} reason={patch.reason}"],
            artifacts={"code_patches": [patch.patch_id]},
            values={
                "code_patch": patch_dict,
                "code_patches_by_experiment_id": patches,
            },
        )
