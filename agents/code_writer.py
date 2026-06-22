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
