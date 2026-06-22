from __future__ import annotations
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.agent_base import Agent, AgentContext, AgentResult
from core.state import ResearchState
from schemas.auto_debug_record import AutoDebugRecord


_TRACEBACK_PATTERN = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+)', re.IGNORECASE
)


class AutoDebuggerAgent(Agent):
    name = "auto_debugger"

    def run(self, state: ResearchState, context: AgentContext) -> AgentResult:
        results = state.values.get("experiment_results") or []
        failed = [r for r in results if isinstance(r, dict) and r.get("status") in ("error", "failed")]
        if not failed:
            return AgentResult(notes=["auto_debugger: no failed results to debug"])

        failed_result = failed[-1]
        experiment_id = failed_result.get("experiment_id", "unknown")
        attempt = failed_result.get("attempt", 0)
        patch_id = failed_result.get("patch_id", "")

        max_attempts = int(context.settings.get("max_debug_attempts", 3))
        if attempt >= max_attempts:
            record = AutoDebugRecord(
                experiment_id=experiment_id, result_id=failed_result.get("result_id", ""),
                patch_id=patch_id, attempt_number=attempt,
                error_summary="max debug attempts reached",
                fix_description="",
            )
            return self._persist(record, state, context, experiment_id)

        enable_llm = bool(context.settings.get("enable_llm"))
        if not enable_llm:
            record = AutoDebugRecord(
                experiment_id=experiment_id, result_id=failed_result.get("result_id", ""),
                patch_id=patch_id, attempt_number=attempt,
                error_summary="skipped: LLM disabled",
                fix_description="auto-debug requires --enable-llm",
            )
            return self._persist(record, state, context, experiment_id)

        patches = state.values.get("code_patches_by_experiment_id", {})
        code_patch = patches.get(experiment_id, {})
        if not code_patch:
            record = AutoDebugRecord(
                experiment_id=experiment_id, result_id=failed_result.get("result_id", ""),
                patch_id=patch_id, attempt_number=attempt,
                error_summary="error: no CodePatch for experiment",
                fix_description="",
            )
            return self._persist(record, state, context, experiment_id)

        work_dir = Path(code_patch.get("work_dir", ""))
        error_text = failed_result.get("error_message", "")
        log_tail = failed_result.get("log_tail", "")
        combined_log = f"{error_text}\n{log_tail}"

        traceback_info = self._parse_traceback(combined_log, work_dir)

        record = AutoDebugRecord(
            experiment_id=experiment_id,
            result_id=failed_result.get("result_id", ""),
            patch_id=patch_id,
            attempt_number=attempt,
            error_summary=error_text[:500] if error_text else "no error message",
            fix_description=f"traceback parsed: {traceback_info}" if traceback_info else "no traceback found; manual review needed",
            fix_file_contents={},
        )
        return self._persist(record, state, context, experiment_id)

    def _parse_traceback(self, text: str, work_dir: Path) -> dict[str, int] | None:
        matches = _TRACEBACK_PATTERN.findall(text)
        if not matches:
            return None
        result: dict[str, int] = {}
        for filepath, line_num in matches:
            try:
                rel = str(Path(filepath).resolve().relative_to(work_dir.resolve()))
            except ValueError:
                rel = filepath
            result[rel] = int(line_num)
        return result if result else None

    def _persist(self, record: AutoDebugRecord, state: ResearchState, context: AgentContext, experiment_id: str) -> AgentResult:
        record_dict = asdict(record)
        records = state.values.setdefault("last_debug_records_by_experiment_id", {})
        records[experiment_id] = record_dict
        state.values["last_debug_record"] = record_dict
        context.artifact_store.save_json(state.run_id, "auto_debug_records", record.record_id, record_dict)
        note = f"auto_debugger: experiment={experiment_id} attempt={record.attempt_number}"
        if record.error_summary:
            note = f"auto_debugger: {record.error_summary}"
        return AgentResult(
            notes=[note],
            artifacts={"auto_debug_records": [record.record_id]},
            values={
                "last_debug_record": record_dict,
                "last_debug_records_by_experiment_id": records,
            },
        )
