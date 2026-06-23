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

        traceback_info, ignored_paths = self._parse_traceback(combined_log, work_dir)

        record = AutoDebugRecord(
            experiment_id=experiment_id,
            result_id=failed_result.get("result_id", ""),
            patch_id=patch_id,
            attempt_number=attempt,
            error_summary=error_text[:500] if error_text else "no error message",
            fix_description=f"traceback parsed: {traceback_info}" if traceback_info else "no traceback found; manual review needed",
            fix_file_contents={},
        )
        state.values["ignored_traceback_paths"] = ignored_paths
        return self._persist(record, state, context, experiment_id)

    def _parse_traceback(self, text: str, work_dir: Path) -> tuple[dict[str, int] | None, list[str]]:
        matches = _TRACEBACK_PATTERN.findall(text)
        if not matches:
            return None, []
        result: dict[str, int] = {}
        ignored: list[str] = []
        work_resolved = work_dir.resolve()
        for filepath, line_num in matches:
            try:
                resolved = Path(filepath).resolve()
                rel = str(resolved.relative_to(work_resolved))
                result[rel] = int(line_num)
            except ValueError:
                ignored.append(filepath)
        return (result if result else None, ignored)

    _MAX_FULL_FILE_LINES = 800
    _TRUNCATION_CONTEXT_LINES = 100

    def _read_file_contexts(self, candidates: list[str], work_dir: Path, traceback_lines: dict[str, int] | None = None) -> tuple[dict[str, str], set[str]]:
        """Read candidate files. Returns (contexts, read_only_paths).
        Files > _MAX_FULL_FILE_LINES lines are truncated around error lines
        and marked as read_only_context.
        """
        contexts: dict[str, str] = {}
        read_only: set[str] = set()
        for rel_path in candidates:
            target = work_dir / rel_path
            if not target.exists() or not target.is_file():
                continue
            lines = target.read_text(encoding="utf-8").splitlines()
            if len(lines) <= self._MAX_FULL_FILE_LINES:
                contexts[rel_path] = "\n".join(lines)
            else:
                read_only.add(rel_path)
                err_line = (traceback_lines or {}).get(rel_path, 1)
                start = max(0, err_line - self._TRUNCATION_CONTEXT_LINES - 1)
                end = min(len(lines), err_line + self._TRUNCATION_CONTEXT_LINES)
                snippet = lines[start:end]
                contexts[rel_path] = (
                    f"[... {start} lines truncated ...]\n"
                    + "\n".join(snippet)
                    + f"\n[... {len(lines) - end} lines truncated ...]"
                )
        return contexts, read_only

    def _build_debug_prompt(
        self, experiment_id: str, attempt: int,
        plan: dict, failed: dict, patch: dict,
        contexts: dict[str, str],
    ) -> list[dict[str, str]]:
        topic_keywords = ", ".join(
            plan.get("files_to_change", [])[:10]
        )
        context_text = "\n\n".join(
            f"--- {path} ---\n{content}"
            for path, content in contexts.items()
        )
        return [
            {"role": "system", "content": (
                "You are debugging a failed experiment. Your job is to analyze the error "
                "and propose a minimal, safe fix that preserves the experiment's intent. "
                "Return exactly one JSON object with 'fix_description' (string) and "
                "'fix_file_contents' (dict mapping relative path to complete corrected file content). "
                "Only return files among the provided file contexts. "
                "If a file is marked as read_only_context in the prompt, do NOT include it "
                "in fix_file_contents — set its value to null instead. "
                "Keep fixes minimal: change only what's needed to resolve the error."
            )},
            {"role": "user", "content": (
                f"Experiment: {experiment_id} (attempt {attempt})\n"
                f"Hypothesis: {plan.get('hypothesis', 'unknown')}\n"
                f"Modification: {plan.get('modification', 'unknown')}\n"
                f"Files to change: {', '.join(plan.get('files_to_change', []))}\n"
                f"Keywords: {topic_keywords}\n\n"
                f"Error: {failed.get('error_message', '')}\n"
                f"Status: {failed.get('status', '')}\n"
                f"Command: {failed.get('run_command', '')}\n"
                f"Log tail:\n{failed.get('log_tail', '')}\n\n"
                f"File contexts:\n{context_text}"
            )},
        ]

    def _new_llm_call_id(self) -> str:
        import uuid
        return f"llm_call_{uuid.uuid4().hex[:12]}"

    def _write_llm_call_artifact(self, state, artifact_store, call_data: dict):
        call_id = self._new_llm_call_id()
        artifact_store.save_json(state.run_id, "llm_calls", call_id, call_data)

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
