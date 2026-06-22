from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import re
import shlex
import time
from typing import Any

from core.agent_base import Agent, AgentContext, AgentResult
from core.state import ResearchState
from agents.result_parser import parse_experiment_output
from schemas.experiment_result import ExperimentResult
from tools.code_executor import ScopedCodeExecutor


_PYTHON_EXE = os.environ.get(
    "RESEARCH_AGENT_PYTHON",
    "D:/Develop_Tools/Anaconda3/envs/video_llava/python.exe",
)

_SMOKE_TIMEOUT = int(os.environ.get("RESEARCH_AGENT_SMOKE_TIMEOUT", "600"))


class AutonomousExperimentAgent(Agent):
    name = "autonomous_experiment"

    def run(self, state: ResearchState, context: AgentContext) -> AgentResult:
        plans = state.values.get("experiment_plans", []) or []
        if not plans:
            state.values["experiment_results"] = []
            return AgentResult(
                notes=["skipped: no experiment plans"],
                values={"experiment_results": []},
            )

        codebase = state.topic.codebase
        repo_path = codebase.get("repo_path", "")

        if not context.settings.get("enable_experiments"):
            state.values["experiment_results"] = []
            return AgentResult(
                notes=["skipped: --enable-experiments not set"],
                values={"experiment_results": []},
            )

        if not repo_path or not Path(repo_path).exists():
            state.values["experiment_results"] = []
            return AgentResult(
                notes=[f"skipped: repo_path not found: {repo_path}"],
                values={"experiment_results": []},
            )

        all_results: list[dict[str, Any]] = []
        for plan in plans:
            experiment_id = plan.get("experiment_id", "unknown")
            smoke_commands = self._smoke_commands(state)
            for cmd in smoke_commands:
                result = self._execute_and_parse(experiment_id, cmd, repo_path, state)
                context.artifact_store.save_json(
                    state.run_id, "experiment_results", result.result_id, result
                )
                all_results.append(asdict(result))
                # Stop on error within the plan
                if result.status == "error":
                    break
            # Stop processing further plans after an error
            if all_results and all_results[-1].get("status") == "error":
                break

        state.values["experiment_results"] = all_results
        summary = self._summarize(all_results)
        return AgentResult(
            notes=summary,
            artifacts={"experiment_results": [r["result_id"] for r in all_results]},
            values={"experiment_results": all_results},
        )

    def _smoke_commands(self, state: ResearchState) -> list[str]:
        report = state.values.get("codebase_report", {})
        commands = report.get("smoke_commands", [])
        if not commands:
            commands = ["python -c \"print('no smoke commands configured; nothing to run')\""]
        return [_rewrite_python(cmd) for cmd in commands]

    def _execute_and_parse(
        self,
        experiment_id: str,
        command: str,
        repo_path: str,
        state: ResearchState,
    ) -> ExperimentResult:
        cwd, clean_command = _normalize_command(command, repo_path)
        executor = ScopedCodeExecutor(repo_path)
        cmd_parts = shlex.split(clean_command)
        start = time.monotonic()
        try:
            completed = executor.run(cmd_parts, cwd=cwd, timeout=_SMOKE_TIMEOUT)
            duration = round(time.monotonic() - start, 2)
            return parse_experiment_output(
                experiment_id=experiment_id,
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                command=command,
                expected_metrics=state.topic.experiment_metrics,
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = round(time.monotonic() - start, 2)
            return ExperimentResult(
                experiment_id=experiment_id,
                status="error",
                error_message=str(exc)[:500],
                run_command=command,
                duration_seconds=duration,
            )

    def _summarize(self, results: list[dict[str, Any]]) -> list[str]:
        notes: list[str] = []
        for r in results:
            status = r.get("status", "unknown")
            metrics = r.get("metrics", {})
            metric_str = ", ".join(
                f"{k}={v}" for k, v in sorted(metrics.items())
            ) if metrics else "no metrics"
            notes.append(f"experiment {status}: {metric_str}")
        return notes


_CD_PATTERN = re.compile(r"^cd\s+(?:/d\s+)?(\S+)\s*&&\s*(.+)$", re.IGNORECASE)


def _normalize_command(command: str, default_repo: str) -> tuple[str, str]:
    """Extract cd prefix into cwd, returning (cwd, clean_command)."""
    m = _CD_PATTERN.match(command.strip())
    if m:
        cwd = m.group(1)
        clean = m.group(2)
        return cwd, _rewrite_python(clean)
    return default_repo, _rewrite_python(command)


def _rewrite_python(command: str) -> str:
    parts = command.split()
    if parts and parts[0] == "python":
        parts[0] = _PYTHON_EXE
    return " ".join(parts)
