from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from agents.code_writer import CodeWriterAgent
from core.agent_base import AgentContext
from core.artifact_store import ArtifactStore
from core.state import ResearchState
from schemas.topic_pack import TopicPack


def _state() -> ResearchState:
    topic = TopicPack(topic_name="test", codebase={"repo_path": "/fake/repo", "allowed_auto_edit": ["model/"], "protected_files": ["model/secrets.py"]})
    state = ResearchState(topic=topic)
    state.values["experiment_plans"] = [{"experiment_id": "exp_1", "hypothesis": "test", "modification": "change decoder", "files_to_change": ["model/decoder.py"]}]
    state.values["code_tasks"] = [{"task_id": "ct_1", "experiment_id": "exp_1", "allowed_paths": ["model/"], "protected_paths": ["model/secrets.py"]}]
    return state


class CodeWriterAgentTest(TestCase):
    def test_skips_when_code_writes_disabled(self):
        with TemporaryDirectory() as tmp:
            state = _state()
            context = AgentContext(
                artifact_store=ArtifactStore(Path(tmp)),
                memory_store=None, tool_registry=None,
                settings={"enable_code_writes": False},
            )
            agent = CodeWriterAgent()
            result = agent.run(state, context)
            self.assertIn("code_patches_by_experiment_id", state.values)
            patch = state.values["code_patches_by_experiment_id"].get("exp_1")
            self.assertEqual(patch["status"], "skipped")
            self.assertIn("code writes disabled", patch["reason"])


if __name__ == "__main__":
    main()
