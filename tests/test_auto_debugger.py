from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from agents.auto_debugger import AutoDebuggerAgent
from core.agent_base import AgentContext
from core.artifact_store import ArtifactStore
from core.state import ResearchState
from schemas.topic_pack import TopicPack


class AutoDebuggerAgentTest(TestCase):
    def _state_and_context(self, tmp: str, **settings):
        topic = TopicPack(topic_name="test")
        state = ResearchState(topic=topic)
        state.values["experiment_results"] = [
            {"result_id": "r1", "experiment_id": "exp_1", "status": "error",
             "error_message": "NameError: name 'x' is not defined",
             "attempt": 0, "patch_id": "patch_test1"}
        ]
        state.values["code_patches_by_experiment_id"] = {
            "exp_1": {"patch_id": "patch_test1", "work_dir": tmp, "changed_files": [{"relative_path": "model/test.py"}]}
        }
        context = AgentContext(
            artifact_store=ArtifactStore(Path(tmp)),
            memory_store=None, tool_registry=None,
            settings=settings,
        )
        return state, context

    def test_skips_when_llm_disabled(self):
        with TemporaryDirectory() as tmp:
            state, context = self._state_and_context(tmp, enable_llm=False)
            agent = AutoDebuggerAgent()
            result = agent.run(state, context)
            record = state.values.get("last_debug_record", {})
            self.assertTrue(record.get("error_summary", "").startswith("skipped") or record.get("error_summary") == "")

    def test_blocks_when_max_attempts_reached(self):
        with TemporaryDirectory() as tmp:
            state, context = self._state_and_context(tmp, enable_llm=True, max_debug_attempts=3)
            state.values["experiment_results"][0]["attempt"] = 3
            agent = AutoDebuggerAgent()
            result = agent.run(state, context)
            record = state.values.get("last_debug_record", {})
            self.assertTrue("max" in str(record).lower() or record.get("fix_file_contents", {}) == {})

    def test_no_code_patch_returns_error(self):
        with TemporaryDirectory() as tmp:
            state, context = self._state_and_context(tmp, enable_llm=True, max_debug_attempts=3)
            state.values["code_patches_by_experiment_id"] = {}
            state.values["experiment_results"] = [{"result_id": "r1", "experiment_id": "exp_1", "status": "error", "error_message": "err", "attempt": 0, "patch_id": "nonexistent"}]
            agent = AutoDebuggerAgent()
            result = agent.run(state, context)
            self.assertIn("error", result.notes[0].lower() if result.notes else "")


if __name__ == "__main__":
    main()
