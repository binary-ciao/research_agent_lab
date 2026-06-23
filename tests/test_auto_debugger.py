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


    def test_large_file_is_read_only_context(self):
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            work = Path(tmp) / "code"
            work.mkdir()
            small = work / "small.py"
            small.write_text("line\n" * 100)
            large = work / "large.py"
            large.write_text("line\n" * 900)
            agent = AutoDebuggerAgent()
            candidates = ["small.py", "large.py"]
            contexts, read_only = agent._read_file_contexts(candidates, work)
            self.assertIn("small.py", contexts)
            self.assertGreater(len(contexts["small.py"]), 0)
            self.assertIn("large.py", read_only)
            # large.py content should be truncated, not full
            self.assertLess(len(contexts.get("large.py", "").splitlines()), 900)

    def test_reads_traceback_and_plan_contexts(self):
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            work = Path(tmp) / "code"
            work.mkdir()
            (work / "train.py").write_text("def train():\n    pass\n")
            (work / "model.py").write_text("class Model:\n    pass\n")
            agent = AutoDebuggerAgent()
            candidates = ["train.py", "model.py"]
            traceback_lines = {"train.py": 2}
            contexts, read_only = agent._read_file_contexts(candidates, work, traceback_lines)
            self.assertIn("train.py", contexts)
            self.assertIn("model.py", contexts)
            self.assertEqual(len(read_only), 0)

    def test_discards_traceback_outside_work_dir(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp) / "code"
            work.mkdir()
            (work / "train.py").write_text("x = 1")
            state, context = self._state_and_context(
                tmp, enable_llm=True, max_debug_attempts=2,
            )
            agent = AutoDebuggerAgent()
            train_py = work / "train.py"
            text = (
                'File "/etc/system.py", line 42, in run\n'
                f'File "{train_py}", line 10, in forward\n'
            )
            resolved, ignored = agent._parse_traceback(text, work)
            self.assertIsNotNone(resolved)
            self.assertIn("train.py", resolved)
            self.assertEqual(resolved["train.py"], 10)
            self.assertNotIn("/etc/system.py", resolved)
            self.assertNotIn("system.py", resolved)
            self.assertTrue(len(ignored) >= 1)
            self.assertTrue(any("system" in p for p in ignored))


    def test_build_debug_prompt_includes_all_sections(self):
        agent = AutoDebuggerAgent()
        plan = {"experiment_id": "exp_1", "hypothesis": "test hypothesis",
                "modification": "change x to y", "files_to_change": ["model.py"]}
        failed = {"result_id": "r1", "status": "error",
                  "error_message": "NameError: x not defined",
                  "log_tail": "Traceback...", "run_command": "python train.py"}
        patch = {"patch_id": "p1", "changed_files": [{"relative_path": "model.py"}]}
        contexts = {"model.py": "class Model:\n    x = 1\n"}
        read_only = set()
        messages = agent._build_debug_prompt("exp_1", 0, plan, failed, patch, contexts, read_only)
        user_content = messages[1]["content"]
        self.assertIn("test hypothesis", user_content)
        self.assertIn("change x to y", user_content)
        self.assertIn("NameError", user_content)
        self.assertIn("class Model:", user_content)
        self.assertIn("Patch: p1", user_content)
        self.assertIn("Changed files: model.py", user_content)

    def test_llm_call_artifact_written(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp))
            state, _ = self._state_and_context(tmp, enable_llm=True)
            agent = AutoDebuggerAgent()
            call_data = {
                "agent": "auto_debugger",
                "experiment_id": "exp_1",
                "result_id": "r1",
                "patch_id": "p1",
                "status": "skipped_call_budget",
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "route_enabled": True,
                "usage": {},
                "error": "",
            }
            agent._write_llm_call_artifact(state, store, call_data)
            paths = store.list_artifacts(state.run_id, "llm_calls")
            self.assertTrue(len(paths) >= 1)


if __name__ == "__main__":
    main()
