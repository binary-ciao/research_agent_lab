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

    def test_rejects_absolute_path(self):
        agent = CodeWriterAgent()
        ok, reason = agent._validate_paths(
            ["/etc/passwd"], Path("/tmp/work"), ["model/"], []
        )
        self.assertFalse(ok)
        self.assertIn("absolute", reason.lower())

    def test_rejects_parent_traversal(self):
        agent = CodeWriterAgent()
        ok, reason = agent._validate_paths(
            ["../outside.py"], Path("/tmp/work"), ["model/"], []
        )
        self.assertFalse(ok)
        self.assertIn("..", reason)

    def test_rejects_path_outside_work_dir(self):
        agent = CodeWriterAgent()
        ok, reason = agent._validate_paths(
            ["model/../../etc/hacked"], Path("/tmp/work/sub/proj"), ["model/"], []
        )
        self.assertFalse(ok)

    def test_rejects_protected_file(self):
        agent = CodeWriterAgent()
        ok, reason = agent._validate_paths(
            ["model/secrets.py"], Path("/tmp/work"), ["model/"], ["model/secrets.py"]
        )
        self.assertFalse(ok)
        self.assertIn("protected", reason.lower())

    def test_accepts_valid_path_in_allowed(self):
        agent = CodeWriterAgent()
        ok, reason = agent._validate_paths(
            ["model/decoder.py"], Path("/tmp/work"), ["model/"], ["model/secrets.py"]
        )
        self.assertTrue(ok)

    def test_copy_mode_creates_work_dir(self):
        import shutil
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "src"
            src_dir.mkdir()
            (src_dir / "model").mkdir()
            (src_dir / "model" / "test.py").write_text("print('hello')")
            (src_dir / ".git").mkdir()

            state = _state()
            state.topic.codebase["repo_path"] = str(src_dir)
            context = AgentContext(
                artifact_store=ArtifactStore(Path(tmp) / "runs"),
                memory_store=None, tool_registry=None,
                settings={"enable_code_writes": True, "enable_llm": False},
            )
            state.values["experiment_plans"] = [{"experiment_id": "exp_1", "hypothesis": "test", "modification": "change"}]
            state.values["code_tasks"] = [{"task_id": "ct_1", "experiment_id": "exp_1", "allowed_paths": ["model/test.py"], "protected_paths": []}]

            agent = CodeWriterAgent()
            agent.run(state, context)

            patch = state.values["code_patches_by_experiment_id"]["exp_1"]
            self.assertIn("code_copies", patch["work_dir"])
            self.assertEqual(patch["mode"], "copy")
            self.assertTrue(Path(patch["work_dir"]).exists())
            self.assertTrue((Path(patch["work_dir"]) / "model" / "test.py").exists())
            self.assertFalse((Path(patch["work_dir"]) / ".git").exists(),
                             ".git should not be copied")

    def test_policy_blocks_protected_file(self):
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "src"
            src_dir.mkdir()
            (src_dir / "model").mkdir(parents=True)
            (src_dir / "model" / "secrets.py").write_text("SECRET = 'xyz'")
            (src_dir / "model" / "decoder.py").write_text("x = 1")
            state = _state()
            state.topic.codebase["repo_path"] = str(src_dir)
            state.topic.codebase["copy_can_modify"] = True  # sandbox mode
            state.topic.codebase["protected_files"] = ["model/secrets.py"]
            state.topic.codebase["allowed_auto_edit"] = ["model/"]
            state.values["experiment_plans"] = [{
                "experiment_id": "exp_1", "hypothesis": "test",
                "modification": "change secrets",
                "files_to_change": ["model/secrets.py"],
            }]
            context = AgentContext(
                artifact_store=ArtifactStore(Path(tmp) / "runs"),
                memory_store=None, tool_registry=None,
                settings={"enable_code_writes": True},
            )
            agent = CodeWriterAgent()
            result = agent.run(state, context)
            patch = state.values["code_patches_by_experiment_id"]["exp_1"]
            self.assertEqual(patch["status"], "blocked")
            self.assertIn("protected", patch["reason"])

    def test_policy_blocks_max_files(self):
        with TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "src"
            src_dir.mkdir()
            (src_dir / "model").mkdir(parents=True)
            for i in range(10):
                (src_dir / "model" / f"file_{i}.py").write_text(f"# file {i}")
            state = _state()
            state.topic.codebase["repo_path"] = str(src_dir)
            state.topic.codebase["copy_can_modify"] = True
            state.topic.codebase["max_files_per_patch"] = 3
            state.topic.codebase["allowed_auto_edit"] = ["model/"]
            state.values["experiment_plans"] = [{
                "experiment_id": "exp_1", "hypothesis": "test",
                "modification": "change many files",
                "files_to_change": [f"model/file_{i}.py" for i in range(10)],
            }]
            context = AgentContext(
                artifact_store=ArtifactStore(Path(tmp) / "runs"),
                memory_store=None, tool_registry=None,
                settings={"enable_code_writes": True},
            )
            agent = CodeWriterAgent()
            result = agent.run(state, context)
            patch = state.values["code_patches_by_experiment_id"]["exp_1"]
            self.assertEqual(patch["status"], "blocked")
            self.assertIn("files", patch["reason"].lower())


if __name__ == "__main__":
    main()
