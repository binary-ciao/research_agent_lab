from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import unittest

from core.artifact_store import ArtifactStore
from core.run_logger import RunLogger
from memory.sqlite_memory import SQLiteMemoryStore
from schemas.topic_pack import TopicPack, load_topic_pack
from tools.tool_registry import build_default_tool_registry
from workflows.factory import build_full_research_workflow


class FullResearchLoopTest(unittest.TestCase):
    def test_offline_workflow_produces_reviewable_artifacts(self) -> None:
        topic = load_topic_pack(Path("topics/pedestrian_diffusion.json"))
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workflow = build_full_research_workflow(
                artifact_store=ArtifactStore(tmp_path / "runs"),
                memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite3"),
                tool_registry=build_default_tool_registry(),
                logger=RunLogger(),
                max_papers=3,
            )
            state = workflow.run(topic)

            self.assertEqual(state.stage, "completed")
            self.assertEqual(state.values["paper_count"], 3)
            self.assertEqual(state.values["review_status"], "needs_human_review")
            run_dir = tmp_path / "runs" / state.run_id
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "artifacts" / "reports" / "synthesis_report.md").exists())
            self.assertEqual(len(list((run_dir / "artifacts" / "method_cards").glob("*.json"))), 3)
            self.assertIn("run_evaluation_status", state.values)
            self.assertIn("run_quality_score", state.values)
            self.assertTrue((run_dir / "artifacts" / "run_evaluations").exists())
            self.assertGreaterEqual(
                len(list((run_dir / "artifacts" / "run_evaluations").glob("*.json"))),
                1,
            )


class CliParserTest(unittest.TestCase):
    def test_reference_expansion_flags_parse(self):
        from app.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--topic", "topics/intent_led_virat.json",
            "--enable-reference-expansion",
            "--max-reference-seeds", "3",
        ])

        self.assertTrue(args.enable_reference_expansion)
        self.assertEqual(args.max_reference_seeds, 3)

    def test_train_budget_epochs_flag_parses(self):
        from app.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--topic", "topics/intent_led_virat.json",
            "--train-budget-epochs", "5",
        ])

        self.assertEqual(args.train_budget_epochs, 5)

    def test_train_budget_minutes_flag_parses(self):
        from app.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--topic", "topics/intent_led_virat.json",
            "--train-budget-minutes", "30",
        ])

        self.assertEqual(args.train_budget_minutes, 30)


class RetrievalEvaluationCliParserTest(unittest.TestCase):
    def test_retrieval_evaluation_flags_parse(self):
        from app.main import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--topic", "topics/intent_led_virat.json",
            "--enable-retrieval-evaluation",
            "--enable-retrieval-judge",
            "--retrieval-judge-top-k", "3",
        ])

        self.assertTrue(args.enable_retrieval_evaluation)
        self.assertTrue(args.enable_retrieval_judge)
        self.assertEqual(args.retrieval_judge_top_k, 3)


class RetrievalEvaluationWorkflowTest(unittest.TestCase):
    def test_offline_workflow_can_write_retrieval_evaluation(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = ArtifactStore(tmp_path / "runs")
            topic = TopicPack(
                topic_name="workflow_retrieval_eval",
                search_seeds={"keywords": ["trajectory prediction", "diffusion"]},
            )
            workflow = build_full_research_workflow(
                artifact_store=store,
                memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite3"),
                tool_registry=build_default_tool_registry(),
                logger=RunLogger(),
                max_papers=2,
                enable_retrieval_evaluation=True,
            )

            state = workflow.run(topic)
            run_dir = store.run_dir(state.run_id)

            self.assertIn("retrieval_evaluation", state.values)
            self.assertTrue((run_dir / "artifacts" / "retrieval_evaluations").exists())
            self.assertGreaterEqual(
                len(list((run_dir / "artifacts" / "retrieval_evaluations").glob("*.json"))),
                1,
            )


class RunValidationWorkflowIntegrationTest(TestCase):
    def test_run_validation_agent_runs_after_run_evaluator(self):
        from core.artifact_store import ArtifactStore
        from core.run_logger import RunLogger
        from tools.tool_registry import build_default_tool_registry
        from workflows.factory import build_full_research_workflow

        with TemporaryDirectory() as tmp:
            workflow = build_full_research_workflow(
                artifact_store=ArtifactStore(Path(tmp) / "runs"),
                memory_store=None,
                tool_registry=build_default_tool_registry(),
                logger=RunLogger(),
            )
            names = [agent.name for agent in workflow.agents]
            self.assertIn("run_evaluator", names)
            self.assertIn("run_validation_agent", names)
            self.assertLess(names.index("run_evaluator"), names.index("run_validation_agent"))


if __name__ == "__main__":
    unittest.main()
