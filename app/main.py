from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.artifact_store import ArtifactStore
from core.run_logger import RunLogger
from memory.literature_memory import LiteratureMemoryStore
from memory.memory_policy import memory_scope_for_topic
from memory.sqlite_memory import SQLiteMemoryStore
from schemas.topic_pack import load_topic_pack
from tools.agent_laboratory_adapter import AgentLaboratoryAdapter
from tools.arxiv_tool import ArxivTool
from tools.env_loader import load_env_file, mask_secret
from tools.local_paper_library import LocalPaperLibrary
from tools.model_router import ModelRouter
from tools.tool_registry import build_default_tool_registry
from workflows.factory import build_full_research_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a research workflow")
    run_parser.add_argument("--topic", required=True, help="Path to a topic pack JSON file")
    run_parser.add_argument("--data-dir", default="data", help="Runtime data directory")
    run_parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online tools such as arXiv when network access is available",
    )
    run_parser.add_argument(
        "--max-papers",
        type=int,
        default=8,
        help="Maximum papers or seed papers to carry through the first pass",
    )
    run_parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Allow agents with configured model routes to call external LLM APIs",
    )
    run_parser.add_argument(
        "--llm-call-budget",
        type=int,
        default=3,
        help="Maximum external LLM calls allowed in one run when --enable-llm is set",
    )
    run_parser.add_argument(
        "--llm-token-budget",
        type=int,
        default=20000,
        help="Approximate total-token budget for external LLM calls in one run",
    )
    run_parser.add_argument(
        "--enable-experiments",
        action="store_true",
        help="Allow AutonomousExperimentAgent to execute commands on the target codebase",
    )
    run_parser.add_argument(
        "--enable-tree-search",
        action="store_true",
        help="Allow TreeSearchAgent to generate branch experiment plans on failure",
    )
    run_parser.add_argument(
        "--promote",
        type=str,
        default=None,
        help="Manually promote a branch node (by node_id) to tree root",
    )
    run_parser.add_argument(
        "--max-parallel-branches",
        type=int,
        default=1,
        help="Maximum pending branches to select and execute in one run",
    )
    run_parser.add_argument(
        "--enable-reference-expansion",
        action="store_true",
        help="Use persisted extracted references as additional literature-search seeds",
    )
    run_parser.add_argument(
        "--max-reference-seeds",
        type=int,
        default=4,
        help="Maximum reference-network seeds to use when reference expansion is enabled",
    )
    run_parser.add_argument(
        "--enable-retrieval-evaluation",
        action="store_true",
        help="Evaluate literature retrieval quality and write retrieval_evaluations artifacts",
    )
    run_parser.add_argument(
        "--enable-retrieval-judge",
        action="store_true",
        help="Allow optional LLM judge for top retrieved papers; requires --enable-llm",
    )
    run_parser.add_argument(
        "--retrieval-judge-top-k",
        type=int,
        default=5,
        help="Maximum selected papers to judge when retrieval judge is enabled",
    )

    agentlab_parser = subparsers.add_parser(
        "agentlab-config",
        help="Generate an Agent Laboratory YAML config from a topic pack",
    )
    agentlab_parser.add_argument("--topic", required=True, help="Path to a topic pack JSON file")
    agentlab_parser.add_argument(
        "--output",
        default="agentlab_configs/generated_agentlab.yaml",
        help="Output YAML path",
    )
    agentlab_parser.add_argument(
        "--agentlab-dir",
        default="external/AgentLaboratory",
        help="Path to the cloned Agent Laboratory repository",
    )
    agentlab_parser.add_argument("--llm-backend", default="gpt-4o")

    check_parser = subparsers.add_parser(
        "check-config",
        help="Check topic config, local papers, and model-route availability without exposing secrets",
    )
    check_parser.add_argument("--topic", required=True, help="Path to a topic pack JSON file")

    summarize_parser = subparsers.add_parser(
        "summarize-runs",
        help="Summarize recent run_evaluation artifacts",
    )
    summarize_parser.add_argument("--data-dir", default="data")
    summarize_parser.add_argument("--limit", type=int, default=20)

    return parser


def _recalc_dict_depths(nodes: list[dict], node_id: str, depth: int, visited: set[str] | None = None) -> None:
    """Recalculate depth for node dicts (used in manual promotion)."""
    if visited is None:
        visited = set()
    if node_id in visited:
        return
    visited.add(node_id)
    node = next((n for n in nodes if n.get("node_id") == node_id), None)
    if node is None:
        return
    node["depth"] = depth
    for cid in node.get("children_ids", []) or []:
        _recalc_dict_depths(nodes, cid, depth + 1, visited)


def run_workflow(args: argparse.Namespace) -> int:
    load_env_file(".env")
    topic = load_topic_pack(Path(args.topic))
    data_dir = Path(args.data_dir)
    store = ArtifactStore(data_dir / "runs")
    memory = SQLiteMemoryStore(data_dir / "memory.sqlite3")
    lit_memory = LiteratureMemoryStore(data_dir / "literature_memory.sqlite3")
    logger = RunLogger()

    if args.promote:
        scope = memory_scope_for_topic(topic.topic_name)
        tree = lit_memory.load_branch(scope)
        if tree:
            nodes = tree.get("nodes") or []
            target = next(
                (n for n in nodes if n.get("node_id") == args.promote), None
            )
            if target:
                old_root_id = tree.get("root_id", "")
                old_root = next(
                    (n for n in nodes if n.get("node_id") == old_root_id), None
                )
                if old_root and old_root.get("node_id") == target.get("node_id"):
                    print(f"node {args.promote} is already the root, nothing to promote")
                else:
                    # Archive old root and promote target
                    if old_root:
                        old_root["status"] = "archived"
                        # Move old root's other children under new root
                        for cid in list(old_root.get("children_ids") or []):
                            if cid != target["node_id"]:
                                target.setdefault("children_ids", []).append(cid)
                                child = next((n for n in nodes if n.get("node_id") == cid), None)
                                if child:
                                    child["parent_id"] = target["node_id"]
                        old_root["children_ids"] = [target["node_id"]]
                    tree["root_id"] = target["node_id"]
                    target["status"] = "active"
                    # Fix target's parent_id (was child of old root, now root has no parent)
                    target["parent_id"] = ""
                    _recalc_dict_depths(nodes, target["node_id"], 0)
                    lit_memory.write_branch(tree, scope)
                    print(f"promoted node {args.promote} to root")
            else:
                print(f"WARNING: node {args.promote} not found in tree")
        else:
            print("WARNING: no persisted tree found for --promote")

    tools = build_default_tool_registry()

    if args.online:
        tools.register(ArxivTool(max_results=args.max_papers))

    workflow = build_full_research_workflow(
        artifact_store=store,
        memory_store=memory,
        tool_registry=tools,
        logger=logger,
        max_papers=args.max_papers,
        enable_llm=args.enable_llm,
        llm_call_budget=args.llm_call_budget,
        llm_token_budget=args.llm_token_budget,
        enable_experiments=args.enable_experiments,
        enable_tree_search=args.enable_tree_search,
        literature_memory_store=lit_memory,
        max_parallel_branches=args.max_parallel_branches,
        enable_reference_expansion=args.enable_reference_expansion,
        max_reference_seeds=args.max_reference_seeds,
        enable_retrieval_evaluation=args.enable_retrieval_evaluation,
        enable_retrieval_judge=args.enable_retrieval_judge,
        retrieval_judge_top_k=args.retrieval_judge_top_k,
    )
    state = workflow.run(topic)

    print(f"run_id={state.run_id}")
    print(f"stage={state.stage}")
    print(f"run_dir={store.run_dir(state.run_id)}")
    print(f"review_status={state.values.get('review_status', 'unknown')}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        return run_workflow(args)
    if args.command == "agentlab-config":
        load_env_file(".env")
        topic = load_topic_pack(Path(args.topic))
        adapter = AgentLaboratoryAdapter(args.agentlab_dir)
        config = adapter.write_config(topic, args.output, llm_backend=args.llm_backend)
        print(f"config={config.path}")
        print("command=" + " ".join(config.command))
        return 0
    if args.command == "check-config":
        loaded = load_env_file(".env")
        topic = load_topic_pack(Path(args.topic))
        papers = LocalPaperLibrary().scan(topic)
        router = ModelRouter(topic)
        print(f"topic={topic.topic_name}")
        print(f"env_file_loaded_keys={','.join(sorted(loaded.keys())) if loaded else '<none>'}")
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        print(f"DEEPSEEK_API_KEY_present={bool(deepseek_key)}")
        print(f"DEEPSEEK_API_KEY_mask={mask_secret(deepseek_key)}")
        print(f"local_paper_count={len(papers)}")
        for agent_name in [
            "paper_triage",
            "retrieval_judge",
            "synthesis",
            "method_card_extractor",
            "experiment_planner",
            "reviewer_agent",
            "result_parser",
        ]:
            route = router.route_for(agent_name)
            print(
                f"route[{agent_name}] provider={route.provider} model={route.model} "
                f"difficulty={route.task_difficulty or '<unset>'} "
                f"enabled={route.enabled} api_key_env={route.api_key_env}"
            )
        return 0
    if args.command == "summarize-runs":
        from tools.run_evaluation_trends import (
            format_run_evaluation_summary,
            summarize_run_evaluations,
        )
        summary = summarize_run_evaluations(Path(args.data_dir) / "runs", limit=args.limit)
        print(format_run_evaluation_summary(summary))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
