from __future__ import annotations

from agents import (
    AutonomousExperimentAgent,
    BranchSelectionAgent,
    BranchToPlanAgent,
    CodebaseAnalyzerAgent,
    DeveloperAgent,
    ExperimentDecisionAgent,
    EvidenceCheckerAgent,
    ExperimentPlannerAgent,
    LiteratureMemoryPersistenceAgent,
    LiteratureSearchAgent,
    LocalPaperLibraryAgent,
    LocalPaperParserAgent,
    MethodCardExtractorAgent,
    MethodCardRetrieverAgent,
    OpportunityAgent,
    PaperReaderAgent,
    PaperTriageAgent,
    ReferenceExtractorAgent,
    ResearchManagerAgent,
    RetrievalEvaluationAgent,
    ReviewerAgent,
    RunEvaluationAgent,
    SynthesisAgent,
    TreePrunerAgent,
    TreeSearchAgent,
)
from core.artifact_store import ArtifactStore
from core.run_logger import RunLogger
from core.workflow import Workflow
from tools.tool_registry import ToolRegistry


def build_full_research_workflow(
    artifact_store: ArtifactStore,
    memory_store: object,
    tool_registry: ToolRegistry,
    logger: RunLogger,
    max_papers: int = 8,
    enable_llm: bool = False,
    llm_call_budget: int | None = None,
    llm_token_budget: int | None = None,
    enable_experiments: bool = False,
    enable_tree_search: bool = False,
    literature_memory_store: object = None,
    max_parallel_branches: int = 1,
    enable_reference_expansion: bool = False,
    max_reference_seeds: int = 4,
    enable_retrieval_evaluation: bool = False,
    enable_retrieval_judge: bool = False,
    retrieval_judge_top_k: int = 5,
) -> Workflow:
    agents: list = [
        ResearchManagerAgent(),
        LocalPaperLibraryAgent(),
        LiteratureSearchAgent(lit_memory_store=literature_memory_store),
        PaperTriageAgent(),
    ]
    if enable_retrieval_evaluation:
        agents.append(RetrievalEvaluationAgent())
    agents.extend([
        LocalPaperParserAgent(),
        PaperReaderAgent(),
        ReferenceExtractorAgent(),
        EvidenceCheckerAgent(),
        MethodCardExtractorAgent(),
        SynthesisAgent(),
        CodebaseAnalyzerAgent(),
        MethodCardRetrieverAgent(lit_memory_store=literature_memory_store),
        OpportunityAgent(),
        ExperimentPlannerAgent(),
    ])
    # Branch selection + plan conversion run before DeveloperAgent so the
    # selected branch plan is consumed in the current run.
    if enable_tree_search:
        agents.append(TreePrunerAgent())
        agents.append(BranchSelectionAgent(lit_memory_store=literature_memory_store))
        agents.append(BranchToPlanAgent())
    agents.append(DeveloperAgent())
    agents.append(AutonomousExperimentAgent())
    agents.append(ExperimentDecisionAgent())
    # TreeSearchAgent runs after experiment decision to generate new branches
    # or write back results.
    if enable_tree_search:
        agents.append(TreeSearchAgent(lit_memory_store=literature_memory_store))
    agents.append(ReviewerAgent())
    agents.append(RunEvaluationAgent())
    agents.append(LiteratureMemoryPersistenceAgent(lit_memory_store=literature_memory_store))

    return Workflow(
        name="full_research_loop_v1",
        agents=agents,
        artifact_store=artifact_store,
        memory_store=memory_store,
        tool_registry=tool_registry,
        logger=logger,
        settings={
            "max_papers": max_papers,
            "enable_llm": enable_llm,
            "enable_experiments": enable_experiments,
            "llm_call_budget": llm_call_budget,
            "llm_token_budget": llm_token_budget,
            "llm_calls_used": 0,
            "llm_tokens_used": 0,
            "enable_tree_search": enable_tree_search,
            "max_parallel_branches": max_parallel_branches,
            "enable_reference_expansion": enable_reference_expansion,
            "max_reference_seeds": max_reference_seeds,
            "enable_retrieval_evaluation": enable_retrieval_evaluation,
            "enable_retrieval_judge": enable_retrieval_judge,
            "retrieval_judge_top_k": retrieval_judge_top_k,
        },
    )
