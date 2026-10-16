"""Orchestrates the prior-art analysis pipeline."""
import logging
from concurrent.futures import ThreadPoolExecutor

from agents.common import DEFAULT_MODEL
from agents.document_understanding_agent import DocumentUnderstandingAgent
from agents.query_generation_agent import QueryGenerationAgent
from agents.prior_art_retrieval_agent import PriorArtRetrievalAgent
from agents.novelty_analysis_agent import ExplainableNoveltyAnalysisAgent
from agents.report_generation_agent import ReportGenerationAgent

logger = logging.getLogger(__name__)


class ManagerAgent:
    def __init__(
        self,
        records_path: str = "data/reference_patents.json",
        model: str = DEFAULT_MODEL,
        top_k: int = 10,
    ):
        self.document_agent = DocumentUnderstandingAgent(model=model)
        self.query_agent = QueryGenerationAgent()
        self.retrieval_agent = PriorArtRetrievalAgent(records_path=records_path, top_k=top_k)
        self.novelty_agent = ExplainableNoveltyAnalysisAgent(model=model)
        self.report_agent = ReportGenerationAgent()

    def analyze_project(self, document_text: str) -> dict:
        """Run the agents in dependency order and return both intermediate and final results."""
        # Validate type before calling .strip() so non-string inputs raise a clear error.
        if not isinstance(document_text, str) or not document_text.strip():
            return {"error": "No document text received."}

        # Warm the retrieval agent (records + embeddings) in a background thread
        # while the first LLM stage runs. Saves a few seconds on cold starts.
        with ThreadPoolExecutor(max_workers=1) as pool:
            warm = pool.submit(self.retrieval_agent.warmup)
            try:
                invention = self.document_agent.analyze(document_text)
            except Exception as exc:
                logger.exception("Document understanding stage failed")
                return {"error": f"Analysis pipeline failed: {type(exc).__name__}: {exc}"}
            finally:
                # Surface any warmup error (e.g. missing records file) as a pipeline error.
                try:
                    warm.result()
                except Exception as exc:
                    logger.exception("Retrieval warmup failed")
                    return {"error": f"Analysis pipeline failed: {type(exc).__name__}: {exc}"}

        try:
            queries = self.query_agent.generate(invention)
            retrieval = self.retrieval_agent.search(invention, queries)
            novelty = self.novelty_agent.analyze(invention, retrieval)
            report = self.report_agent.generate(invention, queries, retrieval, novelty)
            return {
                "invention_analysis": invention,
                "query_generation": queries,
                "prior_art_retrieval": retrieval,
                "novelty_analysis": novelty,
                "report": report,
            }
        except Exception as exc:
            # Log the full traceback server-side; return a concise error to the caller.
            logger.exception("Analysis pipeline failed")
            return {"error": f"Analysis pipeline failed: {type(exc).__name__}: {exc}"}