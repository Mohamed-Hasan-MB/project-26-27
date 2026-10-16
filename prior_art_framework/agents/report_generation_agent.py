class ReportGenerationAgent:
    """Assemble a predictable report from the agent outputs without another LLM call."""

    def generate(self, invention_analysis: dict, query_result: dict,
                 retrieval_result: dict, novelty_result: dict) -> dict:
        return {
            "report_title": f"Preliminary Prior-Art Analysis: {invention_analysis.get('invention_title') or 'Untitled Invention'}",
            "generated_from": "Uploaded document and locally configured reference collection",
            "invention_summary": {
                "title": invention_analysis.get("invention_title", ""),
                "technical_domain": invention_analysis.get("technical_domain", ""),
                "problem_statement": invention_analysis.get("problem_statement", ""),
                "proposed_solution": invention_analysis.get("proposed_solution", ""),
                "technical_features": invention_analysis.get("technical_features", []),
            },
            "search_strategy": {
                "queries": query_result.get("queries", []),
                "embedding_model": retrieval_result.get("embedding_model", ""),
                "searched_records": retrieval_result.get("searched_records", 0),
            },
            "top_references": [
                {
                    "patent_id": item.get("patent_id"),
                    "title": item.get("title"),
                    "similarity_score": item.get("similarity_score"),
                    "url": item.get("url", ""),
                    "abstract": item.get("abstract", ""),
                    "claims": item.get("claims", ""),
                    "source": item.get("source", ""),
                }
                for item in retrieval_result.get("results", [])
            ],
            "explainable_comparison": novelty_result.get("feature_comparison", []),
            "potentially_distinctive_features": novelty_result.get("potentially_distinctive_features", []),
            "preliminary_assessment": novelty_result.get("overall_assessment", ""),
            "limitations": novelty_result.get("limitations", []),
            "disclaimer": novelty_result.get(
                "legal_disclaimer",
                "This report supports preliminary research only and is not a legal opinion on patentability.",
            ),
        }
