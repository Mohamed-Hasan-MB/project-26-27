from agents.common import FAST_MODEL, call_llm_json, safe_list


class QueryGenerationAgent:
    """Generate varied search queries from the extracted invention features."""

    def __init__(self, model: str = FAST_MODEL, max_queries: int = 8):
        # A 1B model handles this structured, short-output task well and runs
        # ~5x faster than the main model. Schema is unchanged.
        self.model = model
        self.max_queries = max_queries

    def generate(self, invention_analysis: dict) -> dict:
        prompt = f"""
Create search queries for finding potentially relevant prior-art documents.
Generate varied queries: (1) core problem, (2) proposed mechanism/method, (3) technical components,
and (4) important synonyms. Avoid overly broad queries and do not assert that the invention is novel.
Use only information in the analysis. Return JSON only in this structure:
{{"queries": ["..."], "query_rationale": ["..."]}}

INVENTION ANALYSIS:
{invention_analysis}
"""
        try:
            result = call_llm_json(prompt, model=self.model, max_tokens=400)
            queries = [str(q).strip() for q in safe_list(result.get("queries")) if str(q).strip()]
            rationales = [str(r).strip() for r in safe_list(result.get("query_rationale"))]
        except Exception:
            # Deterministic fallback allows the pipeline to continue if the LLM response fails.
            queries = []
            for key in ("short_search_summary", "problem_statement", "proposed_solution"):
                value = invention_analysis.get(key, "")
                if isinstance(value, str) and value.strip():
                    queries.append(value.strip())
            queries.extend(
                str(x).strip()
                for x in safe_list(invention_analysis.get("technical_features"))
                if str(x).strip()
            )
            rationales = ["Fallback query created from extracted invention text."] * len(queries)

        # De-duplicate while preserving order.
        unique = []
        seen = set()
        for query in queries:
            normalized = " ".join(query.lower().split())
            if normalized and normalized not in seen:
                unique.append(query)
                seen.add(normalized)

        unique = unique[: self.max_queries]
        # Keep rationales aligned with the final query list after dedup + cap.
        rationales = (rationales + [""] * len(unique))[: len(unique)]

        return {
            "queries": unique,
            "query_rationale": rationales,
        }