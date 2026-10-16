from agents.common import DEFAULT_MODEL, call_llm_json


class DocumentUnderstandingAgent:
    """Extract an invention's problem, solution, and technical features."""

    def __init__(self, model: str = DEFAULT_MODEL, max_chars: int = 12000):
        self.model = model
        # 12000 chars is enough for most abstracts + intro + claims; keeps prompt
        # prefill short so the stage finishes in seconds instead of minutes.
        self.max_chars = max_chars

    @staticmethod
    def _smart_truncate(text: str, limit: int, head_ratio: float = 0.6) -> str:
        """Keep the head and tail of the document, drop the middle.

        The beginning usually holds the abstract/problem statement; the end
        usually holds the conclusions and claims. Prefix-only truncation loses
        the latter and often degrades feature extraction on long PDFs.
        """
        if len(text) <= limit:
            return text
        head_len = int(limit * head_ratio)
        tail_len = limit - head_len
        return (
            text[:head_len].rstrip()
            + "\n\n[… middle omitted for length …]\n\n"
            + text[-tail_len:].lstrip()
        )

    def analyze(self, document_text: str) -> dict:
        if not document_text or not document_text.strip():
            raise ValueError("The uploaded document contains no extractable text.")

        text = self._smart_truncate(document_text.strip(), self.max_chars)
        prompt = f"""
Analyze the supplied invention description or research paper for prior-art searching.
Use only facts supported by the supplied text. Do not invent features. If a field is missing,
use an empty string or empty list. Keep each technical feature specific and independently testable.

Return this JSON structure:
{{
  "invention_title": "",
  "technical_domain": "",
  "problem_statement": "",
  "proposed_solution": "",
  "technical_features": [""],
  "components": [""],
  "methods_or_algorithms": [""],
  "keywords": [""],
  "short_search_summary": ""
}}

DOCUMENT TEXT:
{text}
"""
        # 900 output tokens is plenty for the ~9 short fields above.
        result = call_llm_json(prompt, model=self.model, max_tokens=900)
        for key in ("technical_features", "components", "methods_or_algorithms", "keywords"):
            if not isinstance(result.get(key), list):
                result[key] = []
        for key in ("invention_title", "technical_domain", "problem_statement",
                    "proposed_solution", "short_search_summary"):
            if not isinstance(result.get(key), str):
                result[key] = ""
        if not result["technical_features"]:
            result["technical_features"] = result["keywords"][:]
        return result