# Lightweight Prior-Art Analysis Agent Framework

This is a prototype implementation of the original project direction: extract technical features from an uploaded invention description, generate search queries, retrieve semantically similar records from a local collection, compare features with supporting text, and produce a structured report.

## Important scope note

The included `data/reference_patents.json` contains **synthetic demo records only**. They are not real patents and must not be presented as real prior art. The retrieval agent searches only the local JSON file; it does not connect to Google Patents, USPTO, or another live patent source. Replace the demo records with accurately sourced records before evaluating the system as a prior-art search tool.

## Files

- `agents/document_understanding_agent.py` — extracts invention details and technical features using Ollama.
- `agents/query_generation_agent.py` — creates several targeted search queries using Ollama, with a fallback if JSON generation fails.
- `agents/prior_art_retrieval_agent.py` — embeds the local reference records and searches them using cosine similarity. Reference embeddings are cached on disk and CUDA is selected automatically when available.
- `agents/novelty_analysis_agent.py` — compares extracted features with retrieved abstracts/claims and asks the LLM to provide exact supporting quotes where available.
- `agents/report_generation_agent.py` — assembles the final report without an extra LLM call.
- `agents/manager_agent.py` — runs the pipeline in order and returns intermediate results plus the final report.
- `run_demo.py` — command-line entry point for already-extracted text.

## Setup

1. Install Python 3.10 or newer.
2. Install and start Ollama.
3. Pull the model used by the code:

   ```bash
   ollama pull llama3.1:latest
   ```

4. From this directory, install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Save the text extracted from a project/invention PDF to a UTF-8 `.txt` file and run:

   ```bash
   python run_demo.py example_invention.txt
   ```

The output is saved as `analysis_report.json`.

## Reference-record format

Each record in `data/reference_patents.json` should contain:

```json
{
  "patent_id": "publication or document identifier",
  "title": "document title",
  "abstract": "abstract text",
  "claims": "claims or relevant passages, if available",
  "url": "source URL, if available",
  "source": "where the record came from"
}
```

## Connect to your existing PDF upload application

Your current backend should extract text from the uploaded PDF, then call:

```python
from agents.manager_agent import ManagerAgent

manager = ManagerAgent(records_path="data/reference_patents.json")
result = manager.analyze_project(extracted_pdf_text)
```

Return `result` through your existing Flask/FastAPI route. Adjust the file path to an absolute path or a path based on your app's project root if the server runs from a different working directory.

## Limitations

- This starter expects text extraction to happen before `analyze_project()`; it does not parse PDFs itself.
- The included demo records are synthetic.
- Semantic similarity is a ranking signal, not a probability of novelty or proof that a feature is disclosed.
- LLM-generated comparisons can be wrong. Verify quotes and compare against the original source text.
- This output is preliminary technical research, not legal advice or a legal conclusion about patentability.
