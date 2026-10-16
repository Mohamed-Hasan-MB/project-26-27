"""Run: python run_demo.py path/to/extracted_document.txt"""
import json
import sys
from pathlib import Path

from agents.manager_agent import ManagerAgent


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_demo.py path/to/extracted_document.txt")
        raise SystemExit(2)
    document_path = Path(sys.argv[1])
    if not document_path.exists():
        print(f"Input file not found: {document_path}")
        raise SystemExit(2)
    document_text = document_path.read_text(encoding="utf-8", errors="replace")
    manager = ManagerAgent(records_path="data/reference_patents.json")
    result = manager.analyze_project(document_text)
    output_path = Path("analysis_report.json")
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved output to {output_path.resolve()}")


if __name__ == "__main__":
    main()
