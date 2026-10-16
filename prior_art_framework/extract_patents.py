
from pathlib import Path
import pymupdf # PyMuPDF

PDF_DIR = Path("patent_dataset")
OUTPUT_DIR = Path("extracted_patents")
OUTPUT_DIR.mkdir(exist_ok=True)

pdf_files = list(PDF_DIR.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in patent_dataset.")
else:
    for pdf_path in pdf_files:
        try:
            with pymupdf.open(pdf_path) as pdf:
                text = "\n".join(page.get_text() for page in pdf)

            output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
            output_path.write_text(text, encoding="utf-8")

            print(f"{pdf_path.name}: extracted {len(text)} characters")

            if len(text.strip()) < 100:
                print("  WARNING: Very little text extracted; this PDF may need OCR.")

        except Exception as error:
            print(f"Could not process {pdf_path.name}: {error}")

print("Extraction finished.")
