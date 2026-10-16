"""Download PDFs for every patent in the uploaded CSV via Google Patents."""
import csv
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

INPUT_CSV = "job_N_bpT40Dt8zmm8OboksxLuIid602.csv"
OUTPUT_DIR = Path("patents_pdf")
MANIFEST = OUTPUT_DIR / "manifest.csv"
DELAY_SECONDS = 1.2          # polite; keeps you under Google's throttling
TIMEOUT = 30
MAX_RETRIES = 3

HEADERS = {
    # Identify yourself. Google is more tolerant of clear user agents.
    "User-Agent": (
        "Mozilla/5.0 (compatible; PriorArtFetcher/1.0; "
        "+https://example.com/contact)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")


def fetch_pdf_url(session: requests.Session, page_url: str) -> str | None:
    """Return the citation_pdf_url for a Google Patents page, or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(page_url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
                if meta and meta.get("content"):
                    return meta["content"].strip()
                # Fallback: look for a visible "Download PDF" link
                link = soup.find("a", href=re.compile(r"\.pdf($|\?)", re.I))
                if link and link.get("href"):
                    return link["href"]
                return None
            if resp.status_code in (429, 503):
                wait = DELAY_SECONDS * (2 ** attempt)
                log.warning("HTTP %s on %s; sleeping %ss", resp.status_code, page_url, wait)
                time.sleep(wait)
                continue
            log.warning("HTTP %s for %s", resp.status_code, page_url)
            return None
        except requests.RequestException as exc:
            log.warning("Request error for %s (attempt %s): %s", page_url, attempt, exc)
            time.sleep(DELAY_SECONDS * attempt)
    return None


def download_pdf(session: requests.Session, pdf_url: str, dest: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(pdf_url, headers=HEADERS, timeout=TIMEOUT, stream=True) as resp:
                if resp.status_code != 200:
                    log.warning("PDF HTTP %s for %s", resp.status_code, pdf_url)
                    return False
                tmp = dest.with_suffix(".pdf.part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
                return True
        except requests.RequestException as exc:
            log.warning("PDF error (attempt %s) for %s: %s", attempt, pdf_url, exc)
            time.sleep(DELAY_SECONDS * attempt)
    return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    log.info("Loaded %d rows from %s", len(rows), INPUT_CSV)

    session = requests.Session()
    manifest_rows = []

    for i, row in enumerate(rows, 1):
        pub = (row.get("publication_number") or "").strip()
        url = (row.get("url") or "").strip()
        if not pub or not url:
            continue

        # Safe filename
        safe_pub = re.sub(r"[^A-Za-z0-9._-]+", "_", pub)
        dest = OUTPUT_DIR / f"{safe_pub}.pdf"

        if dest.exists() and dest.stat().st_size > 1024:
            log.info("[%d/%d] skip (exists): %s", i, len(rows), dest.name)
            manifest_rows.append({"publication_number": pub, "url": url,
                                  "pdf_url": "", "status": "cached",
                                  "path": str(dest)})
            continue

        log.info("[%d/%d] %s", i, len(rows), pub)
        pdf_url = fetch_pdf_url(session, url)
        if not pdf_url:
            log.warning("  no PDF link found for %s", pub)
            manifest_rows.append({"publication_number": pub, "url": url,
                                  "pdf_url": "", "status": "no_pdf_link",
                                  "path": ""})
            time.sleep(DELAY_SECONDS)
            continue

        ok = download_pdf(session, pdf_url, dest)
        manifest_rows.append({
            "publication_number": pub,
            "url": url,
            "pdf_url": pdf_url,
            "status": "ok" if ok else "download_failed",
            "path": str(dest) if ok else "",
        })
        if ok:
            log.info("  saved %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
        time.sleep(DELAY_SECONDS)

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["publication_number", "url", "pdf_url", "status", "path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    ok = sum(1 for r in manifest_rows if r["status"] in ("ok", "cached"))
    log.info("Done. %d/%d downloaded. Manifest: %s", ok, len(manifest_rows), MANIFEST)


if __name__ == "__main__":
    main()