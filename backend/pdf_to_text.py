import pdfplumber
import re
from pathlib import Path

PDF_DIR = Path("data/pdfs")
TEXT_DIR = Path("data/text_clean")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

PAGE_NO_RE = re.compile(r"-\s*\d+\s*-")
DOT_LEADER_RE = re.compile(r"\.{3,}")

def clean_text(text: str) -> str:
    text = PAGE_NO_RE.sub("", text)
    text = DOT_LEADER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

for pdf_file in PDF_DIR.glob("*.pdf"):
    all_text = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)

    cleaned = clean_text("\n".join(all_text))

    out_path = TEXT_DIR / f"{pdf_file.stem}.txt"
    out_path.write_text(cleaned, encoding="utf-8")

    print(f"Saved text: {out_path}")