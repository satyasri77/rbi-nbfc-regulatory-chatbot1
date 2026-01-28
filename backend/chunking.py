import json
import re
from pathlib import Path
import tiktoken

TEXT_DIR = Path("data/text_clean")
OUTPUT_PATH = Path("data/chunks/rbi_chunks.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

tokenizer = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 450
MIN_TOKENS = 120

# ---- Regex patterns ----
SECTION_RE   = re.compile(r"^Section\s+[IVX]+", re.I)
CHAPTER_RE   = re.compile(r"^Chapter\s+[IVX]+", re.I)
CLAUSE_RE    = re.compile(r"^(\d+)\.\s+")
SUBCLAUSE_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)")

def token_count(text: str) -> int:
    return len(tokenizer.encode(text))

def split_if_needed(text):
    words = text.split()
    chunks, current = [], []

    for word in words:
        current.append(word)
        if token_count(" ".join(current)) >= MAX_TOKENS:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

all_chunks = []
chunk_id = 0

for txt_file in TEXT_DIR.glob("*.txt"):
    document = txt_file.stem
    lines = txt_file.read_text(encoding="utf-8").splitlines()

    current_section = None
    current_chapter = None
    current_clause = None
    buffer = []

    def flush_buffer():
        global chunk_id
        if not buffer:
            return

        text = " ".join(buffer).strip()
        for part_no, part in enumerate(split_if_needed(text), start=1):
            if token_count(part) < MIN_TOKENS:
                continue

            all_chunks.append({
                "id": chunk_id,
                "text": part,
                "metadata": {
                    "document": document,
                    "section": current_section,
                    "chapter": current_chapter,
                    "clause": current_clause,
                    "part": part_no
                }
            })
            chunk_id += 1

        buffer.clear()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if SECTION_RE.match(line):
            flush_buffer()
            current_section = line
            continue

        if CHAPTER_RE.match(line):
            flush_buffer()
            current_chapter = line
            continue

        sub_match = SUBCLAUSE_RE.match(line)
        if sub_match:
            flush_buffer()
            current_clause = sub_match.group(1)
            buffer.append(line)
            continue

        clause_match = CLAUSE_RE.match(line)
        if clause_match:
            flush_buffer()
            current_clause = clause_match.group(1)
            buffer.append(line)
            continue

        buffer.append(line)

    flush_buffer()

OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
print(f"Saved {len(all_chunks)} chunks")