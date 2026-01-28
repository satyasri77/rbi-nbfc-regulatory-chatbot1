import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests

# ---- Config ----
TOP_K = 3

INDEX_PATH = Path("embeddings/rbi_faiss.index")
META_PATH  = Path("embeddings/rbi_metadata.json")

OLLAMA_MODEL =  "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ---- Load embedding model ----
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ---- Load FAISS index ----
index = faiss.read_index(str(INDEX_PATH))

# ---- Load metadata ----
chunks = json.loads(META_PATH.read_text(encoding="utf-8"))


# -------- RETRIEVER --------
def retrieve_top_k(query: str, top_k: int = TOP_K):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results


# -------- REFERENCES (Top 1 available) --------
def extract_references(retrieved_chunks):
    """
    Returns only the top 1 available reference (Chapter / Section / Clause)
    """
    for c in retrieved_chunks:
        md = c.get("metadata", {})
        parts = []

        if md.get("chapter"):
            parts.append(f"Chapter {md['chapter']}")
        if md.get("section"):
            parts.append(f"Section {md['section']}")
        if md.get("subsection"):
            parts[-1] += md["subsection"] if parts else md["subsection"]
        if md.get("clause"):
            parts.append(f"Clause {md['clause']}")

        if parts:
            return [", ".join(parts)]  # only top 1

    # fallback if no chapter/section found
    return ["Not specified in metadata"]


# -------- INTENT CLASSIFIER --------
def classify_intent_llama(query: str) -> str:
    prompt = f"""
You are a strict intent classifier.

Classify the user input into ONE of the following labels ONLY:
GREETING, CHIT_CHAT, REGULATORY

Definitions:
- GREETING: Hi, Hello, Hey, Good morning, etc.
- CHIT_CHAT: Casual conversation, small talk, personal questions (e.g. "How is it going?")
- REGULATORY: Questions or instructions related to RBI regulations for NBFCs

Respond with ONLY ONE LABEL.
No explanations.

User input:
{query}

Answer:
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5}
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip().upper()


def is_obvious_greeting(query: str) -> bool:
    return query.strip().lower() in {"hi", "hello", "hey", "hi there", "hello there"}


# -------- GENERATOR (RAG) --------
def generate_rag_answer(query, retrieved_chunks, chat_history=None):
    if chat_history is None:
        chat_history = []

    context_text = ""
    for c in retrieved_chunks:
        context_text += f"{c['text']}\n\n"

    history_text = "\n".join(chat_history[-4:])

    prompt = f"""
You are an expert strictly limited to RBI regulations governing Non-Banking Financial Companies (NBFCs).


SCOPE RESTRICTION
• You may respond only on matters directly related to RBI regulatory framework for NBFCs.
• If a query falls outside this scope, respond only with:
"I can only assist in RBI regulations governing Non-Banking Financial Companies (NBFCs)"


GREETING HANDLING
• Respond with "Hi, How can I help you assist today?" ONLY if the user input is purely a generic greeting.
• Do NOT include greetings in the answer text if the question is regulatory.


CONTEXT USAGE
• Use ONLY the context provided below.
• Do NOT mention chapter, section, clause, or subsection in the answer text — references will be appended by the system.


RESPONSE STRUCTURE
• If multiple conditions or criteria exist, present them as numbered points.
• Do NOT add explanations, summaries, opinions, or commentary beyond what is stated in the context.

MISSING INFORMATION HANDLING
• If the requested information is not explicitly in the context, respond exactly:
"Not specified in the provided RBI guidelines."

Conversation History (intent only):
{history_text}

Context:
{context_text}

Question:
{query}

Answer:
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512}
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()


# -------- MAIN API WITH JSON OUTPUT --------
def answer_query(query, chat_history=None):
    """
    Returns a JSON-structured response:
    {
        "intent": "GREETING | CHIT_CHAT | REGULATORY",
        "answer": "string",
        "references": []  # only for REGULATORY, empty otherwise
    }
    """
    # Step 1: Handle obvious greetings first
    if is_obvious_greeting(query):
        return {"intent": "GREETING", "answer": "Hi, How can I help you assist today?", "references": []}

    # Step 2: Classify intent via LLaMA
    intent = classify_intent_llama(query)

    # GREETING
    if intent == "GREETING":
        return {"intent": "GREETING", "answer": "Hi, How can I help you assist today?", "references": []}

    # CHIT_CHAT
    if intent == "CHIT_CHAT":
        return {"intent": "CHIT_CHAT",
                "answer": "I can only assist in RBI regulations governing Non-Banking Financial Companies (NBFCs)",
                "references": []}

    # REGULATORY
    retrieved = retrieve_top_k(query)
    answer = generate_rag_answer(query, retrieved, chat_history)
    references = extract_references(retrieved) # <-- top 1 only


    return {"intent": "REGULATORY",
    "answer": answer,
    "references": references}

# -------- TEST --------
# if __name__ == "__main__":
#     q = "List all the conditions for NBFCs to open branch abroad"
#     response = answer_query_json(q)
#     print(json.dumps(response, indent=2))