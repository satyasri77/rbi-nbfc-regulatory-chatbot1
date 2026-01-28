# RBI NBFC Regulatory Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot is developed to answer queries based on RBI regulations governing Non-Banking Financial Companies (NBFCs). This system is optimized for local execution on **MacBook Air M1 (8GB RAM)**.

---

## Overview
This bot provides an interface for regulatory compliance by using the **Llama 3.2: 3b** model in a limited knowledge base. Instead of relying on general training data, the bot strictly uses retrieved snippets from official RBI Master Directions to ensure accuracy and reduce hallucinations.

## Techology used
* **LLM:** Llama 3.2 (3B) via [Ollama](https://ollama.com/)
* **Backend:** FastAPI (Python 3.9+)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** `all-mpnet-base-v2` (Sentence-Transformers)
* **Frontend:** HTML

## How it Works (RAG Architecture)

1. **Ingestion:** RBI Guidelines (PDFs) are parsed, cleaned, and split into semantic chunks.
2. **Vector Storage:** Chunks are converted into 768-dimensional embeddings using `all-mpnet-base-v2` and stored in a **FAISS** index.
3. **Retrieval:** When a user asks a question, the system finds the top 3 most relevant regulatory clauses.
4. **Generation:** A structured prompt containing the user query and retrieved chunks are sent to **Llama 3.2** via **Ollama**.
5. **Interface:** A **FastAPI** backend serves the model's response to a responsive web UI.
   
---

## Application Screenshot

![RBI NBFC Chatbot Screenshot](./screenshot/bot_image_new.png)

---
