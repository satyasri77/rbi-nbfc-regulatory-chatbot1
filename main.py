from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

from backend.rag_search import answer_query

app = FastAPI(title="RBI NBFC RAG Bot")

# ---- CORS (for browser access) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request / Response Schemas ----
class ChatRequest(BaseModel):
    query: str
    chat_history: list[str] = []

class ChatResponse(BaseModel):
    answer: str
    references: list[str]


# ---- Serve HTML ----
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("frontend/index.html")
    return html_path.read_text(encoding="utf-8")


# ---- Chat Endpoint ----
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response_json = answer_query(
        query=req.query,
        chat_history=req.chat_history
    )
    # Return JSON with intent + answer + references
    return ChatResponse(
    answer=response_json["answer"],
    references=response_json.get("references", [])
    )


# ---- Run locally ----
# uvicorn main:app --reload