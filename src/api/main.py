from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.retrieval.rag_chain import ask
import time

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Jordan Legal RAG Assistant",
    description="AI-powered Q&A over Jordanian data protection laws",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────
# Allows the Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ───────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    response_time_seconds: float

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "healthy", "service": "Jordan Legal RAG Assistant"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about Jordanian data protection law.
    Returns an answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    try:
        result = ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_time = round(time.time() - start_time, 2)

    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        response_time_seconds=response_time
    )

@app.get("/")
def root():
    return {
        "message": "Jordan Legal RAG Assistant API",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask"
    }