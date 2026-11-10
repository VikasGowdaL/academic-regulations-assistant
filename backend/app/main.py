
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.chain import generate_answer


app = FastAPI(
    title="Academic Regulations Assistant",
    description="RAG-based university regulations chatbot",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    store_type: str = "faiss"
    session_id: str


@app.get("/")
def home():
    return {
        "message": "Academic Regulations Assistant API is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not request.session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty."
        )

    if request.store_type not in ["faiss", "chroma"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid vector store."
        )

    try:

        result = generate_answer(
            question=request.question,
            store_type=request.store_type,
            k=2,
            session_id=request.session_id
        )

        return {
            "question": request.question,
            "session_id": request.session_id,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
