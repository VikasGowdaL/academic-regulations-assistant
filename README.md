# Academic Regulations Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions about university regulations, curricula, academic calendars, and examination policies using supplied university documents. Answers include source document and page references.

## Features

- PDF document loading with page metadata
- Fixed-size, recursive, and semantic text chunking
- Vector search with FAISS or ChromaDB
- Query planning for multi-part questions
- Cohere-powered answer generation
- Session-based conversational context with buffer and summary memory
- Source document and page references in responses
- React chat interface backed by a FastAPI API

## Tech Stack

**Frontend:** React, Vite, Axios, lucide-react  
**Backend:** Python, FastAPI, LangChain  
**LLM and embeddings:** Cohere (`command-a-03-2025`, `embed-english-v3.0`)  
**Vector stores:** FAISS, ChromaDB  
**Document processing:** PyPDF

## Project Structure

```text
academic-regulations-assistant/
├── backend/
│   ├── documents/          # Add university PDF documents here
│   ├── indexes/             # Generated vector indexes (not committed)
│   └── app/
│       ├── ingestion/       # PDF loading and text splitting
│       ├── vectorstores/    # FAISS and ChromaDB integration
│       ├── rag/             # Retrieval, query planning, and answer chain
│       └── memory/          # Buffer, summary, and conversation memory
└── frontend/
    └── src/                 # React application
```

## Prerequisites

- Python 3.10 or newer
- Node.js and npm
- A Cohere API key

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/VikasGowdaL/academic-regulations-assistant.git
cd academic-regulations-assistant
```

### 2. Configure the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-cohere langchain-text-splitters langchain-experimental pypdf faiss-cpu chromadb langchain-chroma pandas numpy ragas
```

Create a `.env` file inside `backend/`:

```env
COHERE_API_KEY=your_cohere_api_key
```

Do not commit your real `.env` file or expose your API key.

### 3. Add university documents and build indexes

Place the PDF documents you are authorized to use in `backend/documents/`. The PDF files and generated vector indexes are excluded from this repository, so each user must provide the documents and generate the indexes locally.

Run the project's document-ingestion, chunking, and vector-store creation scripts from the `backend/` directory before starting the API. The application expects the configured FAISS or ChromaDB index to be available.

### 4. Start the backend

From the `backend/` directory, with the virtual environment activated:

```bash
uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`).

## API

### `POST /chat`

Example request:

```json
{
  "question": "What are the attendance requirements?",
  "store_type": "faiss",
  "session_id": "your-session-id"
}
```

The response contains the question, session ID, generated answer, and source document/page references.

Supported vector store values: `faiss` and `chroma`.

## Notes and Limitations

- The assistant can only answer reliably from the documents that have been ingested into its vector store. Use current, official documents and rebuild the indexes when the corpus changes.
- Page references use PDF page numbering generated during ingestion; these may differ from printed page labels in a document.
- Conversation memory is held in application memory and is not persistent across backend restarts.
- The retrieval distance threshold is an experimental setting and should be evaluated against a representative question set before production use.
- Do not upload confidential, copyrighted, or otherwise restricted university documents unless you have permission to share them.

## Future Improvements

- Evaluate retrieval using Recall@k and answer quality using correctness and faithfulness metrics.
- Measure out-of-scope refusal accuracy with a curated evaluation set.
- Persist conversation sessions and add production deployment configuration.
