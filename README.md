# Academic Regulations Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions about university regulations, curricula, academic calendars, and examination policies using supplied university documents. Answers include source-document and page references, and the assistant is instructed to avoid answering when retrieved documents do not provide sufficient information.

## Features

- PDF ingestion with page-level metadata using PyPDF.
- Fixed-size, recursive, and semantic document chunking strategies.
- Vector indexing and retrieval with FAISS or ChromaDB.
- Cohere embeddings and chat generation.
- Query planning for questions with multiple parts.
- Conversational buffer and summary memory, separated by session ID.
- FastAPI chat endpoint.
- React + Vite chat interface with source display.

## Tech stack

- **Frontend:** React, Vite, Axios, Lucide React
- **Backend:** Python, FastAPI, Uvicorn, LangChain
- **AI:** Cohere Chat API (`command-a-03-2025`) and embeddings (`embed-english-v3.0`)
- **Vector stores:** FAISS and ChromaDB

## Project structure

```text
academic-regulations-assistant/
├── backend/
│   ├── documents/              # Add your university PDF documents here
│   ├── indexes/                # Generated vector indexes (not committed)
│   └── app/
│       ├── ingestion/          # PDF loading and chunking
│       ├── vectorstores/       # FAISS and ChromaDB creation/loading
│       ├── rag/                # Query planning, retrieval, prompts and RAG chain
│       ├── memory/             # Buffer, summary and conversation memory
│       ├── evaluation/         # Evaluation package (work in progress)
│       └── main.py             # FastAPI application
└── frontend/
    └── src/                    # React chat interface
```

## Prerequisites

- Python 3.10 or newer (use a version compatible with the installed LangChain dependencies).
- Node.js and npm.
- A Cohere API key.

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
```

Install the backend packages used by the project:

```bash
pip install fastapi uvicorn python-dotenv pypdf pandas numpy
pip install langchain langchain-community langchain-cohere
pip install langchain-text-splitters langchain-experimental
pip install faiss-cpu chromadb langchain-chroma
```

Create a `.env` file inside `backend/`:

```env
COHERE_API_KEY=your_cohere_api_key
```

Keep this key private. The `.env` file is excluded from Git.

### 3. Add university documents and build an index

Place PDF files you are authorized to use in:

```text
backend/documents/
```

The PDFs and generated vector indexes are intentionally excluded from this repository. You must provide your own document corpus before running ingestion.

From the `backend/` directory, with the virtual environment active, run one of the following to create an index using recursive chunking:

**FAISS**
```bash
python -m app.vectorstores.faiss_store
```

**ChromaDB**
```bash
python -m app.vectorstores.chroma_store
```

The generated indexes are saved under `backend/indexes/`. The chat API defaults to FAISS; select ChromaDB by sending `"store_type": "chroma"` in the chat request.

To inspect the three available chunking strategies (fixed-size, recursive, and semantic):

```bash
python -m app.ingestion.split_documents
```

Semantic splitting uses Cohere embeddings and may make additional API calls.

### 4. Start the backend

From the `backend/` directory:

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

## Chat API

**Endpoint:** `POST /chat`

Example request:

```json
{
  "question": "What are the attendance requirements?",
  "store_type": "faiss",
  "session_id": "student-session-001"
}
```

The response contains the question, session ID, generated answer, and source list (document name and page number).

## Implementation notes and limitations

- Conversation memory is held in a Python in-memory dictionary. It is separated by session ID but is cleared when the backend process restarts; it is not persistent storage.
- The FAISS distance threshold in the multi-query retriever is experimental and should be calibrated against a representative evaluation set before production use.
- The evaluation package is present as a project area, but a completed 50-question evaluation and reported metrics are not included in this version.
- Update the university name shown in the frontend header to match the institution represented by the PDFs you add.
- Only upload university documents if you have permission to redistribute them. Keep private documents and API credentials out of public repositories.

## Future improvements

- Complete a 50-question evaluation set and report Recall@k, answer correctness, faithfulness, and out-of-scope refusal accuracy.
- Persist conversation sessions and add session expiry.
- Calibrate retrieval thresholds separately for each vector store.
- Add automated tests and deployment configuration.

## License

No license has been added yet. Add a license before others reuse or distribute this project.
