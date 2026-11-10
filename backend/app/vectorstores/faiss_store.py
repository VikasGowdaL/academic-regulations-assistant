
from pathlib import Path

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS


def create_faiss_store(chunks):

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    index_path = Path("indexes/faiss/recursive")
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(str(index_path))

    print("FAISS index created successfully!")
    print("Total indexed chunks:", len(chunks))

    return vectorstore


def load_faiss_store():

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    vectorstore = FAISS.load_local(
        "indexes/faiss/recursive",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


if __name__ == "__main__":

    from app.ingestion.load_documents import load_documents
    from app.ingestion.split_documents import recursive_splitting

    documents = load_documents("documents")

    chunks = recursive_splitting(documents)

    create_faiss_store(chunks)
