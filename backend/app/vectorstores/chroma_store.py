
from pathlib import Path

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def create_chroma_store(chunks):

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    persist_directory = "indexes/chroma/recursive"

    # Create the directory if it doesn't exist
    Path(persist_directory).mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="academic_regulations_recursive",
        persist_directory=persist_directory
    )

    print("ChromaDB index created successfully!")
    print("Total indexed chunks:", len(chunks))

    return vectorstore


def load_chroma_store():

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    vectorstore = Chroma(
        collection_name="academic_regulations_recursive",
        embedding_function=embeddings,
        persist_directory="indexes/chroma/recursive"
    )

    return vectorstore


if __name__ == "__main__":

    from app.ingestion.load_documents import load_documents
    from app.ingestion.split_documents import recursive_splitting

    documents = load_documents("documents")

    chunks = recursive_splitting(documents)

    create_chroma_store(chunks)
