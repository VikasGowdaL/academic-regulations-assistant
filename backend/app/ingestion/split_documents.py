
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

import os


load_dotenv()


# ---------------------------------------
# 1. FIXED-SIZE SPLITTING
# ---------------------------------------

def fixed_size_splitting(documents):

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ---------------------------------------
# 2. RECURSIVE SPLITTING
# ---------------------------------------

def recursive_splitting(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ---------------------------------------
# 3. SEMANTIC SPLITTING
# ---------------------------------------

def semantic_splitting(documents):

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ---------------------------------------
# CHUNK STATISTICS
# ---------------------------------------

def display_statistics(strategy_name, chunks):

    print(f"\n========== {strategy_name} ==========")

    print("Total chunks:", len(chunks))

    if not chunks:
        print("No chunks generated.")
        return

    total_characters = sum(
        len(chunk.page_content)
        for chunk in chunks
    )

    average_size = total_characters / len(chunks)

    print("Total characters:", total_characters)

    print("Average chunk size:", round(average_size, 2))

    print("\nFirst chunk:")

    print(chunks[0].page_content[:500])

    print("\nMetadata:")

    print(chunks[0].metadata)


# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------

if __name__ == "__main__":

    from app.ingestion.load_documents import load_documents

    documents = load_documents("documents")

    print("\nOriginal pages:", len(documents))

    # Fixed-size
    fixed_chunks = fixed_size_splitting(documents)

    display_statistics(
        "FIXED-SIZE SPLITTING",
        fixed_chunks
    )

    # Recursive
    recursive_chunks = recursive_splitting(documents)

    display_statistics(
        "RECURSIVE SPLITTING",
        recursive_chunks
    )

    # Semantic
    semantic_chunks = semantic_splitting(documents)

    display_statistics(
        "SEMANTIC SPLITTING",
        semantic_chunks
    )
