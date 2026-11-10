
from app.vectorstores.faiss_store import load_faiss_store
from app.vectorstores.chroma_store import load_chroma_store


def get_vectorstore(store_type="faiss"):

    if store_type == "faiss":
        return load_faiss_store()

    elif store_type == "chroma":
        return load_chroma_store()

    else:
        raise ValueError(
            "Invalid vector store. Choose 'faiss' or 'chroma'."
        )


def retrieve_documents(question, store_type="faiss", k=2):

    vectorstore = get_vectorstore(store_type)

    # Retrieve documents along with FAISS distance scores
    results = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    return results


def display_results(results):

    if not results:
        print("No documents retrieved.")
        return

    for index, (doc, score) in enumerate(results, start=1):

        print(f"\n========== RESULT {index} ==========")

        print(
            "Source:",
            doc.metadata.get("source", "Unknown")
        )

        print(
            "Page:",
            doc.metadata.get("page_number", "Unknown")
        )

        print(
            "Distance score:",
            round(float(score), 4)
        )

        print("\nContent:")
        print(doc.page_content[:500])

        print("------------------------------------")


if __name__ == "__main__":

    questions = [
        "What are the rules for course attendance?",
        "What is the academic calendar for 2026-27?",
        "What subjects are included in the curriculum?",
        "Who won the FIFA World Cup?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        results = retrieve_documents(
            question,
            store_type="faiss",
            k=2
        )

        display_results(results)
