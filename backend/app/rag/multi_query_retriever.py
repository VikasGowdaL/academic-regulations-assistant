
from app.rag.query_planner import plan_queries
from app.rag.retriever import retrieve_documents


# Experimental threshold.
# Lower FAISS L2 distance generally indicates greater similarity.
DISTANCE_THRESHOLD = 1.0


def retrieve_for_multiple_queries(
    question,
    store_type="faiss",
    k=2
):

    # Step 1: Generate sub-queries
    queries = plan_queries(question)

    print("\n========== PLANNED QUERIES ==========")

    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")

    # Step 2: Retrieve documents for every query
    all_documents = []

    for query in queries:

        results = retrieve_documents(
            query,
            store_type=store_type,
            k=k
        )

        for doc, distance in results:

            print(
                f"Retrieved: {doc.metadata.get('source')} "
                f"| Page: {doc.metadata.get('page_number')} "
                f"| Distance: {float(distance):.4f}"
            )

            # Step 3: Filter using the experimental threshold
            if float(distance) <= DISTANCE_THRESHOLD:
                all_documents.append(doc)

    # Step 4: Remove duplicate chunks
    unique_documents = []
    seen = set()

    for doc in all_documents:

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page_number"),
            doc.page_content
        )

        if key not in seen:
            seen.add(key)
            unique_documents.append(doc)

    print("\n========== RETRIEVAL SUMMARY ==========")
    print("Total relevant unique chunks:", len(unique_documents))

    return unique_documents



if __name__ == "__main__":

    questions = [
        "What are the attendance requirements?",
        "What is the academic calendar for 2026-27?",
        "Who won the FIFA World Cup?",
        "How do I cook biryani?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        documents = retrieve_for_multiple_queries(
            question,
            store_type="faiss",
            k=2
        )

        if not documents:
            print("\nRESULT: OUT OF SCOPE")
            print("No sufficiently relevant documents found.")

        else:
            print("\nRESULT: DOCUMENTS RETRIEVED")
            print("Total chunks:", len(documents))

            for doc in documents:
                print(
                    "Source:", doc.metadata.get("source"),
                    "| Page:", doc.metadata.get("page_number")
                )
