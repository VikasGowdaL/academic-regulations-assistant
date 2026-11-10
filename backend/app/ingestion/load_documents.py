
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(directory):

    all_documents = []

    pdf_files = list(Path(directory).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {directory}"
        )

    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        documents = loader.load()

        # Preserve source and human-readable page number
        for document in documents:

            document.metadata["source"] = pdf_file.name

            # PyPDFLoader page numbering starts at 0
            document.metadata["page_number"] = (
                document.metadata["page"] + 1
            )

        all_documents.extend(documents)

        print(f"Pages loaded: {len(documents)}")

    return all_documents


if __name__ == "__main__":

    documents = load_documents("documents")

    print("\n========== INGESTION SUMMARY ==========")

    print("Total pages loaded:", len(documents))

    print("\n========== SAMPLE DOCUMENTS ==========")

    for document in documents[:3]:

        print("\nSource:", document.metadata["source"])

        print("Page:", document.metadata["page_number"])

        print("Extracted text:")
        print(document.page_content[:500])

        print("--------------------------------------")
