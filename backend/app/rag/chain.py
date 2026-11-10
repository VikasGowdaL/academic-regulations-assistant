
import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere

from app.rag.multi_query_retriever import (
    retrieve_for_multiple_queries
)
from app.rag.prompts import SYSTEM_PROMPT
from app.memory.conversation_memory import ConversationMemory

load_dotenv()

# Maintain separate memory for each session
session_memories = {}


def get_session_memory(session_id):

    if session_id not in session_memories:
        session_memories[session_id] = ConversationMemory(
            max_messages=6
        )

    return session_memories[session_id]


def generate_answer(
    question,
    store_type="faiss",
    k=2,
    session_id="default"
):

    memory = get_session_memory(session_id)

    print("\n========== PROCESSING QUESTION ==========")

    # Step 1: Retrieve previous conversation context
    conversation_context = memory.get_context()

    contextual_question = f"""
    Previous conversation:
    {conversation_context}

    Current student question:
    {question}
    """

    # Step 2: Retrieve relevant documents
    documents = retrieve_for_multiple_queries(
        contextual_question,
        store_type=store_type,
        k=k
    )

    # Step 3: Refuse if no relevant documents
    if not documents:

        answer = (
            "I could not find sufficient information "
            "about this question in the available "
            "university documents."
        )

        memory.add_message("user", question)
        memory.add_message("assistant", answer)

        return {
            "answer": answer,
            "sources": []
        }

    # Step 4: Prepare document context
    context_parts = []
    sources = []

    for doc in documents:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_number", "Unknown")

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

        sources.append({
            "source": source,
            "page": page
        })

    context = "\n\n".join(context_parts)

    # Step 5: Prepare prompt
    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=question
    )

    prompt += f"""

    PREVIOUS CONVERSATION:
    {conversation_context}

    Use conversation history only to understand
    follow-up questions.

    Use only retrieved university documents
    for factual answers.
    """

    # Step 6: Generate answer
    llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Step 7: Store messages in this session only
    memory.add_message("user", question)
    memory.add_message("assistant", answer)

    # Step 8: Remove duplicate sources
    unique_sources = []
    seen_sources = set()

    for source in sources:

        key = (source["source"], source["page"])

        if key not in seen_sources:
            seen_sources.add(key)
            unique_sources.append(source)

    return {
        "answer": answer,
        "sources": unique_sources
    }
