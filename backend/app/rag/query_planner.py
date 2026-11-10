
import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere

load_dotenv()


def plan_queries(question):

    llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    prompt = f"""
    You are a query planning assistant for a university
    academic regulations system.

    Break the student's question into smaller,
    independent search queries.

    Rules:
    1. If the question contains multiple parts,
       create separate queries for each part.
    2. If it contains only one question,
       return that question unchanged.
    3. Do not answer the questions.
    4. Return only the queries, one per line.
    5. Do not include numbering or explanations.

    Student question:
    {question}
    """

    response = llm.invoke(prompt)

    queries = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    return queries


if __name__ == "__main__":

    question = (
        "What is the attendance requirement, "
        "what happens if I fail to meet it, "
        "and can I take the examination?"
    )

    print("\nORIGINAL QUESTION:")
    print(question)

    print("\nPLANNED QUERIES:")

    queries = plan_queries(question)

    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")
