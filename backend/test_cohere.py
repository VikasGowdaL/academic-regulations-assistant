
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
import os

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("Cohere API key not found!")

# Initialize Cohere LLM
llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
    cohere_api_key=api_key
)

# Test LLM
response = llm.invoke(
    "Explain Retrieval Augmented Generation in two sentences."
)

print("LLM RESPONSE:")
print(response.content)

# Initialize Cohere Embeddings
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=api_key
)

# Test embeddings
vector = embeddings.embed_query(
    "Academic examination regulations"
)

print("\nEMBEDDING TEST:")
print("Vector dimensions:", len(vector))
print("First five values:", vector[:5])
