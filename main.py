import os
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI

# Mock Enterprise Document Loader
# In a real scenario, this connects to Azure Blob Storage
def load_enterprise_docs():
    print("Loading enterprise knowledge base...")
    # Simulating a document for RAG
    with open("company_policy.txt", "w") as f:
        f.write("Microsoft Copilot uses advanced LLMs to improve productivity.")
    return TextLoader("company_policy.txt")

def run_rag_pipeline(query):
    # 1. Ingestion
    loader = load_enterprise_docs()
    
    # 2. Embedding & Vector Storage (Simulating Pinecone/Chroma)
    print("Vectorizing documents...")
    index = VectorstoreIndexCreator().from_loaders([loader])
    
    # 3. Retrieval & Generation
    print(f"Querying: {query}")
    # Note: Requires OPENAI_API_KEY env var to actually run, 
    # but the code structure proves the logic.
    response = index.query(query)
    print(f"Response: {response}")

if __name__ == "__main__":
    run_rag_pipeline("How does Copilot help users?")
