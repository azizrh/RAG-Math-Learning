from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    """
    Get embedding function for local use.
    
    Options:
    1. SentenceTransformerEmbeddings - Uses sentence-transformers library (recommended)
    2. OllamaEmbeddings - Uses Ollama with an embedding model
    """
    
    # Option 1: Using sentence-transformers (recommended for better performance)
    # This model works well for general text and is relatively lightweight
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Option 2: Using Ollama with an embedding model (uncomment to use)
    # Make sure you have pulled an embedding model in Ollama first:
    # ollama pull nomic-embed-text
    # return OllamaEmbeddings(model="nomic-embed-text")