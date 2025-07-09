from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"

# Ollama API configuration from environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
def get_embedding_function(OLLAMA_API_URL):
    """
    Get embedding function for local use.
    
    Options:
    1. SentenceTransformerEmbeddings - Uses sentence-transformers library (recommended)
    2. OllamaEmbeddings - Uses Ollama with an embedding model
    """
    
    # Option 1: Using sentence-transformers (recommended for better performance)
    # This model works well for general text and is relatively lightweight
    # return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Option 2: Using Ollama with an embedding model (uncomment to use)
    # Make sure you have pulled an embedding model in Ollama first:
    # ollama pull nomic-embed-text
    # return OllamaEmbeddings(model="nomic-embed-text",base_url=OLLAMA_API_URL)
    return OllamaEmbeddings(model="mxbai-embed-large",base_url=OLLAMA_API_URL)