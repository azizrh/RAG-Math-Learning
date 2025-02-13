from langchain_core.embeddings import Embeddings
from google import genai

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search documents."""
        embeddings = []
        
        for text in texts:
            result = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            # Extract the values from ContentEmbedding object
            embedding_values = result.embeddings[0].values
            embeddings.append(embedding_values)
            
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        result = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        # Extract the values from ContentEmbedding object
        return result.embeddings[0].values

def get_embedding_function():
    api_key = open("api","r").read()  # Replace with your actual API key
    return GeminiEmbeddings(api_key=api_key)