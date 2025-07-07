import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Different prompts for different query types
SPECIFIC_PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

GENERAL_PROMPT_TEMPLATE = """
Based on the following document excerpts, provide a comprehensive overview of what the documents are about:

{context}

---

Question: {question}

Please provide a summary of the main topics, themes, and key information covered in these documents.
"""

def detect_general_query(query_text: str) -> bool:
    """Detect if the query is asking for a general overview"""
    general_keywords = [
        "what is this about", "what's this about", "what are these documents about",
        "summarize", "overview", "main topic", "what do these documents discuss",
        "what is the subject", "what is covered", "general information"
    ]
    
    query_lower = query_text.lower()
    return any(keyword in query_lower for keyword in general_keywords)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Check if it's a general query
    is_general = detect_general_query(query_text)
    
    if is_general:
        # For general queries, get more diverse chunks
        results = db.similarity_search_with_score(query_text, k=10)
        
        # Also try to get some random samples to ensure variety
        try:
            all_docs = db.get()
            if len(all_docs['documents']) > 10:
                # Get a broader sample of documents
                import random
                random.seed(42)  # For reproducible results
                sample_indices = random.sample(range(len(all_docs['documents'])), min(5, len(all_docs['documents'])))
                sample_docs = [all_docs['documents'][i] for i in sample_indices]
                
                # Combine with similarity search results
                context_docs = [doc.page_content for doc, _score in results[:5]]
                context_docs.extend(sample_docs)
            else:
                context_docs = [doc.page_content for doc, _score in results]
        except:
            # Fallback to just similarity search
            context_docs = [doc.page_content for doc, _score in results]
        
        context_text = "\n\n---\n\n".join(context_docs)
        prompt_template = ChatPromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
    else:
        # For specific queries, use normal retrieval
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(SPECIFIC_PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize Ollama with Gemma model
    model = Ollama(model="gemma:2b")
    
    # Generate response using Gemma
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response

if __name__ == "__main__":
    main()