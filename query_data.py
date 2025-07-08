# import argparse
# from langchain.vectorstores.chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama

# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"

# # Different prompts for different query types
# SPECIFIC_PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# GENERAL_PROMPT_TEMPLATE = """
# Based on the following document excerpts, provide a comprehensive overview of what the documents are about:

# {context}

# ---

# Question: {question}

# Please provide a summary of the main topics, themes, and key information covered in these documents.
# """

# def detect_general_query(query_text: str) -> bool:
#     """Detect if the query is asking for a general overview"""
#     general_keywords = [
#         "what is this about", "what's this about", "what are these documents about",
#         "summarize", "overview", "main topic", "what do these documents discuss",
#         "what is the subject", "what is covered", "general information"
#     ]
    
#     query_lower = query_text.lower()
#     return any(keyword in query_lower for keyword in general_keywords)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)

# def query_rag(query_text: str):
#     # Prepare the DB
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Check if it's a general query
#     is_general = detect_general_query(query_text)
    
#     if is_general:
#         # For general queries, get more diverse chunks
#         results = db.similarity_search_with_score(query_text, k=10)
        
#         # Also try to get some random samples to ensure variety
#         try:
#             all_docs = db.get()
#             if len(all_docs['documents']) > 10:
#                 # Get a broader sample of documents
#                 import random
#                 random.seed(42)  # For reproducible results
#                 sample_indices = random.sample(range(len(all_docs['documents'])), min(5, len(all_docs['documents'])))
#                 sample_docs = [all_docs['documents'][i] for i in sample_indices]
                
#                 # Combine with similarity search results
#                 context_docs = [doc.page_content for doc, _score in results[:5]]
#                 context_docs.extend(sample_docs)
#             else:
#                 context_docs = [doc.page_content for doc, _score in results]
#         except:
#             # Fallback to just similarity search
#             context_docs = [doc.page_content for doc, _score in results]
        
#         context_text = "\n\n---\n\n".join(context_docs)
#         prompt_template = ChatPromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
#     else:
#         # For specific queries, use normal retrieval
#         results = db.similarity_search_with_score(query_text, k=5)
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#         prompt_template = ChatPromptTemplate.from_template(SPECIFIC_PROMPT_TEMPLATE)

#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # Initialize Ollama with Gemma model
#     model = Ollama(model="gemma3:4b")
    
#     # Generate response using Gemma
#     response = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response}\nSources: {sources}"
#     print(formatted_response)
#     return response

# if __name__ == "__main__":
#     main()

import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from typing import List, Set
import re

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

# Multi-query generation prompt
MULTI_QUERY_PROMPT = """
You are an AI assistant that helps generate multiple search queries to improve information retrieval.

Given the following question: "{question}"

Generate 3 different but related search queries that would help find relevant information. The queries should:
1. Rephrase the original question using different words
2. Break down complex questions into simpler parts
3. Use synonyms or alternative terminology
4. Focus on different aspects of the topic

Return only the 3 queries, one per line, without numbering or additional text.
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

def generate_multiple_queries(original_query: str, model: Ollama) -> List[str]:
    """Generate multiple related queries from the original query"""
    prompt = MULTI_QUERY_PROMPT.format(question=original_query)
    response = model.invoke(prompt)
    
    # Parse the response to extract individual queries
    queries = []
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line:
            # Remove any numbering or bullet points
            cleaned_line = re.sub(r'^\d+\.?\s*', '', line)
            cleaned_line = re.sub(r'^[-•*]\s*', '', cleaned_line)
            if cleaned_line and cleaned_line not in queries:
                queries.append(cleaned_line)
    
    # Include the original query
    all_queries = [original_query] + queries
    return all_queries[:4]  # Limit to 4 queries total (original + 3 generated)

def retrieve_documents_multiquery(db: Chroma, queries: List[str], k_per_query: int = 3) -> List:
    """Retrieve documents for multiple queries and combine results"""
    all_results = []
    seen_docs = set()
    
    for query in queries:
        results = db.similarity_search_with_score(query, k=k_per_query)
        
        for doc, score in results:
            # Use document content as a simple way to deduplicate
            doc_key = doc.page_content[:100]  # First 100 chars as key
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                all_results.append((doc, score, query))  # Include which query retrieved this
    
    # Sort by score (lower is better for L2 distance)
    all_results.sort(key=lambda x: x[1])
    
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--disable-multiquery", action="store_true", 
                       help="Disable multi-query generation and use original single query")
    args = parser.parse_args()
    
    query_text = args.query_text
    use_multiquery = not args.disable_multiquery
    
    query_rag(query_text, use_multiquery)

def query_rag(query_text: str, use_multiquery: bool = True):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Initialize Ollama with Gemma model
    model = Ollama(model="gemma3:4b")
    
    # Check if it's a general query
    is_general = detect_general_query(query_text)
    
    if use_multiquery and not is_general:
        # Generate multiple queries for specific questions
        print("Generating multiple queries...")
        queries = generate_multiple_queries(query_text, model)
        print(f"Generated queries: {queries}")
        
        # Retrieve documents using all queries
        all_results = retrieve_documents_multiquery(db, queries, k_per_query=2)
        
        # Limit total results
        results = all_results[:8]  # Top 8 results across all queries
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score, _query in results])
        
        # Keep track of which queries found results
        sources = []
        for doc, _score, source_query in results:
            source_id = doc.metadata.get("id", "Unknown")
            sources.append(f"{source_id} (via: {source_query[:50]}...)")
        
    elif is_general:
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
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        
    else:
        # Original single query approach
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Choose appropriate prompt template
    if is_general:
        prompt_template = ChatPromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
    else:
        prompt_template = ChatPromptTemplate.from_template(SPECIFIC_PROMPT_TEMPLATE)
    
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate response using Gemma
    response = model.invoke(prompt)
    
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response

if __name__ == "__main__":
    main()