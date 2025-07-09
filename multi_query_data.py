
# import argparse
# from langchain.vectorstores.chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from get_embedding_function import get_embedding_function
# from typing import List, Set
# import re

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

# # Multi-query generation prompt
# MULTI_QUERY_PROMPT = """
# You are an AI assistant that helps generate multiple search queries to improve information retrieval.

# Given the following question: "{question}"

# Generate 3 different but related search queries that would help find relevant information. The queries should:
# 1. Rephrase the original question using different words
# 2. Break down complex questions into simpler parts
# 3. Use synonyms or alternative terminology
# 4. Focus on different aspects of the topic

# Return only the 3 queries, one per line, without numbering or additional text.
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

# def generate_multiple_queries(original_query: str, model: Ollama) -> List[str]:
#     """Generate multiple related queries from the original query"""
#     prompt = MULTI_QUERY_PROMPT.format(question=original_query)
#     response = model.invoke(prompt)
    
#     # Parse the response to extract individual queries
#     queries = []
#     lines = response.strip().split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if line:
#             # Remove any numbering or bullet points
#             cleaned_line = re.sub(r'^\d+\.?\s*', '', line)
#             cleaned_line = re.sub(r'^[-•*]\s*', '', cleaned_line)
#             if cleaned_line and cleaned_line not in queries:
#                 queries.append(cleaned_line)
    
#     # Include the original query
#     all_queries = [original_query] + queries
#     return all_queries[:4]  # Limit to 4 queries total (original + 3 generated)

# def retrieve_documents_multiquery(db: Chroma, queries: List[str], k_per_query: int = 3) -> List:
#     """Retrieve documents for multiple queries and combine results"""
#     all_results = []
#     seen_docs = set()
    
#     for query in queries:
#         results = db.similarity_search_with_score(query, k=k_per_query)
        
#         for doc, score in results:
#             # Use document content as a simple way to deduplicate
#             doc_key = doc.page_content[:100]  # First 100 chars as key
#             if doc_key not in seen_docs:
#                 seen_docs.add(doc_key)
#                 all_results.append((doc, score, query))  # Include which query retrieved this
    
#     # Sort by score (lower is better for L2 distance)
#     all_results.sort(key=lambda x: x[1])
    
#     return all_results

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     parser.add_argument("--disable-multiquery", action="store_true", 
#                        help="Disable multi-query generation and use original single query")
#     args = parser.parse_args()
    
#     query_text = args.query_text
#     use_multiquery = not args.disable_multiquery
    
#     query_rag(query_text, use_multiquery)

# def query_rag(query_text: str, use_multiquery: bool = True):
#     # Prepare the DB
#     embedding_function = get_embedding_function(OLLAMA_API_URL)
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
#     # Initialize Ollama with Gemma model
#     model = Ollama(model="gemma3:4b")
    
#     # Check if it's a general query
#     is_general = detect_general_query(query_text)
    
#     if use_multiquery and not is_general:
#         # Generate multiple queries for specific questions
#         print("Generating multiple queries...")
#         queries = generate_multiple_queries(query_text, model)
#         print(f"Generated queries: {queries}")
        
#         # Retrieve documents using all queries
#         all_results = retrieve_documents_multiquery(db, queries, k_per_query=2)
        
#         # Limit total results
#         results = all_results[:8]  # Top 8 results across all queries
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _score, _query in results])
        
#         # Keep track of which queries found results
#         sources = []
#         for doc, _score, source_query in results:
#             source_id = doc.metadata.get("id", "Unknown")
#             sources.append(f"{source_id} (via: {source_query[:50]}...)")
        
#     elif is_general:
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
#         sources = [doc.metadata.get("id", None) for doc, _score in results]
        
#     else:
#         # Original single query approach
#         results = db.similarity_search_with_score(query_text, k=5)
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#         sources = [doc.metadata.get("id", None) for doc, _score in results]
    
#     # Choose appropriate prompt template
#     if is_general:
#         prompt_template = ChatPromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
#     else:
#         prompt_template = ChatPromptTemplate.from_template(SPECIFIC_PROMPT_TEMPLATE)
    
#     prompt = prompt_template.format(context=context_text, question=query_text)
    
#     # Generate response using Gemma
#     response = model.invoke(prompt)
    
#     formatted_response = f"Response: {response}\nSources: {sources}"
#     print(formatted_response)
#     return response

# if __name__ == "__main__":
#     main()

import argparse
import requests
import json
import os

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from typing import List, Set
import re
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"

# Ollama API configuration from environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

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

class OllamaAPI:
    def __init__(self, base_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate response using Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(self.generate_url, json=payload)
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "response" in json_response:
                            full_response += json_response["response"]
                        if json_response.get("done", False):
                            break
                return full_response
            else:
                # Handle non-streaming response
                return response.json()["response"]
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return "Error: Unable to generate response"
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return "Error: Invalid response format"
    
    def chat(self, messages: List[dict], stream: bool = False) -> str:
        """Chat using Ollama API with message history"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(self.chat_url, json=payload)
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "message" in json_response and "content" in json_response["message"]:
                            full_response += json_response["message"]["content"]
                        if json_response.get("done", False):
                            break
                return full_response
            else:
                # Handle non-streaming response
                return response.json()["message"]["content"]
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return "Error: Unable to generate response"
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return "Error: Invalid response format"
    
    def invoke(self, prompt: str) -> str:
        """Compatibility method that mimics LangChain's invoke"""
        return self.generate(prompt)

def detect_general_query(query_text: str) -> bool:
    """Detect if the query is asking for a general overview"""
    general_keywords = [
        "what is this about", "what's this about", "what are these documents about",
        "summarize", "overview", "main topic", "what do these documents discuss",
        "what is the subject", "what is covered", "general information"
    ]
   
    query_lower = query_text.lower()
    return any(keyword in query_lower for keyword in general_keywords)

def generate_multiple_queries(original_query: str, ollama_api: OllamaAPI) -> List[str]:
    """Generate multiple related queries from the original query"""
    prompt = MULTI_QUERY_PROMPT.format(question=original_query)
    response = ollama_api.generate(prompt)
    
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
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_API_URL,
                       help=f"Ollama API base URL (default: {OLLAMA_API_URL})")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL,
                       help=f"Ollama model to use (default: {OLLAMA_MODEL})")
    parser.add_argument("--stream", action="store_true",
                       help="Enable streaming response")
    args = parser.parse_args()
    
    query_text = args.query_text
    use_multiquery = not args.disable_multiquery
    
    query_rag(query_text, use_multiquery, args.ollama_url, args.model, args.stream)

def query_rag(query_text: str, use_multiquery: bool = True, ollama_url: str = OLLAMA_API_URL, 
              model: str = OLLAMA_MODEL, stream: bool = False):
    # Prepare the DB
    embedding_function = get_embedding_function(OLLAMA_API_URL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Initialize Ollama API client
    ollama_api = OllamaAPI(base_url=ollama_url, model=model)
    
    # Check if it's a general query
    is_general = detect_general_query(query_text)
    
    if use_multiquery and not is_general:
        # Generate multiple queries for specific questions
        print("Generating multiple queries...")
        queries = generate_multiple_queries(query_text, ollama_api)
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
    
    # Generate response using Ollama API
    if stream:
        print("Response: ", end="", flush=True)
        response = ollama_api.generate(prompt, stream=True)
        print(response)
    else:
        response = ollama_api.generate(prompt)
        print(f"Response: {response}")
    
    print(f"Sources: {sources}")
    return response

if __name__ == "__main__":
    main()