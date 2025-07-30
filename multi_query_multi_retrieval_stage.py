import argparse
import requests
import json
import os
from collections import Counter, defaultdict
import re
from typing import List, Set, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
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

class TerminologyAnalyzer:
    """Analyzes documents to discover domain-specific terminology patterns"""
    
    def __init__(self):
        self.term_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.term_frequency = defaultdict(int)
        self.context_patterns = {}
        
    def extract_key_terms(self, documents: List[str], top_k: int = 20) -> List[str]:
        """Extract key terms from documents using TF-IDF"""
        # Clean and preprocess texts
        cleaned_docs = []
        for doc in documents:
            # Remove special characters, keep alphanumeric and spaces
            cleaned = re.sub(r'[^\w\s]', ' ', doc.lower())
            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())
            cleaned_docs.append(cleaned)
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=2,  # Term must appear in at least 2 documents
            max_df=0.8  # Term shouldn't appear in more than 80% of documents
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores for each term
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Sort terms by importance
            term_scores = list(zip(feature_names, mean_scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [term for term, score in term_scores[:top_k]]
        except:
            # Fallback: simple word frequency
            all_words = []
            for doc in cleaned_docs:
                words = doc.split()
                all_words.extend([w for w in words if len(w) > 3])
            
            word_freq = Counter(all_words)
            return [word for word, freq in word_freq.most_common(top_k)]
    
    def find_related_terms(self, query_terms: List[str], candidate_docs: List[str]) -> Dict[str, List[str]]:
        """Find terms that appear in similar contexts to query terms"""
        related_terms = defaultdict(set)
        
        for doc in candidate_docs:
            words = re.findall(r'\b\w+\b', doc.lower())
            
            # Find co-occurring terms within a window
            window_size = 5
            for i, word in enumerate(words):
                if word in query_terms:
                    # Look at surrounding words
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    context_words = words[start:end]
                    
                    for context_word in context_words:
                        if (context_word != word and 
                            len(context_word) > 3 and 
                            context_word.isalpha()):
                            related_terms[word].add(context_word)
        
        # Convert sets to lists and filter by frequency
        filtered_related = {}
        for term, related_set in related_terms.items():
            # Keep only terms that appear multiple times
            related_list = list(related_set)
            if len(related_list) > 0:
                filtered_related[term] = related_list[:10]  # Top 10 related terms
        
        return filtered_related
    
    def analyze_terminology_patterns(self, documents: List[str], query: str) -> Dict[str, any]:
        """Analyze documents to find terminology patterns relevant to the query"""
        
        # Extract key terms from the corpus
        key_terms = self.extract_key_terms(documents)
        
        # Extract query terms
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_words = [w for w in query_words if len(w) > 3]
        
        # Find terms related to query terms
        related_terms = self.find_related_terms(query_words, documents)
        
        # Find domain-specific patterns
        domain_terms = []
        for term in key_terms:
            # Look for technical terms, acronyms, or specialized vocabulary
            if (len(term) > 6 or  # Longer terms are often technical
                term.isupper() or  # Acronyms
                '_' in term or     # Technical identifiers
                any(char.isdigit() for char in term)):  # Terms with numbers
                domain_terms.append(term)
        
        return {
            'key_terms': key_terms,
            'query_terms': query_words,
            'related_terms': related_terms,
            'domain_terms': domain_terms,
            'expansion_candidates': self._generate_expansion_candidates(query_words, related_terms, key_terms)
        }
    
    def _generate_expansion_candidates(self, query_terms: List[str], 
                                     related_terms: Dict[str, List[str]], 
                                     key_terms: List[str]) -> List[str]:
        """Generate query expansion candidates based on discovered patterns"""
        candidates = set(query_terms)
        
        # Add related terms
        for term in query_terms:
            if term in related_terms:
                candidates.update(related_terms[term][:3])  # Top 3 related terms
        
        # Add relevant key terms that might be synonyms
        for key_term in key_terms[:10]:  # Top 10 key terms
            # Simple heuristic: if key term shares characters with query terms
            for query_term in query_terms:
                if (len(set(key_term) & set(query_term)) >= 3 or  # Shared characters
                    any(qt in key_term or key_term in qt for qt in query_terms)):
                    candidates.add(key_term)
        
        return list(candidates)

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
    
    def invoke(self, prompt: str) -> str:
        """Compatibility method that mimics LangChain's invoke"""
        return self.generate(prompt)

def retrieve_documents_multistage_uncertain(db: Chroma, query: str, k_final: int = 5) -> List[Tuple]:
    """
    Multi-Stage Retrieval with Uncertainty for unknown domains
    
    Stage 1: Broad semantic search to gather candidates
    Stage 2: Analyze candidates for terminology patterns
    Stage 3: Expand query based on discovered patterns
    Stage 4: Refined retrieval with expanded query
    """
    
    print(f"🔍 Starting multi-stage retrieval for: '{query}'")
    
    # Stage 1: Broad semantic search to gather candidates
    print("📊 Stage 1: Broad semantic search...")
    broad_k = min(50, db._collection.count())  # Don't exceed collection size
    stage1_results = db.similarity_search_with_score(query, k=broad_k)
    
    if not stage1_results:
        print("⚠️  No results found in Stage 1")
        return []
    
    print(f"   Found {len(stage1_results)} candidates")
    
    # Stage 2: Analyze candidates for terminology patterns
    print("🧠 Stage 2: Analyzing terminology patterns...")
    candidate_texts = [doc.page_content for doc, score in stage1_results]
    
    analyzer = TerminologyAnalyzer()
    terminology_analysis = analyzer.analyze_terminology_patterns(candidate_texts, query)
    
    print(f"   Key terms discovered: {terminology_analysis['key_terms'][:5]}...")
    print(f"   Domain terms: {terminology_analysis['domain_terms'][:3]}...")
    print(f"   Expansion candidates: {terminology_analysis['expansion_candidates'][:5]}...")
    
    # Stage 3: Expand query based on discovered patterns
    print("🔧 Stage 3: Query expansion...")
    original_query = query
    expanded_terms = terminology_analysis['expansion_candidates']
    
    # Create multiple query variations
    query_variations = [original_query]
    
    # Add queries with expanded terms
    for term in expanded_terms[:3]:  # Use top 3 expansion candidates
        if term.lower() not in original_query.lower():
            expanded_query = f"{original_query} {term}"
            query_variations.append(expanded_query)
    
    # Add queries with key domain terms
    for domain_term in terminology_analysis['domain_terms'][:2]:
        if domain_term.lower() not in original_query.lower():
            domain_query = f"{original_query} {domain_term}"
            query_variations.append(domain_query)
    
    print(f"   Generated {len(query_variations)} query variations")
    
    # Stage 4: Refined retrieval with expanded queries
    print("🎯 Stage 4: Refined retrieval...")
    all_results = []
    seen_docs = set()
    
    # Weight original query higher
    query_weights = [1.0] + [0.7] * (len(query_variations) - 1)
    
    for i, var_query in enumerate(query_variations):
        weight = query_weights[i]
        k_per_variation = max(1, k_final // len(query_variations))
        
        try:
            results = db.similarity_search_with_score(var_query, k=k_per_variation * 2)
            
            for doc, score in results:
                doc_key = doc.page_content[:100]  # Use first 100 chars as key
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    # Apply query weight to score (lower score is better)
                    weighted_score = score / weight  # Divide to make weighted scores comparable
                    all_results.append((doc, weighted_score, var_query, weight))
        except Exception as e:
            print(f"   Warning: Error with query variation '{var_query}': {e}")
            continue
    
    # Sort by weighted score and apply diversity filtering
    all_results.sort(key=lambda x: x[1])
    
    # Apply diversity filtering to avoid too similar documents
    final_results = []
    for doc, score, source_query, weight in all_results:
        if len(final_results) >= k_final:
            break
            
        # Check if this document is too similar to already selected ones
        is_diverse = True
        for existing_doc, _, _, _ in final_results:
            # Simple diversity check based on content overlap
            doc_words = set(doc.page_content.lower().split())
            existing_words = set(existing_doc.page_content.lower().split())
            overlap = len(doc_words & existing_words) / len(doc_words | existing_words)
            
            if overlap > 0.8:  # Too similar
                is_diverse = False
                break
        
        if is_diverse:
            final_results.append((doc, score, source_query, weight))
    
    print(f"✅ Retrieved {len(final_results)} diverse, relevant documents")
    
    # Return in the expected format (doc, score, source_query)
    return [(doc, score, source_query) for doc, score, source_query, _ in final_results]

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

# Keep the original function for backward compatibility
def retrieve_documents_multiquery(db: Chroma, queries: List[str], k_per_query: int = 3) -> List:
    """Original multi-query retrieval function"""
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
    parser.add_argument("--use-multistage", action="store_true",
                       help="Use multi-stage retrieval with uncertainty (recommended for unknown domains)")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_API_URL,
                       help=f"Ollama API base URL (default: {OLLAMA_API_URL})")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL,
                       help=f"Ollama model to use (default: {OLLAMA_MODEL})")
    parser.add_argument("--stream", action="store_true",
                       help="Enable streaming response")
    args = parser.parse_args()
    
    query_text = args.query_text
    use_multiquery = not args.disable_multiquery
    use_multistage = args.use_multistage
    
    query_rag(query_text, use_multiquery, use_multistage, args.ollama_url, args.model, args.stream)

def query_rag(query_text: str, use_multiquery: bool = True, use_multistage: bool = False,
              ollama_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL, stream: bool = False):
    # Prepare the DB
    embedding_function = get_embedding_function(OLLAMA_API_URL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Initialize Ollama API client
    ollama_api = OllamaAPI(base_url=ollama_url, model=model)
    
    # Check if it's a general query
    is_general = detect_general_query(query_text)
    
    if use_multistage and not is_general:
        print("🚀 Using Multi-Stage Retrieval with Uncertainty...")
        # Use the new multi-stage retrieval
        results = retrieve_documents_multistage_uncertain(db, query_text, k_final=8)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score, _query in results])
        
        # Keep track of which queries found results
        sources = []
        for doc, _score, source_query in results:
            source_id = doc.metadata.get("id", "Unknown")
            sources.append(f"{source_id} (via: {source_query[:50]}...)")
        
    elif use_multiquery and not is_general:
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