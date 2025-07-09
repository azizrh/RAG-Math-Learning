# import argparse
# import os
# import shutil
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from get_embedding_function import get_embedding_function
# from langchain.vectorstores.chroma import Chroma


# CHROMA_PATH = "chroma"
# DATA_PATH = "data"


# def main():
#     # Check if the database should be cleared (using the --clear flag).
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         print("✨ Clearing Database")
#         clear_database()

#     # Create (or update) the data store.
#     documents = load_documents()
#     chunks = split_documents(documents)
#     add_to_chroma(chunks)


# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)


# def add_to_chroma(chunks: list[Document]):
#     # Load the existing database.
#     db = Chroma(
#         persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
#     )

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)

#     # Add or Update the documents.
#     existing_items = db.get(include=[])  # IDs are always included by default
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         db.persist()
#     else:
#         print("✅ No new documents to add")


# def calculate_chunk_ids(chunks):
#     # This will create IDs like "data/monopoly.pdf:6:2"
#     # Page Source : Page Number : Chunk Index
#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id

#     return chunks


# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)


# if __name__ == "__main__":
#     main()

import argparse
import os
import shutil
import requests
import json
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Ollama API configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

class OllamaAPI:
    def __init__(self, base_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Warning: Ollama API call failed: {e}")
            return ""

class GemmaChunker:
    def __init__(self, ollama_api: OllamaAPI, max_chunk_size=800):
        self.ollama_api = ollama_api
        self.max_chunk_size = max_chunk_size
    
    def semantic_chunk(self, text):
        """Use Gemma to create semantically coherent chunks"""
        print(f"  🤖 Using Gemma for semantic chunking...")
        
        # Split text into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if len(potential_chunk) > self.max_chunk_size and current_chunk:
                # Ask Gemma if this is a good place to break
                if self._should_break_here(current_chunk, sentence):
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # If Gemma says no, but we're over limit, break anyway
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        return chunks
    
    def topic_based_chunk(self, text):
        """Use Gemma to identify topics and chunk accordingly"""
        print(f"  🧠 Using Gemma for topic-based chunking...")
        
        # Ask Gemma to identify main topics
        topic_prompt = f"""
        Analyze this text and identify natural breaking points based on topic changes.
        Look for shifts in subject matter, new concepts being introduced, or logical divisions.
        
        Provide the first few words (3-5 words) of each new section/topic, one per line.
        
        Text: {text[:2000]}{'...' if len(text) > 2000 else ''}
        """
        
        try:
            response = self.ollama_api.generate(topic_prompt)
            break_phrases = self._parse_break_phrases(response)
            
            if break_phrases:
                chunks = self._create_topic_chunks(text, break_phrases)
                return chunks
        except Exception as e:
            print(f"    Warning: Topic-based chunking failed, falling back to semantic: {e}")
        
        # Fallback to semantic chunking
        return self.semantic_chunk(text)
    
    def _should_break_here(self, current_chunk, next_sentence):
        """Ask Gemma if this is a good place to break"""
        
        # Only ask Gemma for longer chunks to avoid too many API calls
        if len(current_chunk) < 400:
            return False
        
        prompt = f"""
        Should we end the current text chunk here? Consider topic coherence and logical flow.
        
        Current chunk ending: ...{current_chunk[-150:]}
        Next sentence: {next_sentence[:100]}
        
        Answer only "YES" or "NO":
        """
        
        try:
            response = self.ollama_api.generate(prompt).strip().upper()
            return "YES" in response
        except:
            return len(current_chunk) > self.max_chunk_size * 0.8  # Fallback logic
    
    def _split_into_sentences(self, text):
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _parse_break_phrases(self, response):
        """Parse break phrases from Gemma response"""
        lines = response.strip().split('\n')
        phrases = []
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 5:  # 5 words or less
                phrases.append(line.lower())
        return phrases
    
    def _create_topic_chunks(self, text, break_phrases):
        """Create chunks based on topic break phrases"""
        chunks = []
        current_pos = 0
        text_lower = text.lower()
        
        for phrase in break_phrases:
            # Find where this phrase appears
            pos = text_lower.find(phrase, current_pos)
            if pos != -1 and pos > current_pos:
                # Add chunk from current position to phrase start
                chunk = text[current_pos:pos].strip()
                if len(chunk) > 50:  # Only add substantial chunks
                    chunks.append(chunk)
                current_pos = pos
        
        # Add remaining text
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if len(remaining) > 50:
                chunks.append(remaining)
        
        # If no good chunks were created, fall back
        if not chunks:
            return [text]
        
        return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--use-gemma", action="store_true", 
                       help="Use Gemma for intelligent chunking instead of default text splitter")
    parser.add_argument("--gemma-method", choices=["semantic", "topic"], default="semantic",
                       help="Gemma chunking method to use (default: semantic)")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_API_URL,
                       help=f"Ollama API base URL (default: {OLLAMA_API_URL})")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL,
                       help=f"Ollama model to use (default: {OLLAMA_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=800,
                       help="Maximum chunk size (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=80,
                       help="Chunk overlap for traditional splitter (default: 80)")
    
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    # Test Ollama connection if using Gemma
    if args.use_gemma:
        print(f"🔍 Testing Ollama connection at {args.ollama_url}...")
        ollama_api = OllamaAPI(base_url=args.ollama_url, model=args.model)
        test_response = ollama_api.generate("Hello")
        if not test_response:
            print("❌ Cannot connect to Ollama. Falling back to traditional chunking.")
            args.use_gemma = False
        else:
            print(f"✅ Connected to Ollama using model: {args.model}")
    
    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents, args)
    add_to_chroma(chunks)

def load_documents():
    print(f"📚 Loading documents from {DATA_PATH}...")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"📄 Loaded {len(documents)} document pages")
    return documents

def split_documents(documents: list[Document], args):
    print(f"✂️ Splitting documents into chunks...")
    
    if args.use_gemma:
        return split_documents_with_gemma(documents, args)
    else:
        return split_documents_traditional(documents, args)

def split_documents_traditional(documents: list[Document], args):
    """Traditional recursive character text splitting"""
    print(f"  📝 Using traditional text splitter (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  ✅ Created {len(chunks)} chunks using traditional method")
    return chunks

def split_documents_with_gemma(documents: list[Document], args):
    """Gemma-powered intelligent chunking"""
    print(f"  🤖 Using Gemma for intelligent chunking (method: {args.gemma_method})")
    
    ollama_api = OllamaAPI(base_url=args.ollama_url, model=args.model)
    gemma_chunker = GemmaChunker(ollama_api, max_chunk_size=args.chunk_size)
    
    all_chunks = []
    
    for i, doc in enumerate(documents):
        print(f"  📄 Processing document {i+1}/{len(documents)}: {doc.metadata.get('source', 'Unknown')}")
        
        try:
            # Choose chunking method
            if args.gemma_method == "topic":
                text_chunks = gemma_chunker.topic_based_chunk(doc.page_content)
            else:  # semantic
                text_chunks = gemma_chunker.semantic_chunk(doc.page_content)
            
            print(f"    ✂️ Created {len(text_chunks)} chunks")
            
            # Convert text chunks back to Document objects
            for j, chunk_text in enumerate(text_chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_method": f"gemma_{args.gemma_method}",
                        "chunk_index": j,
                        "original_chunk_count": len(text_chunks)
                    }
                )
                all_chunks.append(chunk_doc)
                
        except Exception as e:
            print(f"    ⚠️ Error processing document: {e}")
            print(f"    🔄 Falling back to traditional chunking for this document")
            
            # Fallback to traditional chunking for this document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            fallback_chunks = text_splitter.split_documents([doc])
            
            # Mark these chunks as fallback
            for chunk in fallback_chunks:
                chunk.metadata["chunk_method"] = "traditional_fallback"
            
            all_chunks.extend(fallback_chunks)
    
    print(f"  ✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

def add_to_chroma(chunks: list[Document]):
    print(f"💾 Adding chunks to Chroma database...")
    
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(OLLAMA_API_URL)
    )
    
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"📊 Number of existing documents in DB: {len(existing_ids)}")
    
    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        
        # Print chunking method statistics
        method_stats = {}
        for chunk in new_chunks:
            method = chunk.metadata.get("chunk_method", "traditional")
            method_stats[method] = method_stats.get(method, 0) + 1
        
        print("📈 Chunking method statistics:")
        for method, count in method_stats.items():
            print(f"  {method}: {count} chunks")
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        
        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()