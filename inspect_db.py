import sqlite3
import pandas as pd
import json

def get_document_info(db_path, doc_id=102):
    """
    Get all available information about a specific document from ChromaDB
    """
    conn = sqlite3.connect(db_path)
    
    # Query all relevant tables
    query = """
    SELECT 
        eq.id as queue_id,
        eq.metadata,
        fs.c0 as fulltext_content,
        em.key as metadata_key,
        em.string_value as metadata_value
    FROM embeddings_queue eq
    LEFT JOIN embedding_fulltext_search_content fs ON eq.id = fs.id
    LEFT JOIN embedding_metadata em ON eq.id = em.id
    WHERE eq.id = ?
    """
    
    try:
        # Get the data
        df = pd.read_sql_query(query, conn, params=[doc_id])
        
        print("\n=== Document Information ===")
        print(f"Document ID: {doc_id}")
        
        # Print fulltext content if available
        if not df.empty and 'fulltext_content' in df.columns:
            content = df['fulltext_content'].iloc[0]
            print("\n--- Document Content ---")
            print(content if pd.notna(content) else "No content available")
        
        # Print metadata
        if not df.empty and 'metadata' in df.columns:
            print("\n--- Queue Metadata ---")
            metadata = df['metadata'].iloc[0]
            if pd.notna(metadata):
                try:
                    metadata_dict = json.loads(metadata)
                    for key, value in metadata_dict.items():
                        print(f"{key}: {value}")
                except json.JSONDecodeError:
                    print("Could not parse metadata JSON")
        
        # Print additional metadata from embedding_metadata table
        if not df.empty:
            print("\n--- Additional Metadata ---")
            metadata_pairs = df[['metadata_key', 'metadata_value']].dropna()
            for _, row in metadata_pairs.iterrows():
                print(f"{row['metadata_key']}: {row['metadata_value']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    # Replace with your ChromaDB SQLite file path
    DB_PATH = r"C:\Users\ajizr\Documents\RAG\rag-tutorial-v2\chroma\chroma.sqlite3"
    inspect_database(DB_PATH)