import sqlite3
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import json

def load_embeddings_from_chroma(db_path):
    """
    Extract embeddings from ChromaDB SQLite database
    """
    print("Connecting to database...")
    try:
        conn = sqlite3.connect(db_path)
        
        # Query to get embeddings and metadata from embeddings_queue table
        query = """
        SELECT 
            eq.vector as embedding,
            eq.metadata,
            eq.id
        FROM embeddings_queue eq
        WHERE eq.operation = 2  -- Only get INSERT operations
        """
        
        print("Executing query...")
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print("Processing embeddings...")
        # Convert binary vector data to numpy arrays
        df['embedding_array'] = df['embedding'].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        
        # Parse metadata JSON
        print("Parsing metadata...")
        df['metadata_dict'] = df['metadata'].apply(json.loads)
        
        print(f"Found {len(df)} embeddings")
        print("Column names in dataframe:", df.columns.tolist())
        print("First embedding shape:", len(df['embedding_array'].iloc[0]) if len(df) > 0 else "No embeddings found")
        
        return df
        
    except Exception as e:
        print("Error loading embeddings:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

def reduce_dimensions(embeddings, n_components=2):
    """
    Reduce dimensionality of embeddings using t-SNE
    """
    tsne = TSNE(n_components=n_components, 
                random_state=42, 
                perplexity=5,
                n_iter=1000)
    embeddings_2d = tsne.fit_transform(np.stack(embeddings))
    return embeddings_2d

def calculate_cosine_similarities(embeddings):
    """
    Calculate pairwise cosine similarities between all embeddings
    """
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i,j] = 1 - cosine(embeddings[i], embeddings[j])
    
    return similarity_matrix

def visualize_embeddings(df, embeddings_2d, similarities):
    """
    Create visualizations of the embeddings and similarities
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Scatter plot of reduced dimensions
    scatter = ax1.scatter(embeddings_2d[:, 0], 
                         embeddings_2d[:, 1],
                         alpha=0.6)
    ax1.set_title('t-SNE Visualization of Document Embeddings')
    
    # Add document labels using first 30 chars of ID
    for i, txt in enumerate(df['id']):
        label = str(txt)[:30] + '...' if len(str(txt)) > 30 else str(txt)
        ax1.annotate(f'Doc {i}\n{label}', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7)
    
    # Heatmap of cosine similarities
    sns.heatmap(similarities, 
                ax=ax2, 
                cmap='YlOrRd', 
                vmin=0, 
                vmax=1,
                xticklabels=[f'Doc {i}' for i in range(len(similarities))],
                yticklabels=[f'Doc {i}' for i in range(len(similarities))])
    ax2.set_title('Cosine Similarity Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_embeddings(df, similarities):
    """
    Print analysis of the embeddings and similarities
    """
    print("\nEmbedding Analysis:")
    print(f"Number of documents: {len(df)}")
    print(f"Embedding dimension: {len(df['embedding_array'].iloc[0])}")
    
    # Find most similar pairs
    n = len(similarities)
    most_similar = []
    for i in range(n):
        for j in range(i+1, n):
            most_similar.append((i, j, similarities[i,j]))
    
    most_similar.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 Most Similar Document Pairs:")
    for i, j, sim in most_similar[:5]:
        print(f"Doc {i} & Doc {j}: {sim:.3f} similarity")
        print(f"Doc {i} ID: {df['id'].iloc[i]}")
        print(f"Doc {j} ID: {df['id'].iloc[j]}\n")

def main(db_path):
    """
    Main function to run the visualization pipeline
    """
    # Load embeddings
    print("Loading embeddings from ChromaDB...")
    df = load_embeddings_from_chroma(db_path)
    
    # Reduce dimensions
    print("Reducing dimensions with t-SNE...")
    embeddings_2d = reduce_dimensions(df['embedding_array'].values)
    
    # Calculate similarities
    print("Calculating cosine similarities...")
    similarities = calculate_cosine_similarities(df['embedding_array'].values)
    
    # Create visualization
    print("Creating visualization...")
    fig = visualize_embeddings(df, embeddings_2d, similarities)
    
    # Print analysis
    analyze_embeddings(df, similarities)
    
    return df, embeddings_2d, similarities, fig

if __name__ == "__main__":
    # Replace with your ChromaDB SQLite file path
    DB_PATH = r"C:\Users\ajizr\Documents\RAG\rag-tutorial-v2\chroma\chroma.sqlite3"
    df, embeddings_2d, similarities, fig = main(DB_PATH)