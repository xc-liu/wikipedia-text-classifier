from bertopic import BERTopic
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import argparse

def train_topic_model(docs, embeddings, vocab, min_topic_size=500):
    """
    Train a BERTopic model on the given documents and embeddings.
    Args:
        docs (list): List of documents to train the model on.
        embeddings (np.ndarray): Embeddings for the documents.
        vocab (list): Vocabulary to use for the model.
        min_topic_size (int): Minimum size of a topic.
    Returns:
        BERTopic: The trained BERTopic model.
    """
    # Create UMAP model
    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

    # Create HDBSCAN model
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', prediction_data=True)

    # Create CountVectorizer model
    vectorizer_model = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), stop_words="english")

    # Create BERTopic model
    topic_model = BERTopic(
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model, 
        vectorizer_model=vectorizer_model, 
        verbose=True,
        n_gram_range=(1, 2),
        min_topic_size=min_topic_size,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    
    # Fit the model
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    return topic_model

def save_topic_model(topic_model, output_path):
    """
    Save the trained topic model to the specified path.
    Args:
        topic_model (BERTopic): The trained topic model.
        output_path (str): Path to save the model.
    """
    # Save the model
    topic_model.save(
        path=output_path,
        serialization='safetensors',
        save_ctfidf=True,
        save_embedding_model='sentence-transformers/all-MiniLM-L6-v2'
    )
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a BERTopic model on Wikipedia data.")
    parser.add_argument("--output_path", type=str, default="models/wikipedia_topic_model", help="Path to save the trained topic model.")
    args = parser.parse_args()
    output_path = args.output_path

    # Load the data
    with open("data/processed/vocab.txt", "r") as f:
        vocab = f.read().splitlines()
    with open("data/processed/wikipedia_docs.txt", "r") as f:
        docs = f.read().split("\n\n\n")[:-1]  # Remove the last empty string
        
    # Load the embeddings
    embeddings = np.load("data/processed/wikipedia_embeddings.npz")["embeddings"]
    print(f"Embeddings shape: {embeddings.shape}")

    # Train the topic model
    topic_model = train_topic_model(docs, embeddings, vocab)

    # Save the topic model
    save_topic_model(topic_model, output_path=output_path)
    print("Topic model training and saving completed.")