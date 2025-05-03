import collections
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
import argparse

from tqdm import tqdm

def load_wikipedia_dataset(version: str, file_path: str):
    """
    Load the Wikipedia dataset for the specified version.
    Args:
        version (str): The version of the Wikipedia dataset to load.
        file_path (str): The path where the dataset is saved.
    Returns:
        Dataset: The loaded Wikipedia dataset.
    """
    print(f"Loading Wikipedia dataset version {version} from {file_path}...")
    # Load the Wikipedia dataset
    dataset = load_dataset("wikipedia", version, data_dir=file_path)
    print("Dataset loaded successfully.")
    return dataset

def extract_first_paragraph(text):
    """
    Extracts the first paragraph from a given text.
    """
    # Split the text into paragraphs and return the first one
    first_para_end_idx = text.find('\n\n')
    return text[:first_para_end_idx] if first_para_end_idx != -1 else text

def extract_vocab(docs, min_freq=15):
    """
    Extract vocabulary from documents.
    Args:
        docs (list): List of documents to extract vocabulary from.
        min_freq (int): Minimum frequency for a word to be included in the vocabulary.
    Returns:
        list: List of words in the vocabulary.
    """
    # Initialize a Counter to count word frequencies
    vocab = collections.Counter()
    tokenizer = CountVectorizer().build_tokenizer()
    for doc in tqdm(docs):
        vocab.update(tokenizer(doc))
    vocab = [word for word, frequency in vocab.items() if frequency >= min_freq]
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def create_embeddings(docs, model_name="all-MiniLM-L6-v2", output_path="data/processed/wikipedia_embeddings.npz"):
    """
    Create embeddings for the given documents using the specified model.
    Args:
        docs (list): List of documents to create embeddings for.
        model_name (str): Name of the model to use for creating embeddings.
        output_path (str): Path to save the embeddings.
    """
    # Load the model
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(f"sentence-transformers/{model_name}", token=False)
    print("Model loaded successfully.")
    # Create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True)
    print("Embeddings created successfully.")
    # Save the embeddings
    print(f"Saving embeddings to {output_path}...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Wikipedia dataset.")
    parser.add_argument("--version", type=str, default="20220301.simple", help="Version of the Wikipedia dataset to load.")
    parser.add_argument("--file_path", type=str, default="data/wikipedia_simple_en.hf", help="Path to the saved dataset.")
    parser.add_argument("--output_path", type=str, default="data/processed/wikipedia_embeddings.npz", help="Path to save the embeddings.")
    args = parser.parse_args()
    version = args.version
    file_path = args.file_path
    # Check if the file path exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        raise FileNotFoundError(f"{file_path} does not exist.")
    # Check if the file path is a directory
    if not os.path.isdir(file_path):
        print(f"Error: {file_path} is not a directory.")
        raise NotADirectoryError(f"{file_path} is not a directory.")
    # Check if the file path is empty
    if not os.listdir(file_path):
        print(f"Error: {file_path} is empty.")
        raise ValueError(f"{file_path} is empty.")
    
    # Load the dataset
    dataset = load_wikipedia_dataset(version, file_path)
    
    # Extract the first paragraph from each document
    docs = [extract_first_paragraph(doc['text']) for doc in dataset['train']]

    # Save the vocabulary to a file
    vocab = extract_vocab(docs)
    with open("data/processed/vocab.txt", "w") as f:
        for word in vocab:
            f.write(f"{word}\n")
    print(f"Vocabulary saved to data/processed/vocab.txt")

    # Save the documents to a file
    with open("data/processed/wikipedia_docs.txt", "w") as f:
        for doc in docs:
            f.write(f"{doc}\n\n\n")
    print(f"First paragraphs saved to data/processed/wikipedia_docs.txt")
    
    # Create embeddings for the documents
    create_embeddings(docs, output_path="data/processed/wikipedia_embeddings.npz")