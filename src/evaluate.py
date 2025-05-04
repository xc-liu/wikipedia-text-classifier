from bertopic import BERTopic
import random
import argparse


def load_topic_model(model_path):
    """
    Load a BERTopic model from the specified path.
    Args:
        model_path (str): Path to the model.
    Returns:
        BERTopic: The loaded BERTopic model.
    """
    topic_model = BERTopic.load(model_path)
    return topic_model

def evaluate_topic_model(topic_model, doc):
    """
    Evaluate the topic model on a given document.
    Args:
        topic_model (BERTopic): The BERTopic model to evaluate.
        doc (str): The document to evaluate.
    Returns:
        list: List of topics for the document.
    """
    topics, probs = topic_model.transform([doc])
    topic = topics[0]
    prob = probs[0]

    topic_top_words = topic_model.get_topic(topic=topic)
    return topic_top_words, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BERTopic model.")
    parser.add_argument("--model_path", type=str, default="models/wikipedia_topic_model", help="Path to the trained BERTopic model.")
    
    args = parser.parse_args()

    # Load the topic model
    topic_model = load_topic_model(args.model_path)

    # Use a random document from the training set to evaluate
    with open("data/processed/wikipedia_docs.txt", "r") as f:
        docs = f.read().split("\n\n\n")[:-1]
        
    doc = docs[random.randint(0, len(docs) - 1)]
    print(f"Evaluating document: {doc}")  # Print the first 50 characters of the document

    # Evaluate the topic model on the document
    topics, probs = evaluate_topic_model(topic_model, doc)
    
    print(f"Topic: {topics}")
    print(f"Probability: {probs}")