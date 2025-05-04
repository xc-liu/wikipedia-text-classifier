from bertopic import BERTopic

# Only load once
topic_model = BERTopic.load("models/wikipedia_topic_model")