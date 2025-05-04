from fastapi import FastAPI
from app.model_loader import topic_model
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Wikipedia Topic Classification API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_topic(request: PredictionRequest):
    """
    Predict the topic of a given text using the pre-trained BERTopic model.
    Args:
        request (PredictionRequest): The request containing the text to classify.
    Returns:
        PredictionResponse: The response containing the topic and probability.
    """
    topics, probs = topic_model.transform([request.text])
    topic = topics[0]
    prob = probs[0]

    topic_top_words = topic_model.get_topic(topic=topic)

    return {
        "topic_top_words": topic_top_words,
        "prob": prob
    }