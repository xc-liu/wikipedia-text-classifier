# Wikipedia Text Classifier

A machine learning pipeline for topic modeling on Simple English Wikipedia articles using BERTopic, DVC, and FastAPI, packaged and served with Docker.

---

## 🚀 Project Overview

This project demonstrates an end-to-end MLOps workflow:
- Downloading and preprocessing a public dataset
- Generating sentence embeddings using transformer models
- Performing unsupervised topic modeling using BERTopic
- Structuring the ML pipeline using DVC
- Serving predictions via a FastAPI application
- Containerizing the app with Docker

The project is used for learning and practicing MLOps tools, and future steps may include deploying the app or integrating RAG.

---

## 🧰 Tools & Technologies

| Category        | Stack                         |
|----------------|-------------------------------|
| Data            | 🤗 Hugging Face `datasets`    |
| Embeddings      | `sentence-transformers`       |
| Topic Modeling  | `BERTopic`, `UMAP`, `HDBSCAN` |
| Pipeline Mgmt   | `DVC`                         |
| Serving         | `FastAPI`, `Uvicorn`          |
| Containerization| `Docker`                      |
| Language        | Python                        |

---

## 📂 Project Structure

```

.
├── app/                # FastAPI app
├── data/               # Raw & processed datasets (DVC tracked)
├── models/             # Trained BERTopic model
├── notebooks/          # Exploratory notebooks
├── src/                # Scripts for each pipeline stage
├── dvc.yaml            # DVC pipeline definition
├── Dockerfile          # Docker build script
└── requirements.txt

````

---

## 🧪 How to Run

### 1. Clone and set up environment
```bash
git clone https://github.com/yourusername/wikipedia-text-classifier.git
cd wikipedia-text-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### 2. Reproduce pipeline

```bash
dvc repro
```

### 3. Run FastAPI app locally

```bash
uvicorn app.main:app --reload
```

### 4. Or run inside Docker

```bash
docker build -t wikipedia-topic-api .
docker run -p 8000:8000 wikipedia-topic-api
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

---

## 🧠 What I Learned

* How to manage ML pipelines with DVC
* How to serve models with FastAPI
* How to structure a modular ML project
* How to containerize and run an ML app with Docker

---

## 🛠️ Future Plans

* Explore Retrieval-Augmented Generation (RAG) using the discovered topics
