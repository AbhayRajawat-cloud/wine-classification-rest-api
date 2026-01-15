# Wine Classification REST API

This project demonstrates how to deploy a Machine Learning model as a REST API using FastAPI.

## Project Overview
- Trained a Random Forest classification model on the Wine dataset
- Serialized the model using joblib
- Exposed predictions via a FastAPI REST endpoint
- Interactive API documentation via Swagger UI

## Tech Stack
- Python
- scikit-learn
- FastAPI
- Pandas
- Uvicorn

## How to Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload