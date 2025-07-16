# MLOps Pipeline

This project shows a basic ML pipeline managed with MLflow, Docker, and Kubernetes.

## Features
- Model training and evaluation
- MLflow experiment tracking
- Docker containerization
- Kubernetes deployment manifests
- GitHub Actions for CI/CD

## Run Locally
```
python model/train.py
python model/evaluate.py
```

## Docker
```
docker build -t mlops-pipeline .
docker run mlops-pipeline
```
