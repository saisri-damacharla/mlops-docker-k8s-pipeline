import torch
from model.train import IrisClassifier, X_val, y_val

# Load model
model = IrisClassifier()
model.load_state_dict(torch.load("iris_model.pt"))
model.eval()

# Evaluate
with torch.no_grad():
    preds = model(X_val)
    predicted = preds.argmax(dim=1)
    accuracy = (predicted == y_val).float().mean().item()
    print(f"Validation Accuracy: {accuracy:.2%}")
