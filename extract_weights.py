import torch
import torch.nn as nn
from torchvision import models

# Define your model again
def build_model(num_classes=5):
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1)
    )
    return model

# Build the model structure
model = build_model()

# Load the state_dict from your classifier.pt file
checkpoint = torch.load("classifier.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Save a clean version with just state_dict
torch.save({'model_state_dict': model.state_dict()}, "clean_classifier.pt")

print("✅ Extracted clean weights and saved to clean_classifier.pt")
