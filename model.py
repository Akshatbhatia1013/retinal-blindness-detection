from send_sms import send_sms 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Prediction classes
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Transform
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes=5):
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1)
    )
    model.to(device)
    return model

def load_model(checkpoint_path="clean_classifier.pt"):
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_gradcam(model, image_tensor, class_idx):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_conv = model._modules.get('layer4')
    handle_fw = final_conv.register_forward_hook(forward_hook)
    handle_bw = final_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grads_val = gradients[0]
    activs = activations[0]
    weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * activs, dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    heatmap = (cam * 255).astype(np.uint8)

    handle_fw.remove()
    handle_bw.remove()
    return heatmap
def predict(model, image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = test_transforms(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.exp(output)
        top_p, top_class = probabilities.topk(1, dim=1)

        pred_idx = top_class.item()
        pred_label = classes[pred_idx]
        confidence = top_p.item() * 100  # Confidence in percent

    # Grad-CAM heatmap
    heatmap = generate_gradcam(model, img_tensor, pred_idx)  # Assumes output is 224x224
    orig_img = np.array(image.resize((224, 224)))  # Resize to match
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

    # Send SMS (optional)
    sms_text = f"🧠 Prediction: {pred_label} ({pred_idx}) with {confidence:.2f}% confidence"
    send_sms(sms_text)

    print(sms_text)  # Debug print

    return pred_idx, pred_label, overlaid
def main(image_path):
    model = load_model("clean_classifier.pt")
    return predict(model, image_path)
