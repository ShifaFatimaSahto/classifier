

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

def predict(image_path):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

if _name_ == "_main_":
    result = predict("test_image.jpg")
    print(f"Predicted class: {result}")
