

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(_name_)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Image Classifier API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return jsonify({"predicted_class": classes[predicted.item()]})

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
