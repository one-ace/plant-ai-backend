import os
import requests

MODEL_PATH = "plant_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://huggingface.co/szzw/plant-ai-model/resolve/main/plant_model.pth"
    
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print("Model downloaded successfully!")
    print("File size:", os.path.getsize(MODEL_PATH))
from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model

app = Flask(__name__)

# Load model
model = get_model()
model.load_state_dict(torch.load("plant_model.pth", map_location="cpu", weights_only=False))
model.eval()

# Transform (same as validation!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    result = "Healthy" if predicted.item() == 0 else "Diseased"

    return jsonify({"status": result})

@app.route("/")
def home():
    return "AI Plant Model Running 🚀"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
