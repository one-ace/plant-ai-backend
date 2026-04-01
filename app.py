import os
import requests

MODEL_PATH = "plant_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1TaKh33MEPRRdIK-i3K0JbwjJHg_E6c4m"
    
    r = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("Model downloaded successfully!")


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
    app.run()
