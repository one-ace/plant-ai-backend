import os
import requests

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params = {"id": file_id, "confirm": value}
            response = session.get(URL, params=params, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)


MODEL_PATH = "plant_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_file_from_google_drive(
        "1TaKh33MEPRRdIK-i3K0JbwjJHg_E6c4m",
        MODEL_PATH
    )
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
