import os
import json
import requests
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model

# =========================
# FIREBASE SETUP (SECURE)
# =========================

firebase_key_json = os.environ.get("FIREBASE_KEY")

if not firebase_key_json:
    raise ValueError("FIREBASE_KEY environment variable not set")

firebase_key = json.loads(firebase_key_json)

cred = credentials.Certificate(firebase_key)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-nursery-system-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# =========================
# MODEL DOWNLOAD
# =========================

MODEL_PATH = "plant_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://huggingface.co/szzw/plant-ai-model/resolve/main/plant_model.pth?download=true"
    
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print("Model downloaded successfully!")
    print("File size:", os.path.getsize(MODEL_PATH))
else:
    print("Model already exists.")
    print("File size:", os.path.getsize(MODEL_PATH))

# =========================
# FLASK APP
# =========================

app = Flask(__name__)

# Lazy loaded model
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = get_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
        model.eval()
        print("Model loaded successfully!")

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "AI Plant Model Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    global model

    # Load model only when needed
    if model is None:
        load_model()

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    result = "Healthy" if predicted.item() == 0 else "Diseased"

    # =========================
    # UPDATE FIREBASE
    # =========================
    ref = db.reference("ai")
    ref.set({
        "status": result
    })

    return jsonify({"status": result})

# =========================
# RUN SERVER (LOCAL ONLY)
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
