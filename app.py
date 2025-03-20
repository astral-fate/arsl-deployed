from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load class mapping from file
try:
    class_mapping = torch.load('class_mapping.pth', weights_only=True)
    print(f"Class mapping loaded successfully with {len(class_mapping)} classes")
except Exception as e:
    print(f"Error loading class mapping: {e}")
    exit() # Cannot proceed without class mapping

# Number of classes
num_classes = len(class_mapping)

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Definition matching the training architecture
class ArSLNet(nn.Module):
    def __init__(self, num_classes=32):  # Ensure num_classes matches your training data
        super(ArSLNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load model once when starting the server
model = None
try:
    model = ArSLNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_arsl_model.pth', map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit() # Cannot proceed without model

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400

        if 'image' in request.files:
            # Handle form-data image upload
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')  # Ensure RGB
        else:
            # Handle base64 image data
            image_data = request.json['image']
            try:
                # Handle data URL format
                if 'data:image/' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Ensure RGB
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

        # Convert PIL Image to tensor
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted_idx = torch.max(output, 1)
            
            # Get the predicted class
            predicted_idx_item = predicted_idx.item()
            if predicted_idx_item in class_mapping:
                predicted_class = class_mapping[predicted_idx_item]
            else:
                return jsonify({'error': f'Predicted index {predicted_idx_item} not found in class mapping'}), 500
                
            confidence = float(probabilities[predicted_idx_item].item() * 100)

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create template and static folders if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Make sure index.html is in the templates directory
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            with open('index.html', 'r') as source:
                f.write(source.read())

    # Run app with host='0.0.0.0' to allow external access
    try:
        # First try port 80 (standard HTTP port)
        print("Starting server on port 80...")
        app.run(host='0.0.0.0', port=80, debug=False)
    except PermissionError:
        # If port 80 fails, try port 8080
        print("Could not bind to port 80 (requires admin privileges)")
        print("Starting server on port 8080...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        print(f"Error starting server: {e}")
