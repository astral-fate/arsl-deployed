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
from flask_socketio import SocketIO, emit
import eventlet

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class mapping - hardcoded version for immediate use
class_mapping = {
    0: 'ain', 1: 'al', 2: 'aleff', 3: 'bb', 4: 'dal',
    5: 'dha', 6: 'dhad', 7: 'fa', 8: 'gaaf', 9: 'ghain',
    10: 'ha', 11: 'haa', 12: 'jeem', 13: 'kaaf', 14: 'khaa',
    15: 'la', 16: 'laam', 17: 'meem', 18: 'nun', 19: 'ra',
    20: 'saad', 21: 'seen', 22: 'sheen', 23: 'ta', 24: 'taa',
    25: 'thaa', 26: 'thal', 27: 'toot', 28: 'waw', 29: 'ya',
    30: 'yaa', 31: 'zay'
}

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Basic Block for ResNet-style architecture
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Model Definition
class ArSLNet(nn.Module):
    def __init__(self, num_classes=32):
        super(ArSLNet, self).__init__()
        
        # Feature Extraction using ResNet-style architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2)
        )
        
        # LSTM with correct hidden sizes
        hidden_size = 512  # Increased from 256 to 512
        self.lstm = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        
        # Attention mechanism adjusted for larger hidden size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # *2 because of bidirectional
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Updated classifier to match saved model's layer indices
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),  # *2 because of bidirectional
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Index 8 in the saved model
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Get dimensions
        batch_size, channels, height, width = x.size()
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

# Load model once when starting the server
model = None
try:
    # Use weights_only=True to avoid pickle warning
    checkpoint = torch.load('improved_arsl_model.pth', map_location=device, weights_only=True)
    model = ArSLNet(num_classes=len(class_mapping)).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Add prediction interval configuration
PREDICTION_INTERVAL = 500  # milliseconds

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
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the image from the request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400

        if 'image' in request.files:
            # Handle form-data image upload
            image_file = request.files['image']
            image = Image.open(image_file)
        else:
            # Handle base64 image data
            image_data = request.json['image']
            try:
                # Handle data URL format
                if 'data:image/' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

        # Convert PIL Image to tensor
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = class_mapping[predicted_idx.item()]
            confidence = float(probabilities[predicted_idx.item()].item() * 100)

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add new WebSocket routes before the main block
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('camera_frame')
def handle_camera_frame(data):
    if not model:
        emit('prediction_error', {'error': 'Model not loaded'})
        return

    try:
        # Process the base64 image
        image_data = data['image']
        if 'data:image/' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Make prediction
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = class_mapping[predicted_idx.item()]
            confidence = float(probabilities[predicted_idx.item()].item() * 100)

        emit('prediction_result', {
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        emit('prediction_error', {'error': str(e)})

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
        socketio.run(app, host='0.0.0.0', port=80, debug=False)
    except PermissionError:
        # If port 80 fails, try port 8080
        print("Could not bind to port 80 (requires admin privileges)")
        print("Starting server on port 8080...")
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        print(f"Error starting server: {e}")