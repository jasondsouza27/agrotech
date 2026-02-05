# AgroTech - Leaf Disease Detection System 🌿

> AI-powered crop disease detection with YOLO + organic remedies

## Overview

Complete agricultural disease detection system with:
- **Backend API**: Flask server with YOLO integration
- **Real-time Detection**: Webcam-based live monitoring
- **Web Interface**: Drag-and-drop image upload
- **React Dashboard**: Full-featured AgroGuardian frontend

All recommended remedies are **100% organic and sustainable** (Green Growth certified).

## 🚀 Quick Start (Easiest Way)

**Double-click** `run_project.bat` to start everything automatically!

Or manually:

```bash
# 1. Start Backend
python prediction_server.py

# 2. Open test_api.html in your browser
# Drag & drop leaf images to test
```

## 📋 Installation

```bash
# Required dependencies
pip install flask flask-cors ultralytics opencv-python

# Optional: For full React frontend
# Install Node.js from https://nodejs.org/ then:
cd frontend
npm install
npm run dev
```


## 🔧 Project Structure

```
agrotech/
├── prediction_server.py      # Flask API with YOLO integration
├── real_time_detection.py    # Webcam real-time detection
├── test_api.html             # Simple web interface (no Node.js needed)
├── run_project.bat           # One-click launcher
├── yolov8n.pt                # YOLO model weights
└── frontend/                 # React dashboard (requires Node.js)
    └── src/components/dashboard/LeafDoctor.tsx
```

## 📡 API Endpoints

### Health Check
```
GET /api/health
```

### Scan Crop Image
```
POST /api/scan/image
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "success": true,
  "scan_id": "SCAN-12345",
  "result": {
    "diagnosis": "Nitrogen Deficiency",
    "description": "Yellowing detected in older leaves...",
    "remedy": "Plant beans or legumes nearby...",
    "severity": "moderate",
    "confidence": 0.87
  }
}
```

### System Info
```
GET /api/system/info
```

### List All Diagnoses
```
GET /api/scan/diagnoses
```

## 🎥 Real-Time Detection

For webcam-based continuous monitoring:

```bash
python real_time_detection.py
```

Press `q` to quit.

## 🧪 Custom YOLO Model

To use your own trained model:

```bash
# Option 1: Environment variable
set YOLO_MODEL_PATH=path/to/your/best.pt
python prediction_server.py

# Option 2: Edit MODEL_PATH in prediction_server.py (line 34)
MODEL_PATH = 'path/to/your/best.pt'
```

## 🌐 Frontend Options

### Option 1: Simple Web Interface (No Installation)
- Open `test_api.html` in any browser
- Drag & drop images
- Works immediately

### Option 2: Full React Dashboard
- Requires Node.js
- Professional UI with charts, alerts, metrics
- Run: `cd frontend && npm install && npm run dev`

## 🔮 Detected Conditions

- ✅ Healthy crops
- ⚠️ Nitrogen deficiency
- 🔴 Early blight (fungal)
- 🔴 Late blight
- 🔴 Leaf spot
- 🔴 Rust
- 🔴 Powdery mildew

## 🌱 Green Growth Certified

All remedies are organic and sustainable:
- Neem oil sprays
- Companion planting
- Compost tea
- Natural nitrogen fixation
- Crop rotation

## 📝 License

MIT License - Free for agricultural use

---

**Built with ❤️ for sustainable agriculture**
