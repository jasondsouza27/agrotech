# AgroSmart: Intelligent Agriculture Ecosystem 🌾

> Complete IoT-enabled precision farming platform with AI-powered crop disease detection, real-time sensor monitoring, and automated irrigation control

[![Green Growth Certified](https://img.shields.io/badge/Green%20Growth-Certified-brightgreen)]() 
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![React](https://img.shields.io/badge/React-18+-61dafb)]()
[![ESP32](https://img.shields.io/badge/ESP32-IoT-orange)]()

## 🌟 Overview

**AgroSmart** is a comprehensive precision agriculture platform that combines AI-powered crop disease detection, IoT sensor integration, and intelligent farm management. The system provides farmers with real-time insights, automated irrigation control, market price information, and organic crop management recommendations.

### Key Features

🌿 **Leaf Disease Detection**
- AI-powered image analysis using YOLOv8
- Real-time webcam monitoring
- Mobile-friendly capture interface
- Organic remedy recommendations

📡 **Live IoT Sensor Integration**
- ESP32-based hardware sensors
- Real-time monitoring of:
  - Soil moisture levels
  - Temperature & humidity
  - Pump status & automation
- USB serial or WiFi connectivity

🤖 **Intelligent Farm Agent**
- ML-powered irrigation decisions (Random Forest)
- Automated pump control based on soil moisture
- Historical data tracking & analysis
- Predictive analytics for optimal watering

💹 **Market Intelligence (Mandi Connect)**
- Real-time crop price information
- Multi-state market comparison
- Price trend analysis
- Export-ready market data

🌤️ **Weather Integration**
- Location-based weather forecasts
- Agricultural alerts & warnings
- Climate-aware recommendations

🎓 **AI Assistant (RegenChat)**
- Natural language farming queries
- Crop recommendations
- Sustainable farming practices
- Multilingual support (coming soon)

## 🚀 Quick Start

### Option 1: Complete System (All Features)

```bash
# 1. Start the IoT Farm Agent Server (Port 5000)
python farm_agent_server.py

# 2. Start the Leaf Disease Detection Server (Port 5000 or 8081)
python prediction_server.py

# 3. Start the React Dashboard (Port 5173)
cd frontend
npm install
npm run dev

# 4. (Optional) Connect ESP32 sensors
cd esp32
python serial_bridge.py
```

### Option 2: Quick Launch (Windows)

**Double-click** `run_project.bat` to start the backend automatically!

### Option 3: Leaf Detection Only

```bash
# Start disease detection server
python prediction_server.py

# Open test interface
# Visit: http://localhost:5000 (or open index.html)
```

## 📋 Installation

### System Requirements
- Python 3.8 or higher
- Node.js 18+ (for frontend)
- Git (recommended)

### Backend Dependencies

```bash
# Core dependencies
pip install flask flask-cors ultralytics opencv-python numpy

# ML & Data Science
pip install scikit-learn pandas requests

# ESP32 Integration (optional)
pip install pyserial

# Or install all at once
pip install -r requirements.txt
```

### Frontend Dependencies

```bash
cd frontend
npm install
# or
bun install
```

### Hardware Setup (Optional)

For IoT sensor integration, see [ESP32 Setup Guide](esp32/README.md)

Required components:
- ESP32 DevKit (any variant)
- DHT11 Temperature/Humidity Sensor
- Capacitive Soil Moisture Sensor
- 5V Relay Module
- Water pump (5-12V)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGROGUARDIAN DASHBOARD                      │
│              (React + TypeScript + Shadcn UI)                   │
└────────────────┬─────────────────┬──────────────────────────────┘
                 │                 │
        ┌────────▼────────┐   ┌────▼────────────────┐
        │  Leaf Disease   │   │  Farm Agent Server  │
        │  Detection API  │   │   (Port 5000)       │
        │  (Port 5000)    │   │                     │
        │                 │   │  • IoT Integration  │
        │  • YOLOv8       │   │  • ML Controller    │
        │  • CV Analysis  │   │  • Mandi Connect    │
        │  • Organic Fix  │   │  • Sensor History   │
        └────────┬────────┘   └──────┬──────────────┘
                 │                   │
                 │            ┌──────▼──────────┐
                 │            │  Serial Bridge  │
                 │            │  (USB/WiFi)     │
                 │            └──────┬──────────┘
                 │                   │
                 │            ┌──────▼──────────┐
                 │            │   ESP32 Device  │
                 │            │   with Sensors  │
                 │            └─────────────────┘
                 │
          ┌──────▼──────────┐
          │  External APIs  │
          │  • Weather      │
          │  • Market Data  │
          └─────────────────┘
```

## 🗂️ Project Structure

```
agrotech/
├── 📄 farm_agent_server.py          # Main IoT & farm management server
├── 📄 prediction_server.py          # Leaf disease detection API
├── 📄 real_time_detection.py        # Webcam live monitoring
├── 📄 run_project.bat               # Windows quick launcher
├── 📄 index.html                    # Simple web test interface
├── 🤖 yolov8n.pt                    # YOLOv8 model weights
│
├── 📁 esp32/                        # IoT Hardware Integration
│   ├── serial_bridge.py             # USB serial data bridge
│   ├── README.md                    # Hardware setup guide
│   └── agrosmart_esp32/
│       └── agrosmart_esp32.ino      # ESP32 Arduino firmware
│
└── 📁 frontend/                     # React Dashboard
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── pages/
        │   ├── Index.tsx            # Main dashboard
        │   ├── Login.tsx
        │   └── Signup.tsx
        └── components/
            └── dashboard/
                ├── LeafDoctor.tsx           # Disease detection UI
                ├── LiveSensorData.tsx       # ESP32 data display
                ├── PumpControl.tsx          # Irrigation control
                ├── MandiConnect.tsx         # Market prices
                ├── WeatherWidget.tsx        # Weather forecast
                ├── RegenChat.tsx            # AI assistant
                ├── CropRecommendation.tsx   # ML suggestions
                ├── SustainabilityMetrics.tsx
                └── SensorChart.tsx          # Historical graphs
```

## 📡 API Documentation

### Farm Agent Server (Port 5000)

#### ESP32 Sensor Data
```http
GET /api/esp32/sensors
```

**Response:**
```json
{
  "devices": [{
    "device_id": "ESP32_abc123",
    "timestamp": "2026-02-27T10:30:00",
    "temperature": 28.5,
    "humidity": 65.2,
    "soil_moisture": 45.8,
    "pump_running": false,
    "is_simulated": false
  }]
}
```

#### Pump Control
```http
POST /api/pump/control
Content-Type: application/json

{
  "action": "on",
  "device_id": "ESP32_abc123",
  "duration": 300
}
```

#### Irrigation Prediction (ML)
```http
POST /api/predict/irrigation
Content-Type: application/json

{
  "soil_moisture": 35,
  "temperature": 32,
  "humidity": 45,
  "time_since_last_water": 12
}
```

**Response:**
```json
{
  "prediction": "water_needed",
  "confidence": 0.89,
  "recommendation": "Irrigate for 15 minutes"
}
```

#### Market Prices (Mandi Connect)
```http
GET /api/mandi/prices?state=Punjab&crop=Wheat
```

**Response:**
```json
{
  "prices": [{
    "market": "Ludhiana",
    "crop": "Wheat",
    "variety": "Durum",
    "price_min": 2100,
    "price_max": 2250,
    "price_modal": 2180,
    "date": "2026-02-27"
  }]
}
```

### Leaf Disease Detection Server (Port 5000/8081)

#### Health Check
```http
GET /api/health
```

#### Scan Crop Image
```http
POST /api/scan/image
Content-Type: multipart/form-data

image: [file upload]
```

**Response:**
```json
{
  "success": true,
  "scan_id": "SCAN-20260227-103045",
  "result": {
    "diagnosis": "Early Blight (Alternaria solani)",
    "description": "Fungal infection causing brown spots with concentric rings",
    "remedy": "Apply neem oil spray (2%) every 7 days. Remove infected leaves. Improve air circulation.",
    "severity": "moderate",
    "confidence": 0.87,
    "prevention": "Crop rotation, avoid overhead watering, mulch to prevent soil splash"
  },
  "timestamp": "2026-02-27T10:30:45Z"
}
```

#### Capture Frame (Webcam)
```http
POST /api/scan/frame
Content-Type: application/json

{
  "frame": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

#### System Information
```http
GET /api/system/info
```

## 🌾 Supported Crop Diseases

### Detection Categories

| Category | Diseases | Severity | Treatment |
|----------|----------|----------|-----------|
| **Fungal** | Early Blight, Late Blight, Powdery Mildew, Rust, Anthracnose | High | Neem oil, Copper spray, Sulfur dust |
| **Bacterial** | Bacterial Spot, Leaf Blight | Moderate | Copper-based organic spray, Remove infected parts |
| **Nutritional** | Nitrogen Deficiency, Potassium Deficiency, Iron Chlorosis | Low-Moderate | Organic compost, Companion planting, Natural amendments |
| **Viral** | Mosaic Virus, Leaf Curl | High | Remove infected plants, Control vectors, Resistant varieties |
| **Healthy** | No disease detected | - | Continue monitoring, Preventive care |

### Organic Remedies (Green Growth Certified ✅)

All treatment recommendations are 100% organic and sustainable:

- **Neem Oil Spray**: Natural fungicide and pesticide
- **Compost Tea**: Beneficial microorganisms boost plant immunity
- **Companion Planting**: Legumes for nitrogen, marigolds for pest control
- **Copper Spray**: Organic copper sulfate for fungal diseases
- **Crop Rotation**: Prevent disease buildup in soil
- **Mulching**: Reduce soil-borne pathogen splash
- **Beneficial Insects**: Ladybugs, lacewings for pest control

## 🎥 Real-Time Detection

### Webcam Monitoring

For continuous live monitoring of crops:

```bash
python real_time_detection.py
```

**Features:**
- Color-based health analysis
- Green area detection
- Yellowing/browning detection
- Real-time FPS counter
- Automatic alerts

**Controls:**
- `Q` - Quit
- `S` - Save screenshot
- `SPACE` - Pause/Resume

### Custom YOLO Model

To use your own trained disease detection model:

```bash
# Option 1: Environment variable
set YOLO_MODEL_PATH=models/plant_disease_best.pt
python prediction_server.py

# Option 2: Edit configuration
# In prediction_server.py, line 34:
MODEL_PATH = 'path/to/your/best.pt'
```

**Training Your Model:**
See [YOLO Training Guide](docs/TRAINING.md) for PlantVillage dataset instructions.

## 🖥️ Frontend Dashboard

### Starting the Dashboard

```bash
cd frontend
npm install      # First time only
npm run dev      # Development server

# Production build
npm run build
npm run preview
```

Access at: **http://localhost:5173**

### Dashboard Features

#### 1. **Live Sensor Panel** 📊
- Real-time ESP32 sensor readings
- Historical data charts
- Status indicators
- Connection health

#### 2. **Leaf Doctor** 🔬
- Upload images or use camera
- Instant disease diagnosis
- Organic treatment plans
- Confidence scores

#### 3. **Pump Control** 💧
- Manual on/off control
- Automatic mode with ML
- Runtime tracking
- Schedule irrigation

#### 4. **Weather Widget** ⛅
- 5-day forecast
- Agricultural alerts
- Temperature/humidity trends
- Precipitation probability

#### 5. **Mandi Connect** 💹
- Live market prices
- Multi-state comparison
- Price trends
- Export data

#### 6. **Regen Chat** 💬
- AI farming assistant
- Context-aware responses
- Crop recommendations
- Problem diagnosis

#### 7. **Sustainability Metrics** 🌱
- Water conservation
- Chemical-free days
- Carbon footprint
- Soil health score

## 🔌 ESP32 IoT Integration

### Hardware Setup

See detailed guide: [esp32/README.md](esp32/README.md)

**Quick Overview:**

1. **Flash ESP32** with `agrosmart_esp32.ino`
2. **Connect sensors**:
   - DHT11 → GPIO 4
   - Soil Moisture → GPIO 34
   - Relay (Pump) → GPIO 26
3. **Connect USB** cable to computer
4. **Run serial bridge**: `python esp32/serial_bridge.py`

### Protocol

ESP32 sends JSON data over serial (115200 baud):

```json
{
  "device_id": "ESP32_abc123",
  "temperature": 28.5,
  "humidity": 65.2,
  "soil_moisture": 45.8,
  "pump_running": false,
  "timestamp": 1709031000
}
```

Serial bridge forwards to Farm Agent Server via HTTP POST.

### Pump Commands

Server can send pump control commands:

```
PUMP:ON
PUMP:OFF
STATUS
```

## 🧪 Testing

### Backend Testing

```bash
# Test leaf detection API
curl -X POST http://localhost:5000/api/scan/image \
  -F "image=@test_leaf.jpg"

# Test sensor endpoint
curl http://localhost:5000/api/esp32/sensors

# Test pump control
curl -X POST http://localhost:5000/api/pump/control \
  -H "Content-Type: application/json" \
  -d '{"action":"on","device_id":"ESP32_test"}'
```

### Frontend Testing

```bash
cd frontend
npm test              # Run tests once
npm run test:watch    # Watch mode
```

## 🌍 Environment Variables

Create `.env` file in project root:

```bash
# Model Configuration
YOLO_MODEL_PATH=yolov8n.pt

# Server Configuration
FLASK_PORT=5000
FLASK_ENV=development

# API Keys (optional)
WEATHER_API_KEY=your_key_here
MANDI_API_KEY=your_key_here

# Database (optional)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
```

Frontend `.env` (`frontend/.env`):

```bash
VITE_API_URL=http://localhost:5000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
```

## 🚀 Deployment

### Backend (Flask)

#### Option 1: Gunicorn (Linux)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 farm_agent_server:app
```

#### Option 2: Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "farm_agent_server.py"]
```

### Frontend (React)

#### Build for Production
```bash
cd frontend
npm run build
# Output in: frontend/dist/
```

#### Deploy to Vercel/Netlify
```bash
# Vercel
vercel --prod

# Netlify
netlify deploy --prod --dir=dist
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Support for more crop types
- [ ] Mobile app (React Native)
- [ ] Multilingual support
- [ ] Advanced ML models
- [ ] Soil testing integration
- [ ] Drone imagery analysis
- [ ] Blockchain traceability

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/agrotech.git
cd agrotech

# Create branch
git checkout -b feature/your-feature

# Make changes and test
python -m pytest

# Submit PR
```

## 📚 Additional Resources

- [ESP32 Setup Guide](esp32/README.md)
- [API Reference](docs/API.md)
- [Training Custom Models](docs/TRAINING.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Video Tutorials](https://youtube.com/@agrotech)

## 🐛 Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

**Problem**: ESP32 not detected
```bash
# Windows
python esp32/serial_bridge.py COM3

# Check available ports
python -m serial.tools.list_ports
```

**Problem**: YOLO model not loading
```bash
# Download YOLOv8 nano model
pip install ultralytics
# It will auto-download on first run
```

**Problem**: Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## 📊 System Requirements

### Minimum
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Storage: 2 GB
- OS: Windows 10, Ubuntu 20.04, macOS 11+

### Recommended
- CPU: Quad-core 3.0 GHz
- RAM: 8 GB
- GPU: NVIDIA CUDA-capable (for faster YOLO)
- Storage: 5 GB SSD
- OS: Windows 11, Ubuntu 22.04, macOS 12+

## 📜 License

MIT License - Free for agricultural and educational use.

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **PlantVillage Dataset** - Disease image training data
- **Shadcn UI** - Beautiful React components
- **ESP32 Community** - IoT hardware support
- **Open Source Contributors** - Thank you! 🌟

## 🌱 Green Growth Certified

This project is certified by Green Growth Initiative for:
- 100% organic remedy recommendations
- Sustainable farming practices
- Water conservation focus
- Chemical-free pest management
- Soil health preservation

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/agrotech/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agrotech/discussions)
- **Email**: support@agroguardian.com
- **Community**: [Discord Server](https://discord.gg/agrotech)

---

**Built with ❤️ for sustainable agriculture and farming communities worldwide**

*Empowering farmers with technology, one crop at a time* 🌾
