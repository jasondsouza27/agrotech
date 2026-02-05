# AgroSmart: Green Growth Edition

Autonomous Farm Agent with ML-powered irrigation control and real-time weather integration.

## Features

- 🤖 **ML-Based Predictions**: Random Forest Classifier with 95%+ accuracy
- 🌦️ **Real Weather Data**: Integration with Open-Meteo API (free, no API key required)
- 🌾 **Crop-Specific Logic**: Dynamic thresholds for Rice, Maize, and Chickpea
- 🔒 **Safety Systems**: Emergency stop, critical moisture override, evaporation guard
- 📊 **Feature Importance**: Transparent ML decision-making
- 💬 **AI Mock Endpoints**: Image diagnosis and NPK-based soil advice

## Tech Stack

- **Backend**: Flask (Python 3.13)
- **ML**: scikit-learn (Random Forest)
- **Weather API**: Open-Meteo
- **Data Processing**: NumPy

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install flask scikit-learn numpy requests
```

## Running the Server

```bash
python prediction_server.py
```

Server will start at: **http://127.0.0.1:5000**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Get ML-based irrigation decision |
| POST | `/api/scan/image` | Mock crop/soil image diagnosis |
| POST | `/api/chat` | NPK-based soil regeneration advice |
| GET | `/model/info` | ML model details & feature importance |
| GET | `/health` | Server health check |
| GET | `/` | API documentation |

## Example Usage

### Irrigation Prediction

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "soil_moisture": 40,
    "crop_type": "Rice",
    "pump_runtime_minutes": 5,
    "latitude": 20.5937,
    "longitude": 78.9629
  }'
```

**Response:**
```json
{
  "pump_command": "ON",
  "agent_reason": "ML Model predicts ON with 94.2% confidence",
  "ml_prediction": {
    "confidence": 0.942,
    "probabilities": {"OFF": 0.058, "ON": 0.942}
  },
  "weather_data": {
    "source": "Open-Meteo API",
    "rain_probability": 15,
    "temperature": 32
  }
}
```

### NPK Soil Advice

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"nitrogen": 25, "phosphorus": 60, "potassium": 20}'
```

## ML Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 100
- **Max Depth**: 10
- **Training Data**: 10,000 synthetic samples
- **Features**: soil_moisture, temperature, humidity, rain_probability, wind_speed, hour_of_day, crop_type
- **Accuracy**: ~95%

## Safety Features

1. **Emergency Stop**: Auto-shutoff if pump runs >30 minutes
2. **Critical Override**: Force ON if moisture <15%
3. **Evaporation Guard**: Avoid irrigation 11 AM - 3 PM (unless critical)
4. **Rain Check**: Skip irrigation if rain probability >80%

## Project Structure

```
agrotech/
├── prediction_server.py    # Main Flask application
├── irrigation_model.pkl    # Trained ML model (auto-generated)
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## Author

Built for AgroSmart: Green Growth Edition  
ML-Powered Smart Irrigation System

## License

MIT License
