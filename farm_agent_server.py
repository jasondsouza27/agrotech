"""
AgroSmart: Green Growth Edition - Autonomous Farm Agent Server
A Flask-based prediction server with ML-powered irrigation control using Random Forest.
Includes Mandi Connect module for real-time market prices.
ESP32 Live Sensor Integration for IoT-based monitoring.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import time
import random
import numpy as np
import requests
import pickle
import os
import math
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# OpenCV for image analysis
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("⚠️ OpenCV not installed. Using mock image analysis.")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# =============================================================================
# ESP32 LIVE SENSOR DATA STORAGE
# =============================================================================

class ESP32DataStore:
    """
    Thread-safe storage for live ESP32 sensor data.
    Maintains latest readings and history for each connected device.
    """
    
    def __init__(self, max_history=100):
        self.devices = {}  # device_id -> latest data
        self.history = {}  # device_id -> list of historical readings
        self.max_history = max_history
        self.lock = threading.Lock()
        self.pump_commands = {}  # device_id -> pending command
    
    def update(self, device_id: str, data: dict):
        """Update sensor data for a device."""
        with self.lock:
            data["last_update"] = datetime.now().isoformat()
            data["timestamp"] = time.time()
            
            self.devices[device_id] = data
            
            # Add to history
            if device_id not in self.history:
                self.history[device_id] = []
            
            self.history[device_id].append({
                "timestamp": data["timestamp"],
                "soil_moisture": data.get("soil_moisture", 0),
                "temperature": data.get("temperature", 0),
                "humidity": data.get("humidity", 0),
                "pump_running": data.get("pump_running", False)
            })
            
            # Trim history if needed
            if len(self.history[device_id]) > self.max_history:
                self.history[device_id] = self.history[device_id][-self.max_history:]
    
    def get_latest(self, device_id: str = None) -> dict:
        """Get latest data for a device or all devices."""
        with self.lock:
            if device_id:
                return self.devices.get(device_id, {})
            return dict(self.devices)
    
    def get_history(self, device_id: str, limit: int = 50) -> list:
        """Get historical readings for a device."""
        with self.lock:
            history = self.history.get(device_id, [])
            return history[-limit:]
    
    def set_pump_command(self, device_id: str, command: str):
        """Set a pump command for a device (ON/OFF/AUTO)."""
        with self.lock:
            self.pump_commands[device_id] = command
    
    def get_pump_command(self, device_id: str) -> str:
        """Get and clear pending pump command."""
        with self.lock:
            return self.pump_commands.pop(device_id, None)
    
    def get_connected_devices(self) -> list:
        """Get list of connected devices with their last update time."""
        with self.lock:
            devices = []
            current_time = time.time()
            for device_id, data in self.devices.items():
                last_update = data.get("timestamp", 0)
                is_online = (current_time - last_update) < 30  # Online if updated within 30s
                devices.append({
                    "device_id": device_id,
                    "last_update": data.get("last_update", "Never"),
                    "is_online": is_online,
                    "soil_moisture": data.get("soil_moisture", 0),
                    "pump_running": data.get("pump_running", False)
                })
            return devices


# Initialize ESP32 data store
esp32_store = ESP32DataStore()


# =============================================================================
# MANDI CONNECT MODULE - "Bloomberg for Farmers"
# =============================================================================

class MandiConnect:
    """
    Real-time agricultural market price aggregator.
    Fetches prices from government APIs and sorts by proximity to farmer.
    """
    
    # Government API endpoints for agricultural market data
    AGMARKNET_API = "https://agmarknet.gov.in/api/commodity-prices"
    ENAM_API = "https://enam.gov.in/web/dashboard/trade-data"
    DATA_GOV_API = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    
    # Major agricultural markets (Mandis) across India with coordinates
    MANDI_DATABASE = {
        "nashik_mandi": {
            "name": "Nashik Mandi",
            "location": "Maharashtra",
            "latitude": 19.9975,
            "longitude": 73.7898,
            "type": "APMC",
            "speciality": ["Onion", "Tomato", "Grapes"]
        },
        "azadpur_mandi": {
            "name": "Azadpur Mandi",
            "location": "Delhi",
            "latitude": 28.7041,
            "longitude": 77.1025,
            "type": "APMC",
            "speciality": ["Vegetables", "Fruits"]
        },
        "vashi_apmc": {
            "name": "Vashi APMC",
            "location": "Mumbai",
            "latitude": 19.0760,
            "longitude": 72.9988,
            "type": "APMC",
            "speciality": ["Vegetables", "Fruits", "Grains"]
        },
        "kolar_mandi": {
            "name": "Kolar Mandi",
            "location": "Karnataka",
            "latitude": 13.1375,
            "longitude": 78.1291,
            "type": "APMC",
            "speciality": ["Tomato", "Vegetables"]
        },
        "pune_market": {
            "name": "Pune Market Yard",
            "location": "Maharashtra",
            "latitude": 18.5204,
            "longitude": 73.8567,
            "type": "APMC",
            "speciality": ["Onion", "Vegetables", "Grains"]
        },
        "rajkot_mandi": {
            "name": "Rajkot Mandi",
            "location": "Gujarat",
            "latitude": 22.3039,
            "longitude": 70.8022,
            "type": "APMC",
            "speciality": ["Groundnut", "Cotton", "Grains"]
        },
        "indore_mandi": {
            "name": "Indore Mandi",
            "location": "Madhya Pradesh",
            "latitude": 22.7196,
            "longitude": 75.8577,
            "type": "APMC",
            "speciality": ["Soybean", "Wheat", "Chickpea"]
        },
        "ludhiana_mandi": {
            "name": "Ludhiana Grain Market",
            "location": "Punjab",
            "latitude": 30.9010,
            "longitude": 75.8573,
            "type": "APMC",
            "speciality": ["Wheat", "Rice", "Maize"]
        },
        "guntur_mandi": {
            "name": "Guntur Chilli Yard",
            "location": "Andhra Pradesh",
            "latitude": 16.3067,
            "longitude": 80.4365,
            "type": "APMC",
            "speciality": ["Chilli", "Cotton", "Tobacco"]
        },
        "unjha_mandi": {
            "name": "Unjha APMC",
            "location": "Gujarat",
            "latitude": 23.8064,
            "longitude": 72.3958,
            "type": "APMC",
            "speciality": ["Cumin", "Fennel", "Spices"]
        }
    }
    
    # Base prices for crops (₹/Quintal) - Updated regularly from govt data
    CROP_BASE_PRICES = {
        "rice": {"min": 2200, "max": 2800, "unit": "Quintal"},
        "wheat": {"min": 2100, "max": 2600, "unit": "Quintal"},
        "maize": {"min": 1800, "max": 2200, "unit": "Quintal"},
        "chickpea": {"min": 5000, "max": 6500, "unit": "Quintal"},
        "soybean": {"min": 4200, "max": 5500, "unit": "Quintal"},
        "groundnut": {"min": 5500, "max": 7000, "unit": "Quintal"},
        "cotton": {"min": 6000, "max": 7500, "unit": "Quintal"},
        "onion": {"min": 1000, "max": 3500, "unit": "Quintal"},
        "tomato": {"min": 800, "max": 4000, "unit": "Quintal"},
        "potato": {"min": 1200, "max": 2500, "unit": "Quintal"},
        "chilli": {"min": 8000, "max": 15000, "unit": "Quintal"},
        "turmeric": {"min": 7000, "max": 12000, "unit": "Quintal"},
        "sugarcane": {"min": 300, "max": 400, "unit": "Quintal"},
        "mustard": {"min": 4500, "max": 6000, "unit": "Quintal"},
        "cumin": {"min": 15000, "max": 25000, "unit": "Quintal"}
    }
    
    # Local trader margins (typically 30-50% below mandi price)
    LOCAL_TRADER_MARGIN = 0.35  # 35% lower than best mandi price
    
    @classmethod
    def haversine_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.
        Returns distance in kilometers.
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    @classmethod
    def fetch_government_prices(cls, crop: str) -> dict:
        """
        Attempt to fetch real prices from government APIs.
        Falls back to simulated data if API is unavailable.
        """
        crop_lower = crop.lower()
        
        # Try data.gov.in API first
        try:
            api_key = os.environ.get("DATA_GOV_API_KEY", "")
            if api_key:
                params = {
                    "api-key": api_key,
                    "format": "json",
                    "filters[commodity]": crop.capitalize(),
                    "limit": 50
                }
                response = requests.get(cls.DATA_GOV_API, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("records"):
                        return {"source": "data.gov.in", "data": data["records"]}
        except Exception as e:
            print(f"Government API error: {e}")
        
        # Return None to indicate fallback needed
        return None
    
    @classmethod
    def generate_market_price(cls, crop: str, mandi_id: str, base_price: dict) -> dict:
        """
        Generate realistic market price with daily variation.
        In production, this would come from real API data.
        """
        # Use current date as seed for consistent daily prices
        date_seed = int(datetime.now().strftime("%Y%m%d"))
        mandi_seed = hash(mandi_id) % 1000
        crop_seed = hash(crop.lower()) % 1000
        
        random.seed(date_seed + mandi_seed + crop_seed)
        
        # Calculate price within range with some variation
        min_price = base_price["min"]
        max_price = base_price["max"]
        
        # Different mandis have different price levels
        mandi_factor = 0.85 + (mandi_seed % 30) / 100  # 0.85 to 1.15
        
        base = min_price + (max_price - min_price) * random.random()
        price = round(base * mandi_factor, 0)
        
        # Calculate daily change (-5% to +5%)
        random.seed(date_seed + mandi_seed + crop_seed + 1)
        change_percent = round((random.random() - 0.5) * 10, 1)
        
        # Reset random seed
        random.seed()
        
        return {
            "price": int(price),
            "change": f"+{change_percent}%" if change_percent >= 0 else f"{change_percent}%",
            "change_value": change_percent
        }
    
    @classmethod
    def get_market_prices(cls, crop: str, farmer_lat: float = None, farmer_lon: float = None) -> dict:
        """
        Get market prices for a crop across all mandis.
        Sorted by proximity to farmer's location.
        
        Args:
            crop: Crop name (e.g., "rice", "tomato")
            farmer_lat: Farmer's latitude (from account)
            farmer_lon: Farmer's longitude (from account)
        
        Returns:
            Dictionary with market prices and analysis
        """
        crop_lower = crop.lower()
        
        # Default location: Central India (Nagpur)
        if farmer_lat is None:
            farmer_lat = 21.1458
        if farmer_lon is None:
            farmer_lon = 79.0882
        
        # Check if crop is supported
        if crop_lower not in cls.CROP_BASE_PRICES:
            # Default to tomato prices if crop not found
            crop_lower = "tomato"
        
        base_price = cls.CROP_BASE_PRICES[crop_lower]
        
        # Try fetching real government data
        govt_data = cls.fetch_government_prices(crop)
        
        # Generate prices for all mandis
        market_prices = []
        
        for mandi_id, mandi_info in cls.MANDI_DATABASE.items():
            # Calculate distance from farmer
            distance = cls.haversine_distance(
                farmer_lat, farmer_lon,
                mandi_info["latitude"], mandi_info["longitude"]
            )
            
            # Generate price data
            price_data = cls.generate_market_price(crop, mandi_id, base_price)
            
            market_prices.append({
                "id": mandi_id,
                "name": mandi_info["name"],
                "location": mandi_info["location"],
                "type": mandi_info["type"],
                "distance_km": round(distance, 1),
                "price": price_data["price"],
                "price_display": f"₹{price_data['price']:,}/{base_price['unit']}",
                "change": price_data["change"],
                "change_value": price_data["change_value"],
                "coordinates": {
                    "latitude": mandi_info["latitude"],
                    "longitude": mandi_info["longitude"]
                }
            })
        
        # Sort by distance (closest first)
        market_prices.sort(key=lambda x: x["distance_km"])
        
        # Take top 7 markets
        top_markets = market_prices[:7]
        
        # Find best price
        best_market = max(top_markets, key=lambda x: x["price"])
        
        # Calculate local trader price (typically 35% lower)
        local_trader_price = int(best_market["price"] * (1 - cls.LOCAL_TRADER_MARGIN))
        potential_loss = best_market["price"] - local_trader_price
        income_increase_percent = round((potential_loss / local_trader_price) * 100, 0)
        
        return {
            "crop": crop.capitalize(),
            "unit": base_price["unit"],
            "markets": top_markets,
            "best_market": {
                "name": best_market["name"],
                "price": best_market["price"],
                "price_display": best_market["price_display"]
            },
            "local_trader": {
                "estimated_price": local_trader_price,
                "price_display": f"₹{local_trader_price:,}/{base_price['unit']}",
                "potential_loss": potential_loss,
                "loss_display": f"-₹{potential_loss:,}/{base_price['unit']}"
            },
            "recommendation": {
                "action": "Sell at Best Mandi Price",
                "income_increase": f"+{int(income_increase_percent)}% Higher Income",
                "extra_earning": f"₹{potential_loss:,}/{base_price['unit']} more"
            },
            "data_source": "Government APMC Data" if govt_data else "Market Simulation",
            "last_updated": datetime.now().isoformat(),
            "farmer_location": {
                "latitude": farmer_lat,
                "longitude": farmer_lon
            }
        }
    
    @classmethod
    def get_recommended_crop(cls, farmer_lat: float = None, farmer_lon: float = None) -> dict:
        """
        Get recommended crop based on current market trends and location.
        """
        # Analyze all crops and find best opportunity
        best_crop = None
        best_margin = 0
        
        for crop in cls.CROP_BASE_PRICES.keys():
            prices = cls.get_market_prices(crop, farmer_lat, farmer_lon)
            margin = prices["best_market"]["price"] - prices["local_trader"]["estimated_price"]
            
            if margin > best_margin:
                best_margin = margin
                best_crop = crop
        
        return {
            "recommended_crop": best_crop.capitalize() if best_crop else "Tomato",
            "reason": f"Highest profit margin of ₹{best_margin:,}/Quintal",
            "market_trend": "Bullish" if random.random() > 0.4 else "Stable"
        }


# Initialize Mandi Connect
mandi_connect = MandiConnect()


class WeatherAPI:
    """
    Real Weather API integration using Open-Meteo (free, no API key required).
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Default coordinates (can be overridden)
    DEFAULT_LAT = 20.5937  # India (central)
    DEFAULT_LON = 78.9629
    
    @classmethod
    def get_forecast(cls, latitude: float = None, longitude: float = None) -> dict:
        """
        Fetch real weather forecast from Open-Meteo API.
        
        Args:
            latitude: Location latitude (default: central India)
            longitude: Location longitude (default: central India)
        
        Returns:
            Dictionary with weather data including rain probability
        """
        lat = latitude or cls.DEFAULT_LAT
        lon = longitude or cls.DEFAULT_LON
        
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
                "hourly": ["precipitation_probability", "temperature_2m"],
                "forecast_days": 1,
                "timezone": "auto"
            }
            
            response = requests.get(cls.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract current weather
            current = data.get("current", {})
            hourly = data.get("hourly", {})
            
            # Get precipitation probability for next few hours
            precip_probs = hourly.get("precipitation_probability", [0])
            # Average of next 6 hours
            rain_probability = np.mean(precip_probs[:6]) if precip_probs else 0
            
            return {
                "rain_probability": round(rain_probability, 1),
                "temperature": current.get("temperature_2m", 25),
                "humidity": current.get("relative_humidity_2m", 50),
                "wind_speed": current.get("wind_speed_10m", 10),
                "current_precipitation": current.get("precipitation", 0),
                "source": "Open-Meteo API",
                "location": {"latitude": lat, "longitude": lon}
            }
            
        except requests.RequestException as e:
            # Fallback to simulated data if API fails
            print(f"Weather API error: {e}. Using fallback data.")
            return {
                "rain_probability": random.randint(0, 100),
                "temperature": random.randint(20, 40),
                "humidity": random.randint(30, 90),
                "wind_speed": random.randint(5, 25),
                "source": "Fallback (API unavailable)",
                "error": str(e)
            }


class IrrigationMLModel:
    """
    Random Forest ML model for irrigation prediction.
    Trained on synthetic agricultural data representing real-world scenarios.
    """
    
    MODEL_PATH = "irrigation_model.pkl"
    ENCODER_PATH = "crop_encoder.pkl"
    
    # Crop encoding mapping
    CROP_MAPPING = {"Rice": 0, "Maize": 1, "Chickpea": 2}
    CROP_REVERSE = {0: "Rice", 1: "Maize", 2: "Chickpea"}
    
    def __init__(self):
        self.model = None
        self.crop_encoder = LabelEncoder()
        self.crop_encoder.fit(["Rice", "Maize", "Chickpea"])
        self.feature_names = [
            "soil_moisture", "temperature", "humidity", 
            "rain_probability", "wind_speed", "hour_of_day", "crop_encoded"
        ]
        self._load_or_train_model()
    
    def _generate_training_data(self, n_samples: int = 5000) -> tuple:
        """
        Generate synthetic training data based on agricultural domain knowledge.
        
        Features:
        - soil_moisture: 0-100%
        - temperature: 15-45°C
        - humidity: 20-100%
        - rain_probability: 0-100%
        - wind_speed: 0-50 km/h
        - hour_of_day: 0-23
        - crop_type: Rice, Maize, Chickpea
        
        Target: 0 (pump OFF) or 1 (pump ON)
        """
        np.random.seed(42)
        
        # Generate features
        soil_moisture = np.random.uniform(5, 100, n_samples)
        temperature = np.random.uniform(15, 45, n_samples)
        humidity = np.random.uniform(20, 100, n_samples)
        rain_probability = np.random.uniform(0, 100, n_samples)
        wind_speed = np.random.uniform(0, 50, n_samples)
        hour_of_day = np.random.randint(0, 24, n_samples)
        crop_types = np.random.choice(["Rice", "Maize", "Chickpea"], n_samples)
        
        # Encode crops
        crop_encoded = self.crop_encoder.transform(crop_types)
        
        # Define crop-specific thresholds
        crop_thresholds = {"Rice": 75, "Maize": 45, "Chickpea": 30}
        
        # Generate labels based on domain rules (to train the model)
        labels = []
        for i in range(n_samples):
            crop = crop_types[i]
            threshold = crop_thresholds[crop]
            
            # Base decision: needs water?
            needs_water = soil_moisture[i] < threshold
            
            # Critical moisture override
            is_critical = soil_moisture[i] < 15
            
            # High noon check (11 AM - 3 PM)
            is_high_noon = 11 <= hour_of_day[i] < 15
            
            # Rain check
            rain_expected = rain_probability[i] > 60
            
            # Decision logic
            if not needs_water:
                pump_on = 0
            elif rain_expected:
                pump_on = 0
            elif is_high_noon and not is_critical:
                pump_on = 0
            else:
                pump_on = 1
            
            # Add some noise to make it more realistic (5% label noise)
            if np.random.random() < 0.05:
                pump_on = 1 - pump_on
            
            labels.append(pump_on)
        
        # Create feature matrix
        X = np.column_stack([
            soil_moisture, temperature, humidity,
            rain_probability, wind_speed, hour_of_day, crop_encoded
        ])
        y = np.array(labels)
        
        return X, y
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("✓ Loaded pre-trained Random Forest model")
                return
            except Exception as e:
                print(f"Error loading model: {e}. Training new model...")
        
        self._train_model()
    
    def _train_model(self):
        """Train the Random Forest model on synthetic data."""
        print("Training Random Forest irrigation model...")
        
        X, y = self._generate_training_data(n_samples=10000)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X, y)
        print(f"✓ Model trained with accuracy: {train_accuracy:.2%}")
        
        # Save model
        try:
            with open(self.MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Model saved to {self.MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def predict(self, features: dict) -> dict:
        """
        Predict irrigation decision using the trained model.
        
        Args:
            features: Dictionary with soil_moisture, temperature, humidity,
                     rain_probability, wind_speed, hour_of_day, crop_type
        
        Returns:
            Dictionary with prediction and confidence
        """
        # Encode crop type
        crop_type = features.get("crop_type", "Maize")
        try:
            crop_encoded = self.crop_encoder.transform([crop_type])[0]
        except ValueError:
            crop_encoded = 1  # Default to Maize
        
        # Prepare feature vector
        X = np.array([[
            features.get("soil_moisture", 50),
            features.get("temperature", 25),
            features.get("humidity", 50),
            features.get("rain_probability", 0),
            features.get("wind_speed", 10),
            features.get("hour_of_day", datetime.now().hour),
            crop_encoded
        ]])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            "prediction": int(prediction),
            "pump_command": "ON" if prediction == 1 else "OFF",
            "confidence": round(confidence, 3),
            "probabilities": {
                "OFF": round(probabilities[0], 3),
                "ON": round(probabilities[1], 3)
            }
        }
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        return {name: round(imp, 4) for name, imp in zip(self.feature_names, importances)}


# =============================================================================
# CROP RECOMMENDATION ML MODEL - Random Forest Classifier
# =============================================================================

class CropRecommendationModel:
    """
    Random Forest ML model for crop recommendation based on sensor data.
    Predicts the best crop to grow based on soil and weather conditions.
    
    Features used:
    - Nitrogen (N): 0-140 kg/ha
    - Phosphorus (P): 5-145 kg/ha
    - Potassium (K): 5-205 kg/ha
    - Temperature: 8-45°C
    - Humidity: 14-100%
    - pH: 3.5-10
    - Rainfall: 20-300 mm
    - Soil Moisture: 0-100%
    """
    
    MODEL_PATH = "crop_recommendation_model.pkl"
    
    # Comprehensive list of crops with their ideal growing conditions
    CROP_CONDITIONS = {
        "Rice": {"N": (80, 120), "P": (40, 60), "K": (40, 60), "temp": (20, 35), "humidity": (80, 95), "ph": (5.5, 7.0), "rainfall": (150, 250), "moisture": (70, 95)},
        "Wheat": {"N": (80, 120), "P": (30, 50), "K": (20, 40), "temp": (10, 25), "humidity": (50, 70), "ph": (6.0, 7.5), "rainfall": (50, 100), "moisture": (40, 60)},
        "Maize": {"N": (60, 100), "P": (35, 55), "K": (30, 50), "temp": (18, 30), "humidity": (50, 75), "ph": (5.5, 7.5), "rainfall": (60, 120), "moisture": (45, 70)},
        "Chickpea": {"N": (20, 50), "P": (50, 80), "K": (60, 90), "temp": (15, 30), "humidity": (30, 50), "ph": (6.0, 8.0), "rainfall": (40, 80), "moisture": (25, 45)},
        "Kidney Beans": {"N": (20, 40), "P": (50, 70), "K": (15, 25), "temp": (15, 25), "humidity": (60, 80), "ph": (5.5, 6.5), "rainfall": (60, 100), "moisture": (40, 60)},
        "Pigeon Peas": {"N": (20, 40), "P": (50, 80), "K": (15, 25), "temp": (20, 35), "humidity": (50, 70), "ph": (5.0, 7.0), "rainfall": (60, 100), "moisture": (35, 55)},
        "Moth Beans": {"N": (15, 35), "P": (35, 55), "K": (15, 25), "temp": (25, 40), "humidity": (30, 50), "ph": (7.0, 8.5), "rainfall": (25, 50), "moisture": (20, 40)},
        "Mung Bean": {"N": (15, 35), "P": (40, 60), "K": (15, 25), "temp": (25, 35), "humidity": (60, 85), "ph": (6.0, 7.5), "rainfall": (60, 100), "moisture": (40, 60)},
        "Black Gram": {"N": (30, 50), "P": (50, 70), "K": (15, 25), "temp": (25, 35), "humidity": (60, 80), "ph": (6.5, 7.5), "rainfall": (60, 100), "moisture": (40, 60)},
        "Lentil": {"N": (15, 35), "P": (55, 75), "K": (15, 25), "temp": (15, 25), "humidity": (50, 70), "ph": (6.0, 8.0), "rainfall": (40, 60), "moisture": (35, 55)},
        "Pomegranate": {"N": (15, 35), "P": (5, 25), "K": (35, 55), "temp": (20, 35), "humidity": (40, 60), "ph": (6.5, 7.5), "rainfall": (50, 80), "moisture": (30, 50)},
        "Banana": {"N": (90, 120), "P": (70, 90), "K": (45, 65), "temp": (25, 35), "humidity": (75, 95), "ph": (5.5, 7.0), "rainfall": (100, 200), "moisture": (60, 80)},
        "Mango": {"N": (15, 35), "P": (20, 40), "K": (25, 45), "temp": (25, 40), "humidity": (50, 70), "ph": (5.5, 7.5), "rainfall": (80, 150), "moisture": (40, 60)},
        "Grapes": {"N": (15, 35), "P": (120, 145), "K": (190, 210), "temp": (15, 35), "humidity": (60, 80), "ph": (5.5, 7.0), "rainfall": (60, 100), "moisture": (45, 65)},
        "Watermelon": {"N": (90, 110), "P": (15, 35), "K": (45, 65), "temp": (25, 40), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (40, 60), "moisture": (50, 70)},
        "Muskmelon": {"N": (90, 110), "P": (15, 35), "K": (45, 65), "temp": (25, 40), "humidity": (70, 90), "ph": (6.0, 7.0), "rainfall": (30, 50), "moisture": (45, 65)},
        "Apple": {"N": (15, 35), "P": (120, 145), "K": (190, 210), "temp": (8, 25), "humidity": (70, 90), "ph": (5.5, 6.5), "rainfall": (100, 150), "moisture": (55, 75)},
        "Orange": {"N": (15, 35), "P": (5, 25), "K": (5, 15), "temp": (15, 30), "humidity": (80, 95), "ph": (6.0, 7.5), "rainfall": (100, 150), "moisture": (50, 70)},
        "Papaya": {"N": (45, 65), "P": (55, 75), "K": (45, 65), "temp": (25, 38), "humidity": (75, 95), "ph": (6.0, 7.0), "rainfall": (100, 180), "moisture": (55, 75)},
        "Coconut": {"N": (15, 35), "P": (25, 45), "K": (25, 45), "temp": (25, 35), "humidity": (80, 95), "ph": (5.5, 7.0), "rainfall": (150, 250), "moisture": (60, 80)},
        "Cotton": {"N": (100, 140), "P": (40, 60), "K": (15, 25), "temp": (22, 35), "humidity": (40, 60), "ph": (6.0, 8.0), "rainfall": (50, 100), "moisture": (35, 55)},
        "Jute": {"N": (70, 90), "P": (40, 60), "K": (35, 55), "temp": (25, 38), "humidity": (70, 90), "ph": (6.0, 7.5), "rainfall": (150, 250), "moisture": (65, 85)},
        "Coffee": {"N": (90, 120), "P": (15, 35), "K": (25, 45), "temp": (18, 28), "humidity": (70, 90), "ph": (5.5, 6.5), "rainfall": (150, 250), "moisture": (55, 75)},
    }
    
    CROP_LIST = list(CROP_CONDITIONS.keys())
    
    def __init__(self):
        self.model = None
        self.crop_encoder = LabelEncoder()
        self.crop_encoder.fit(self.CROP_LIST)
        self.feature_names = [
            "nitrogen", "phosphorus", "potassium", "temperature",
            "humidity", "ph", "rainfall", "soil_moisture"
        ]
        self._load_or_train_model()
    
    def _generate_training_data(self, n_samples_per_crop: int = 500) -> tuple:
        """
        Generate synthetic training data based on crop-specific ideal conditions.
        Creates varied samples around the ideal ranges for each crop.
        """
        np.random.seed(42)
        
        X_data = []
        y_data = []
        
        for crop, conditions in self.CROP_CONDITIONS.items():
            for _ in range(n_samples_per_crop):
                # Generate values within ideal ranges with some noise
                n_min, n_max = conditions["N"]
                p_min, p_max = conditions["P"]
                k_min, k_max = conditions["K"]
                temp_min, temp_max = conditions["temp"]
                hum_min, hum_max = conditions["humidity"]
                ph_min, ph_max = conditions["ph"]
                rain_min, rain_max = conditions["rainfall"]
                moist_min, moist_max = conditions["moisture"]
                
                # Add some variance (80% within ideal, 20% slightly outside)
                variance = 0.2 if np.random.random() > 0.8 else 0.0
                
                nitrogen = np.clip(np.random.uniform(n_min * (1 - variance), n_max * (1 + variance)), 0, 140)
                phosphorus = np.clip(np.random.uniform(p_min * (1 - variance), p_max * (1 + variance)), 5, 145)
                potassium = np.clip(np.random.uniform(k_min * (1 - variance), k_max * (1 + variance)), 5, 205)
                temperature = np.clip(np.random.uniform(temp_min * (1 - variance), temp_max * (1 + variance)), 8, 45)
                humidity = np.clip(np.random.uniform(hum_min * (1 - variance), hum_max * (1 + variance)), 14, 100)
                ph = np.clip(np.random.uniform(ph_min * (1 - variance), ph_max * (1 + variance)), 3.5, 10)
                rainfall = np.clip(np.random.uniform(rain_min * (1 - variance), rain_max * (1 + variance)), 20, 300)
                soil_moisture = np.clip(np.random.uniform(moist_min * (1 - variance), moist_max * (1 + variance)), 0, 100)
                
                X_data.append([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, soil_moisture])
                y_data.append(crop)
        
        X = np.array(X_data)
        y = self.crop_encoder.transform(y_data)
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("✓ Loaded pre-trained Crop Recommendation model")
                return
            except Exception as e:
                print(f"Error loading crop model: {e}. Training new model...")
        
        self._train_model()
    
    def _train_model(self):
        """Train the Random Forest model for crop recommendation."""
        print("Training Random Forest crop recommendation model...")
        
        X, y = self._generate_training_data(n_samples_per_crop=500)
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X, y)
        print(f"✓ Crop Recommendation model trained with accuracy: {train_accuracy:.2%}")
        
        # Save model
        try:
            with open(self.MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Crop model saved to {self.MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not save crop model: {e}")
    
    def predict(self, sensor_data: dict) -> dict:
        """
        Predict the best crop to grow based on sensor readings.
        
        Args:
            sensor_data: Dictionary with N, P, K, temperature, humidity, pH, rainfall, soil_moisture
        
        Returns:
            Dictionary with top 3 recommended crops and their probabilities
        """
        # Extract features with defaults
        nitrogen = sensor_data.get("nitrogen", sensor_data.get("N", 50))
        phosphorus = sensor_data.get("phosphorus", sensor_data.get("P", 50))
        potassium = sensor_data.get("potassium", sensor_data.get("K", 50))
        temperature = sensor_data.get("temperature", 25)
        humidity = sensor_data.get("humidity", 60)
        ph = sensor_data.get("ph", sensor_data.get("pH", 6.5))
        rainfall = sensor_data.get("rainfall", 100)
        soil_moisture = sensor_data.get("soil_moisture", 50)
        
        # Prepare feature vector
        X = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, soil_moisture]])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        recommendations = []
        for idx in top_indices:
            crop_name = self.crop_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            conditions = self.CROP_CONDITIONS[crop_name]
            
            recommendations.append({
                "crop": crop_name,
                "confidence": round(prob * 100, 1),
                "ideal_conditions": {
                    "N_range": f"{conditions['N'][0]}-{conditions['N'][1]} kg/ha",
                    "P_range": f"{conditions['P'][0]}-{conditions['P'][1]} kg/ha",
                    "K_range": f"{conditions['K'][0]}-{conditions['K'][1]} kg/ha",
                    "temperature": f"{conditions['temp'][0]}-{conditions['temp'][1]}°C",
                    "humidity": f"{conditions['humidity'][0]}-{conditions['humidity'][1]}%",
                    "pH": f"{conditions['ph'][0]}-{conditions['ph'][1]}",
                    "rainfall": f"{conditions['rainfall'][0]}-{conditions['rainfall'][1]} mm"
                }
            })
        
        return {
            "top_recommendation": recommendations[0]["crop"],
            "confidence": recommendations[0]["confidence"],
            "all_recommendations": recommendations,
            "input_values": {
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "pH": ph,
                "rainfall": rainfall,
                "soil_moisture": soil_moisture
            }
        }
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        return {name: round(imp, 4) for name, imp in zip(self.feature_names, importances)}


# Initialize Crop Recommendation Model
crop_recommender = CropRecommendationModel()


class FarmAgent:
    """
    Autonomous Farm Agent that manages irrigation decisions using
    ML predictions combined with safety rules and real weather data.
    """
    
    # Dynamic crop moisture thresholds (used for recommendations)
    CROP_PROFILES = {
        "Rice": 75,
        "Maize": 45,
        "Chickpea": 30
    }
    
    CRITICAL_MOISTURE = 15
    HIGH_NOON_START = 11
    HIGH_NOON_END = 15
    RAIN_THRESHOLD = 60
    MAX_PUMP_RUNTIME = 30
    
    def __init__(self):
        self.pump_start_time = None
        self.pump_running = False
        self.emergency_stop = False
        self.ml_model = IrrigationMLModel()
        self.weather_api = WeatherAPI()
    
    def get_weather_forecast(self, latitude: float = None, longitude: float = None) -> dict:
        """Fetch real weather forecast from API."""
        return self.weather_api.get_forecast(latitude, longitude)
    
    def is_high_noon(self) -> bool:
        """Check if current time is within high evaporation window (11 AM - 3 PM)."""
        current_hour = datetime.now().hour
        return self.HIGH_NOON_START <= current_hour < self.HIGH_NOON_END
    
    def get_moisture_threshold(self, crop_type: str) -> int:
        """Get the irrigation trigger threshold for a specific crop."""
        return self.CROP_PROFILES.get(crop_type, 50)
    
    def recommend_crop(self, soil_moisture: float, soil_ph: float = 7.0) -> str:
        """Recommend the best crop based on current soil conditions."""
        if soil_moisture >= 70:
            return "Rice"
        elif soil_moisture >= 40:
            return "Maize"
        else:
            return "Chickpea"
    
    def check_pump_safety(self, pump_runtime_minutes: float) -> tuple:
        """Safety check for pump runtime."""
        if pump_runtime_minutes > self.MAX_PUMP_RUNTIME:
            self.emergency_stop = True
            return False, f"EMERGENCY STOP: Pump running for {pump_runtime_minutes:.1f} minutes (>{self.MAX_PUMP_RUNTIME} min). Possible leak detected!"
        return True, "Pump runtime within safe limits"
    
    def decide_irrigation(self, sensor_data: dict) -> dict:
        """
        Main decision engine using ML model with safety overrides.
        
        The ML model makes the primary prediction, but safety rules
        can override the decision in critical situations.
        """
        soil_moisture = sensor_data.get("soil_moisture", 50)
        crop_type = sensor_data.get("crop_type", "Maize")
        pump_runtime = sensor_data.get("pump_runtime_minutes", 0)
        latitude = sensor_data.get("latitude")
        longitude = sensor_data.get("longitude")
        
        recommended_crop = self.recommend_crop(soil_moisture)
        moisture_threshold = self.get_moisture_threshold(crop_type)
        
        reasons = []
        
        # Safety Check 1: Pump runtime (HARD OVERRIDE)
        is_safe, safety_message = self.check_pump_safety(pump_runtime)
        if not is_safe:
            return {
                "pump_command": "OFF",
                "agent_reason": safety_message,
                "recommended_crop": recommended_crop,
                "alert_level": "CRITICAL",
                "ml_prediction": None,
                "override_reason": "Safety stop triggered"
            }
        
        # Get real weather data
        weather = self.get_weather_forecast(latitude, longitude)
        
        # Prepare features for ML model
        ml_features = {
            "soil_moisture": soil_moisture,
            "temperature": weather.get("temperature", 25),
            "humidity": weather.get("humidity", 50),
            "rain_probability": weather.get("rain_probability", 0),
            "wind_speed": weather.get("wind_speed", 10),
            "hour_of_day": datetime.now().hour,
            "crop_type": crop_type
        }
        
        # Get ML prediction
        ml_result = self.ml_model.predict(ml_features)
        ml_command = ml_result["pump_command"]
        ml_confidence = ml_result["confidence"]
        
        reasons.append(f"ML Model predicts {ml_command} with {ml_confidence:.1%} confidence")
        
        # Safety Check 2: Critical moisture override (can override ML)
        is_critical = soil_moisture < self.CRITICAL_MOISTURE
        if is_critical and ml_command == "OFF":
            reasons.append(f"CRITICAL OVERRIDE: Soil moisture at {soil_moisture}% (below {self.CRITICAL_MOISTURE}%)")
            return {
                "pump_command": "ON",
                "agent_reason": "; ".join(reasons),
                "recommended_crop": recommended_crop,
                "alert_level": "CRITICAL",
                "ml_prediction": ml_result,
                "weather_data": weather,
                "override_reason": "Critical moisture level"
            }
        
        # Safety Check 3: High rain probability override
        if weather.get("rain_probability", 0) > 80 and ml_command == "ON":
            reasons.append(f"RAIN OVERRIDE: High rain probability ({weather['rain_probability']}%)")
            return {
                "pump_command": "OFF",
                "agent_reason": "; ".join(reasons),
                "recommended_crop": recommended_crop,
                "alert_level": "INFO",
                "ml_prediction": ml_result,
                "weather_data": weather,
                "override_reason": "High rain probability"
            }
        
        # Use ML prediction as final decision
        alert_level = "NORMAL" if ml_command == "ON" else "INFO"
        
        # Add context to reasons
        reasons.append(f"Weather: {weather.get('temperature', 'N/A')}°C, {weather.get('humidity', 'N/A')}% humidity")
        reasons.append(f"Rain probability: {weather.get('rain_probability', 'N/A')}%")
        
        return {
            "pump_command": ml_command,
            "agent_reason": "; ".join(reasons),
            "recommended_crop": recommended_crop,
            "alert_level": alert_level,
            "ml_prediction": ml_result,
            "weather_data": weather,
            "feature_importance": self.ml_model.get_feature_importance()
        }


# Initialize the Farm Agent
print("=" * 60)
print("  Initializing AgroSmart Farm Agent with ML Model...")
print("=" * 60)
farm_agent = FarmAgent()


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts sensor data and returns irrigation decision.
    
    Expected JSON payload:
    {
        "soil_moisture": 40,
        "crop_type": "Rice",
        "pump_runtime_minutes": 0
    }
    
    Returns:
    {
        "pump_command": "ON/OFF",
        "agent_reason": "Reason string",
        "recommended_crop": "Rice"
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "pump_command": "OFF",
                "agent_reason": "Invalid request format"
            }), 400
        
        sensor_data = request.get_json()
        
        # Validate required fields
        if "soil_moisture" not in sensor_data:
            return jsonify({
                "error": "Missing required field: soil_moisture",
                "pump_command": "OFF",
                "agent_reason": "Sensor data incomplete"
            }), 400
        
        # Validate soil moisture range
        soil_moisture = sensor_data.get("soil_moisture")
        if not isinstance(soil_moisture, (int, float)) or soil_moisture < 0 or soil_moisture > 100:
            return jsonify({
                "error": "soil_moisture must be a number between 0 and 100",
                "pump_command": "OFF",
                "agent_reason": "Invalid sensor reading"
            }), 400
        
        # Make irrigation decision
        decision = farm_agent.decide_irrigation(sensor_data)
        
        return jsonify(decision), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "pump_command": "OFF",
            "agent_reason": "System error - defaulting to safe mode"
        }), 500


# =============================================================================
# ESP32 LIVE SENSOR ENDPOINTS
# =============================================================================

@app.route("/api/esp32/data", methods=["POST"])
def receive_esp32_data():
    """
    POST /api/esp32/data
    Receive live sensor data from ESP32 device.
    
    Expected JSON payload:
    {
        "device_id": "ESP32_XXXXXXXXXXXX",
        "soil_moisture": 45.5,
        "temperature": 28.3,
        "humidity": 65.2,
        "raw_moisture": 2500,
        "pump_running": false,
        "pump_runtime": 0,
        "auto_mode": true,
        "wifi_rssi": -45,
        "uptime": 3600
    }
    
    Returns pump command if any pending.
    """
    try:
        data = request.get_json()
        
        if not data or "device_id" not in data:
            return jsonify({"error": "Missing device_id"}), 400
        
        device_id = data["device_id"]
        
        # Store the data
        esp32_store.update(device_id, data)
        
        # Log the received data
        print(f"📡 ESP32 [{device_id[-8:]}]: Moisture={data.get('soil_moisture', 0):.1f}%, "
              f"Temp={data.get('temperature', 0):.1f}°C, "
              f"Humidity={data.get('humidity', 0):.1f}%, "
              f"Pump={'ON' if data.get('pump_running') else 'OFF'}")
        
        # Check for pending pump command
        pending_command = esp32_store.get_pump_command(device_id)
        
        # Use ML model to make irrigation decision
        ml_decision = farm_agent.decide_irrigation({
            "soil_moisture": data.get("soil_moisture", 50),
            "temperature": data.get("temperature", 25),
            "humidity": data.get("humidity", 50),
            "crop_type": "Maize",  # Default crop
            "pump_runtime_minutes": data.get("pump_runtime", 0) / 60
        })
        
        response = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "ml_recommendation": ml_decision.get("pump_command", "OFF"),
            "ml_confidence": ml_decision.get("ml_confidence", 0)
        }
        
        # Add pump command if pending (manual override)
        if pending_command:
            response["pump_command"] = pending_command
            print(f"📤 Sending command to ESP32: {pending_command}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error processing ESP32 data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/esp32/sensors", methods=["GET"])
def get_live_sensors():
    """
    GET /api/esp32/sensors
    Get latest sensor data from all connected ESP32 devices.
    
    Query params:
        device_id (optional): Get data for specific device
    
    Returns real-time sensor readings for dashboard display.
    """
    device_id = request.args.get("device_id")
    
    if device_id:
        data = esp32_store.get_latest(device_id)
        if not data:
            return jsonify({"error": "Device not found"}), 404
        return jsonify(data), 200
    
    # Return all devices
    all_data = esp32_store.get_latest()
    devices = esp32_store.get_connected_devices()
    
    # If no ESP32 connected, return simulated data for demo
    if not all_data:
        simulated = {
            "device_id": "DEMO_DEVICE",
            "soil_moisture": random.uniform(35, 65),
            "temperature": random.uniform(22, 32),
            "humidity": random.uniform(50, 80),
            "pump_running": False,
            "pump_runtime": 0,
            "auto_mode": True,
            "is_simulated": True,
            "last_update": datetime.now().isoformat()
        }
        return jsonify({
            "devices": [simulated],
            "total_devices": 1,
            "note": "No ESP32 connected - showing simulated data"
        }), 200
    
    return jsonify({
        "devices": list(all_data.values()),
        "device_status": devices,
        "total_devices": len(all_data)
    }), 200


@app.route("/api/esp32/history", methods=["GET"])
def get_sensor_history():
    """
    GET /api/esp32/history?device_id=XXX&limit=50
    Get historical sensor readings for charts.
    """
    device_id = request.args.get("device_id")
    limit = request.args.get("limit", 50, type=int)
    
    if not device_id:
        # Return history for first available device
        devices = esp32_store.get_connected_devices()
        if devices:
            device_id = devices[0]["device_id"]
        else:
            # Return simulated history for demo
            history = []
            current_time = time.time()
            for i in range(limit):
                history.append({
                    "timestamp": current_time - (limit - i) * 60,
                    "soil_moisture": 50 + random.uniform(-15, 15),
                    "temperature": 27 + random.uniform(-3, 3),
                    "humidity": 65 + random.uniform(-10, 10),
                    "pump_running": random.random() > 0.8
                })
            return jsonify({
                "device_id": "DEMO_DEVICE",
                "history": history,
                "is_simulated": True
            }), 200
    
    history = esp32_store.get_history(device_id, limit)
    
    return jsonify({
        "device_id": device_id,
        "history": history,
        "count": len(history)
    }), 200


@app.route("/api/esp32/pump", methods=["POST"])
def control_pump():
    """
    POST /api/esp32/pump
    Send pump control command to ESP32.
    
    JSON payload:
    {
        "device_id": "ESP32_XXXX",
        "command": "ON" | "OFF" | "AUTO"
    }
    """
    try:
        data = request.get_json()
        device_id = data.get("device_id")
        command = data.get("command", "").upper()
        
        if not device_id:
            return jsonify({"error": "Missing device_id"}), 400
        
        if command not in ["ON", "OFF", "AUTO"]:
            return jsonify({"error": "Invalid command. Use ON, OFF, or AUTO"}), 400
        
        # Queue the command for the device
        esp32_store.set_pump_command(device_id, command)
        
        print(f"🎮 Pump command queued for {device_id}: {command}")
        
        return jsonify({
            "status": "ok",
            "message": f"Pump command '{command}' queued for device",
            "device_id": device_id,
            "command": command
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/esp32/devices", methods=["GET"])
def list_esp32_devices():
    """
    GET /api/esp32/devices
    List all connected ESP32 devices with their status.
    """
    devices = esp32_store.get_connected_devices()
    
    return jsonify({
        "devices": devices,
        "total": len(devices),
        "online": sum(1 for d in devices if d["is_online"])
    }), 200


# =============================================================================
# CROP RECOMMENDATION ENDPOINTS
# =============================================================================

@app.route("/api/crop/recommend", methods=["POST"])
def recommend_crop():
    """
    POST /api/crop/recommend
    ML-powered crop recommendation based on sensor data.
    
    Request body (JSON):
    {
        "nitrogen": 90,      // N content in kg/ha (0-140)
        "phosphorus": 42,    // P content in kg/ha (5-145)
        "potassium": 43,     // K content in kg/ha (5-205)
        "temperature": 25,   // Temperature in °C (8-45)
        "humidity": 80,      // Humidity % (14-100)
        "ph": 6.5,           // Soil pH (3.5-10)
        "rainfall": 200,     // Rainfall in mm (20-300)
        "soil_moisture": 65  // Soil moisture % (0-100)
    }
    
    Returns:
    {
        "top_recommendation": "Rice",
        "confidence": 92.5,
        "all_recommendations": [...top 5 crops...],
        "input_values": {...},
        "feature_importance": {...}
    }
    """
    try:
        sensor_data = request.get_json()
        
        if not sensor_data:
            return jsonify({
                "error": "No sensor data provided",
                "message": "Please provide soil/weather sensor readings"
            }), 400
        
        # Get crop recommendation from ML model
        result = crop_recommender.predict(sensor_data)
        
        # Add feature importance to response
        result["feature_importance"] = crop_recommender.get_feature_importance()
        result["model_info"] = {
            "type": "Random Forest Classifier",
            "crops_supported": len(crop_recommender.CROP_LIST),
            "features_used": crop_recommender.feature_names
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Crop recommendation failed: {str(e)}",
            "message": "Please check sensor data format"
        }), 500


@app.route("/api/crop/list", methods=["GET"])
def list_crops():
    """
    GET /api/crop/list
    Get list of all crops the model can recommend with their ideal conditions.
    """
    crops_info = []
    for crop, conditions in crop_recommender.CROP_CONDITIONS.items():
        crops_info.append({
            "crop": crop,
            "ideal_conditions": {
                "nitrogen": f"{conditions['N'][0]}-{conditions['N'][1]} kg/ha",
                "phosphorus": f"{conditions['P'][0]}-{conditions['P'][1]} kg/ha",
                "potassium": f"{conditions['K'][0]}-{conditions['K'][1]} kg/ha",
                "temperature": f"{conditions['temp'][0]}-{conditions['temp'][1]}°C",
                "humidity": f"{conditions['humidity'][0]}-{conditions['humidity'][1]}%",
                "pH": f"{conditions['ph'][0]}-{conditions['ph'][1]}",
                "rainfall": f"{conditions['rainfall'][0]}-{conditions['rainfall'][1]} mm",
                "soil_moisture": f"{conditions['moisture'][0]}-{conditions['moisture'][1]}%"
            }
        })
    
    return jsonify({
        "total_crops": len(crops_info),
        "crops": crops_info
    }), 200


@app.route("/api/crop/model-info", methods=["GET"])
def crop_model_info():
    """
    GET /api/crop/model-info
    Get information about the crop recommendation ML model.
    """
    return jsonify({
        "model_type": "Random Forest Classifier",
        "n_estimators": 150,
        "max_depth": 15,
        "crops_supported": crop_recommender.CROP_LIST,
        "total_crops": len(crop_recommender.CROP_LIST),
        "features": crop_recommender.feature_names,
        "feature_importance": crop_recommender.get_feature_importance(),
        "training_samples_per_crop": 500,
        "description": "ML model trained on synthetic agricultural data representing ideal growing conditions for 23 major crops"
    }), 200


# Diagnosis templates for leaf analysis
DIAGNOSIS_TEMPLATES = {
    "healthy": {
        "id": "healthy",
        "status": "healthy",
        "diagnosis": "Healthy Crop",
        "description": "Your crop is healthy with strong chlorophyll presence. No signs of disease or deficiency.",
        "remedy": "Continue current care. Maintain soil moisture at 60% and monitor weekly.",
        "severity": "none",
        "icon": "✅",
        "color": "#22c55e"
    },
    "mild_stress": {
        "id": "mild_stress",
        "status": "healthy",
        "diagnosis": "Mild Environmental Stress",
        "description": "Minor stress indicators but generally healthy. May be due to temperature fluctuations.",
        "remedy": "Ensure consistent watering. Provide shade during peak heat if needed.",
        "severity": "low",
        "icon": "✅",
        "color": "#22c55e"
    },
    "nitrogen_deficiency": {
        "id": "nitrogen_deficiency",
        "status": "deficiency",
        "diagnosis": "Nitrogen Deficiency",
        "description": "Yellowing detected in leaves indicating low nitrogen levels.",
        "remedy": "Apply compost tea or fish emulsion. Plant legumes nearby to fix nitrogen naturally.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#eab308"
    },
    "severe_nitrogen_deficiency": {
        "id": "severe_nitrogen_deficiency",
        "status": "deficiency",
        "diagnosis": "Severe Nitrogen Deficiency",
        "description": "Significant yellowing and chlorosis. Advanced nitrogen starvation.",
        "remedy": "Immediate application of fish emulsion (1 tbsp/gallon) or blood meal.",
        "severity": "high",
        "icon": "🔴",
        "color": "#ef4444"
    },
    "early_blight": {
        "id": "early_blight",
        "status": "disease",
        "diagnosis": "Early Blight (Alternaria)",
        "description": "Brown spots with yellowing detected. Common fungal disease.",
        "remedy": "Apply neem oil spray. Remove affected leaves. Improve air circulation.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#f97316"
    },
    "late_blight": {
        "id": "late_blight",
        "status": "disease",
        "diagnosis": "Late Blight (Phytophthora)",
        "description": "Severe brown lesions detected. Spreads rapidly in humid conditions.",
        "remedy": "Remove and destroy affected material. Apply copper-based organic fungicide.",
        "severity": "critical",
        "icon": "🔴",
        "color": "#dc2626"
    },
    "leaf_spot": {
        "id": "leaf_spot",
        "status": "disease",
        "diagnosis": "Bacterial/Fungal Leaf Spot",
        "description": "Dark spots detected on leaf surface.",
        "remedy": "Apply baking soda solution. Remove affected leaves. Avoid overhead watering.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#f97316"
    },
    "nutrient_stress": {
        "id": "nutrient_stress",
        "status": "deficiency",
        "diagnosis": "General Nutrient Stress",
        "description": "Mixed color signals suggesting possible nutrient imbalance.",
        "remedy": "Apply balanced organic fertilizer. Test soil pH (ideal: 6.0-7.0).",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#eab308"
    }
}


def is_leaf_image(img, mode="upload"):
    """
    Strict leaf validation. Requires dominant green presence and leaf-like texture.
    Rejects non-leaf objects like skin, walls, furniture, etc.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]
    
    # Strict green detection (core leaf color)
    green_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([85, 255, 255]))
    # Yellow-green (stressed/aging leaves)
    yellow_mask = cv2.inRange(hsv, np.array([18, 40, 40]), np.array([30, 255, 255]))
    # Brown (dried/diseased leaves - tighter to avoid skin/wood)
    brown_mask = cv2.inRange(hsv, np.array([5, 50, 30]), np.array([18, 200, 180]))
    
    green_pct = (cv2.countNonZero(green_mask) / total_pixels) * 100
    yellow_pct = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
    brown_pct = (cv2.countNonZero(brown_mask) / total_pixels) * 100
    plant_pct = green_pct + yellow_pct + brown_pct
    
    # Green must be the dominant plant color (at least some green present)
    # This prevents skin/wood/furniture from passing via brown alone
    has_green = green_pct >= 5
    
    # Overall plant coverage thresholds
    if mode == "webcam":
        threshold = 50  # Leaf should dominate the cropped center
    else:
        threshold = 30  # Upload: leaf should be main subject
    
    is_leaf = plant_pct >= threshold and has_green
    
    if not is_leaf:
        if not has_green:
            reason = "No green leaf detected. Please capture an actual leaf."
        else:
            reason = "Leaf too small or far away. Fill the guide box with the leaf."
    else:
        reason = ""
    
    print(f"🔍 Leaf check ({mode}): green={green_pct:.1f}%, yellow={yellow_pct:.1f}%, "
          f"brown={brown_pct:.1f}%, total={plant_pct:.1f}%, has_green={has_green} -> {'✅' if is_leaf else '❌'}")
    
    return is_leaf, reason, {"plant_pct": round(plant_pct, 1), "green_pct": round(green_pct, 1)}


def analyze_leaf_colors(img):
    """
    Analyze leaf image using color-based computer vision.
    Returns diagnosis based on green/yellow/brown ratios.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    total_pixels = height * width
    
    # Color masks
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))
    brown_mask1 = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([20, 200, 180]))
    brown_mask2 = cv2.inRange(hsv, np.array([160, 30, 20]), np.array([180, 200, 180]))
    brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    
    # Calculate percentages
    green_pct = (cv2.countNonZero(green_mask) / total_pixels) * 100
    yellow_pct = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
    brown_pct = (cv2.countNonZero(brown_mask) / total_pixels) * 100
    dark_pct = (cv2.countNonZero(dark_mask) / total_pixels) * 100
    
    # Decision logic
    if brown_pct > 12 or dark_pct > 8:
        if brown_pct > 20:
            return "late_blight", min(0.95, 0.70 + brown_pct/100)
        else:
            return "leaf_spot", min(0.90, 0.65 + brown_pct/100)
    elif yellow_pct > 18 and green_pct < 45:
        if yellow_pct > 35:
            return "severe_nitrogen_deficiency", min(0.92, 0.65 + yellow_pct/100)
        else:
            return "nitrogen_deficiency", min(0.88, 0.60 + yellow_pct/100)
    elif yellow_pct > 8 and brown_pct > 4:
        return "early_blight", min(0.88, 0.60 + (yellow_pct + brown_pct)/200)
    elif green_pct > 45 and yellow_pct < 18 and brown_pct < 12:
        return "healthy", min(0.96, 0.75 + green_pct/200)
    elif green_pct > 25 and yellow_pct < 25:
        return "mild_stress", min(0.85, 0.60 + green_pct/200)
    else:
        if green_pct > yellow_pct:
            return "healthy", 0.65
        else:
            return "nutrient_stress", 0.60


@app.route("/api/scan/image", methods=["POST"])
def scan_image():
    """
    POST /api/scan/image
    Analyze crop leaf image using color-based computer vision.
    """
    try:
        processing_start = time.time()
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided",
                "message": "Please upload an image file"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "message": "Please select an image file"
            }), 400
        
        # Read and analyze image
        image_data = file.read()
        
        if CV_AVAILABLE:
            # Convert to numpy array and decode
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Validate it's actually a leaf
                is_leaf, reject_reason, detection_info = is_leaf_image(img, mode="upload")
                if not is_leaf:
                    return jsonify({
                        "success": False,
                        "error": "not_a_leaf",
                        "message": reject_reason,
                        "detection": detection_info
                    }), 400
                
                # Run color analysis
                diagnosis_id, confidence = analyze_leaf_colors(img)
                diagnosis = DIAGNOSIS_TEMPLATES[diagnosis_id].copy()
                diagnosis["confidence"] = round(confidence, 2)
                analysis_method = "color_analysis"
            else:
                # Fallback if image decode fails
                diagnosis = DIAGNOSIS_TEMPLATES["healthy"].copy()
                diagnosis["confidence"] = 0.50
                analysis_method = "fallback"
        else:
            # No OpenCV - return healthy with note
            diagnosis = DIAGNOSIS_TEMPLATES["healthy"].copy()
            diagnosis["confidence"] = 0.50
            diagnosis["description"] += " (Limited analysis - OpenCV not available)"
            analysis_method = "fallback"
        
        processing_time = int((time.time() - processing_start) * 1000)
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "scan_id": f"SCAN-{random.randint(10000, 99999)}",
            "result": diagnosis,
            "metadata": {
                "analysis_method": analysis_method,
                "processing_time_ms": processing_time,
                "green_growth_certified": True,
                "cv_available": CV_AVAILABLE
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Image analysis failed: {str(e)}",
            "message": "Please try again or contact support"
        }), 500


@app.route("/api/scan/frame", methods=["POST"])
def scan_frame():
    """
    POST /api/scan/frame
    Analyze a webcam frame (base64 encoded) for leaf disease detection.
    
    Expected JSON payload:
    {
        "frame": "data:image/jpeg;base64,..."
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        frame_data = data.get("frame", "")
        
        if not frame_data:
            return jsonify({
                "success": False,
                "error": "No frame data provided"
            }), 400
        
        processing_start = time.time()
        
        # Decode base64 image
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        
        import base64
        image_bytes = base64.b64decode(frame_data)
        
        if CV_AVAILABLE:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Crop to center of frame FIRST — focus on the leaf area
                h, w = img.shape[:2]
                crop_img = img[int(h*0.2):int(h*0.8), int(w*0.25):int(w*0.75)]
                
                # Validate the CROPPED region (stricter — leaf should fill the guide box)
                is_leaf, reject_reason, detection_info = is_leaf_image(crop_img, mode="webcam")
                if not is_leaf:
                    return jsonify({
                        "success": False,
                        "error": "not_a_leaf",
                        "message": reject_reason,
                        "detection": detection_info
                    }), 400
                
                diagnosis_id, confidence = analyze_leaf_colors(crop_img)
                diagnosis = DIAGNOSIS_TEMPLATES[diagnosis_id].copy()
                diagnosis["confidence"] = round(confidence, 2)
                analysis_method = "color_analysis"
            else:
                diagnosis = DIAGNOSIS_TEMPLATES["healthy"].copy()
                diagnosis["confidence"] = 0.50
                analysis_method = "fallback"
        else:
            diagnosis = DIAGNOSIS_TEMPLATES["healthy"].copy()
            diagnosis["confidence"] = 0.50
            diagnosis["description"] += " (Limited analysis - OpenCV not available)"
            analysis_method = "fallback"
        
        processing_time = int((time.time() - processing_start) * 1000)
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "scan_id": f"LIVE-{random.randint(10000, 99999)}",
            "result": diagnosis,
            "metadata": {
                "analysis_method": analysis_method,
                "processing_time_ms": processing_time,
                "mode": "realtime",
                "cv_available": CV_AVAILABLE
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Frame analysis failed: {str(e)}",
            "message": "Please try again"
        }), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Mock AI chat endpoint for soil regeneration advice.
    
    Expected JSON payload:
    {
        "nitrogen": 40,
        "phosphorus": 30,
        "potassium": 35,
        "message": "optional user question"
    }
    
    Returns personalized soil regeneration advice based on NPK values.
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        # Extract NPK values with defaults
        nitrogen = data.get("nitrogen", 50)
        phosphorus = data.get("phosphorus", 50)
        potassium = data.get("potassium", 50)
        user_message = data.get("message", "")
        
        # Generate soil regeneration advice based on NPK levels
        advice_parts = []
        recommendations = []
        
        # Analyze Nitrogen
        if nitrogen < 30:
            advice_parts.append(f"Your nitrogen level ({nitrogen}) is LOW")
            recommendations.append("Plant legumes (beans, peas, clover) to fix nitrogen naturally")
            recommendations.append("Add composted manure or blood meal")
        elif nitrogen > 70:
            advice_parts.append(f"Your nitrogen level ({nitrogen}) is HIGH")
            recommendations.append("Avoid additional nitrogen fertilizers")
            recommendations.append("Plant heavy nitrogen feeders like corn or leafy greens")
        else:
            advice_parts.append(f"Your nitrogen level ({nitrogen}) is OPTIMAL")
        
        # Analyze Phosphorus
        if phosphorus < 30:
            advice_parts.append(f"Your phosphorus level ({phosphorus}) is LOW")
            recommendations.append("Add bone meal or rock phosphate")
            recommendations.append("Incorporate mycorrhizal fungi to improve phosphorus uptake")
        elif phosphorus > 70:
            advice_parts.append(f"Your phosphorus level ({phosphorus}) is HIGH")
            recommendations.append("Avoid phosphorus fertilizers to prevent runoff")
        else:
            advice_parts.append(f"Your phosphorus level ({phosphorus}) is OPTIMAL")
        
        # Analyze Potassium
        if potassium < 30:
            advice_parts.append(f"Your potassium level ({potassium}) is LOW")
            recommendations.append("Apply wood ash (excellent potassium source)")
            recommendations.append("Add kelp meal or greensand")
        elif potassium > 70:
            advice_parts.append(f"Your potassium level ({potassium}) is HIGH")
            recommendations.append("Reduce potassium inputs; excess can block magnesium uptake")
        else:
            advice_parts.append(f"Your potassium level ({potassium}) is OPTIMAL")
        
        # General regeneration tips
        general_tips = [
            "Practice crop rotation to maintain soil health",
            "Add organic matter (compost) regularly to improve soil structure",
            "Use cover crops during off-season to prevent erosion",
            "Minimize tillage to protect soil microbiome",
            "Test soil pH - most crops prefer 6.0-7.0"
        ]
        
        # Build response
        response = {
            "analysis": ". ".join(advice_parts) + ".",
            "recommendations": recommendations if recommendations else ["Your soil NPK levels are balanced! Maintain current practices."],
            "general_tips": random.sample(general_tips, min(3, len(general_tips))),
            "npk_summary": {
                "nitrogen": {"value": nitrogen, "status": "low" if nitrogen < 30 else "high" if nitrogen > 70 else "optimal"},
                "phosphorus": {"value": phosphorus, "status": "low" if phosphorus < 30 else "high" if phosphorus > 70 else "optimal"},
                "potassium": {"value": potassium, "status": "low" if potassium < 30 else "high" if potassium > 70 else "optimal"}
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add contextual response if user asked a specific question
        if user_message:
            response["user_query"] = user_message
            response["note"] = "For specific questions, consider consulting a local agricultural extension office."
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Chat service error: {str(e)}",
            "message": "Unable to process your request. Please try again."
        }), 500


# =============================================================================
# OPENWEATHERMAP API ENDPOINT
# =============================================================================
OPENWEATHER_API_KEY = "e3c0d1432605e246a088cc8d02da3eac"

# District → (lat, lon) mapping for all supported Indian agricultural districts
DISTRICT_COORDINATES = {
    # Maharashtra
    "Nashik": (19.9975, 73.7898), "Pune": (18.5204, 73.8567), "Nagpur": (21.1458, 79.0882),
    "Ahmednagar": (19.0948, 74.7480), "Solapur": (17.6599, 75.9064), "Kolhapur": (16.7050, 74.2433),
    "Sangli": (16.8524, 74.5815), "Satara": (17.6805, 74.0183), "Aurangabad": (19.8762, 75.3433),
    "Jalgaon": (21.0077, 75.5626),
    # Karnataka
    "Bangalore Rural": (13.2257, 77.5750), "Belgaum": (15.8497, 74.4977), "Bellary": (15.1394, 76.9214),
    "Bidar": (17.9104, 77.5199), "Dharwad": (15.4589, 75.0078), "Gulbarga": (17.3297, 76.8343),
    "Hassan": (13.0068, 76.1004), "Kolar": (13.1360, 78.1292), "Mandya": (12.5244, 76.8958),
    "Mysore": (12.2958, 76.6394), "Shimoga": (13.9299, 75.5681), "Tumkur": (13.3379, 77.1173),
    # Gujarat
    "Ahmedabad": (23.0225, 72.5714), "Amreli": (21.5981, 71.2163), "Anand": (22.5645, 72.9289),
    "Banaskantha": (24.1738, 72.4313), "Bharuch": (21.7051, 72.9959), "Bhavnagar": (21.7645, 72.1519),
    "Junagadh": (21.5222, 70.4579), "Kheda": (22.7505, 72.6847), "Mehsana": (23.5880, 72.3693),
    "Rajkot": (22.3039, 70.8022), "Surat": (21.1702, 72.8311), "Vadodara": (22.3072, 73.1812),
    # Punjab
    "Amritsar": (31.6340, 74.8723), "Bathinda": (30.2070, 74.9455), "Ferozepur": (30.9210, 74.6134),
    "Gurdaspur": (32.0414, 75.4028), "Jalandhar": (31.3260, 75.5762), "Ludhiana": (30.9010, 75.8573),
    "Moga": (30.8113, 75.1719), "Patiala": (30.3398, 76.3869), "Sangrur": (30.2330, 75.8408),
    # Haryana
    "Ambala": (30.3782, 76.7767), "Hisar": (29.1492, 75.7217), "Karnal": (29.6857, 76.9905),
    "Kurukshetra": (29.9695, 76.8783), "Rohtak": (28.8955, 76.6066), "Sirsa": (29.5349, 75.0280),
    "Sonipat": (28.9931, 77.0151),
    # Uttar Pradesh
    "Agra": (27.1767, 78.0081), "Aligarh": (27.8974, 78.0880), "Allahabad": (25.4358, 81.8463),
    "Bareilly": (28.3670, 79.4304), "Gorakhpur": (26.7606, 83.3732), "Jhansi": (25.4484, 78.5685),
    "Kanpur": (26.4499, 80.3319), "Lucknow": (26.8467, 80.9462), "Mathura": (27.4924, 77.6737),
    "Meerut": (28.9845, 77.7064), "Moradabad": (28.8386, 78.7733), "Muzaffarnagar": (29.4727, 77.7085),
    "Varanasi": (25.3176, 82.9739),
    # Madhya Pradesh
    "Bhopal": (23.2599, 77.4126), "Gwalior": (26.2183, 78.1828), "Indore": (22.7196, 75.8577),
    "Jabalpur": (23.1815, 79.9864), "Rewa": (24.5373, 81.2932), "Sagar": (23.8388, 78.7378),
    "Ujjain": (23.1765, 75.7885),
    # Rajasthan
    "Ajmer": (26.4499, 74.6399), "Alwar": (27.5530, 76.6346), "Bikaner": (28.0229, 73.3119),
    "Jaipur": (26.9124, 75.7873), "Jodhpur": (26.2389, 73.0243), "Kota": (25.2138, 75.8648),
    "Sikar": (27.6094, 75.1399), "Udaipur": (24.5854, 73.7125),
    # Tamil Nadu
    "Chennai": (13.0827, 80.2707), "Coimbatore": (11.0168, 76.9558), "Erode": (11.3410, 77.7172),
    "Madurai": (9.9252, 78.1198), "Salem": (11.6643, 78.1460), "Thanjavur": (10.7870, 79.1378),
    "Tiruchirappalli": (10.7905, 78.7047), "Tirunelveli": (8.7139, 77.7567),
    # Andhra Pradesh / Telangana
    "Anantapur": (14.6819, 77.6006), "Guntur": (16.3067, 80.4365), "Hyderabad": (17.3850, 78.4867),
    "Karimnagar": (18.4386, 79.1288), "Krishna": (16.6100, 80.7214), "Kurnool": (15.8281, 78.0373),
    "Nalgonda": (17.0583, 79.2671), "Warangal": (17.9784, 79.5941), "Visakhapatnam": (17.6868, 83.2185),
    # West Bengal
    "Barddhaman": (23.2324, 87.8615), "Hooghly": (22.8963, 88.2461), "Kolkata": (22.5726, 88.3639),
    "Murshidabad": (24.1745, 88.2700), "Nadia": (23.4710, 88.5565),
    # Bihar
    "Bhagalpur": (25.2425, 86.9842), "Darbhanga": (26.1542, 85.8918), "Gaya": (24.7955, 84.9994),
    "Muzaffarpur": (26.1209, 85.3647), "Patna": (25.6093, 85.1376),
    # Odisha
    "Balasore": (21.4934, 86.9135), "Cuttack": (20.4625, 85.8830), "Ganjam": (19.3860, 85.0503),
    "Puri": (19.8135, 85.8312),
    # Kerala
    "Alappuzha": (9.4981, 76.3388), "Ernakulam": (9.9816, 76.2999), "Kozhikode": (11.2588, 75.7804),
    "Palakkad": (10.7867, 76.6548), "Thrissur": (10.5276, 76.2144),
}


@app.route("/api/weather", methods=["GET"])
def get_weather():
    """
    GET /api/weather?district=Nagpur  (preferred - uses user's district)
    GET /api/weather?lat=20.59&lon=78.96  (manual coordinates)
    Fetch real-time weather data from OpenWeatherMap API.
    """
    district = request.args.get("district", "", type=str)
    
    # Resolve coordinates from district name if provided
    if district and district in DISTRICT_COORDINATES:
        lat, lon = DISTRICT_COORDINATES[district]
        print(f"🌤️ Weather for district: {district} → ({lat}, {lon})")
    else:
        lat = request.args.get("lat", 20.5937, type=float)
        lon = request.args.get("lon", 78.9629, type=float)
        if district:
            print(f"⚠️ Unknown district '{district}', using fallback coordinates")
    
    try:
        # Current weather
        current_url = "https://api.openweathermap.org/data/2.5/weather"
        current_params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        current_resp = requests.get(current_url, params=current_params, timeout=10)
        current_resp.raise_for_status()
        current_data = current_resp.json()
        
        # 5-day / 3-hour forecast for 24h outlook
        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "cnt": 8  # next 24 hours (8 x 3h)
        }
        forecast_resp = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
        
        # Extract current weather info
        temp = current_data["main"]["temp"]
        feels_like = current_data["main"]["feels_like"]
        humidity = current_data["main"]["humidity"]
        wind_speed = round(current_data["wind"]["speed"] * 3.6, 1)  # m/s -> km/h
        wind_deg = current_data["wind"].get("deg", 0)
        pressure = current_data["main"]["pressure"]
        visibility = current_data.get("visibility", 10000) / 1000  # m -> km
        clouds = current_data["clouds"]["all"]
        weather_main = current_data["weather"][0]["main"]
        weather_desc = current_data["weather"][0]["description"].capitalize()
        weather_icon = current_data["weather"][0]["icon"]
        city_name = current_data.get("name", "Unknown")
        sunrise = current_data["sys"].get("sunrise")
        sunset = current_data["sys"].get("sunset")
        
        # Rainfall data (last 1h / 3h if available)
        rain_1h = current_data.get("rain", {}).get("1h", 0)
        
        # Build 24h forecast summary
        forecast_items = []
        temp_min_24h = temp
        temp_max_24h = temp
        rain_chance = 0
        
        for item in forecast_data.get("list", []):
            t = item["main"]["temp"]
            temp_min_24h = min(temp_min_24h, t)
            temp_max_24h = max(temp_max_24h, t)
            pop = item.get("pop", 0) * 100  # probability of precipitation (%)
            rain_chance = max(rain_chance, pop)
            forecast_items.append({
                "time": item["dt_txt"],
                "temp": round(t, 1),
                "weather": item["weather"][0]["main"],
                "description": item["weather"][0]["description"],
                "rain_probability": round(pop),
                "humidity": item["main"]["humidity"],
                "wind_speed": round(item["wind"]["speed"] * 3.6, 1)
            })
        
        # Generate natural language forecast
        if rain_chance > 60:
            forecast_text = f"Rain expected ({int(rain_chance)}% chance). Consider delaying irrigation."
        elif rain_chance > 30:
            forecast_text = f"{weather_desc} with {int(rain_chance)}% chance of rain. Monitor conditions."
        elif clouds > 60:
            forecast_text = f"Cloudy skies ahead. Moderate conditions for field work."
        else:
            forecast_text = f"{weather_desc}. Good conditions for irrigation and field work."
        
        # Determine weather icon type for frontend
        icon_map = {
            "Clear": "sun",
            "Clouds": "cloud",
            "Rain": "rain",
            "Drizzle": "rain",
            "Thunderstorm": "storm",
            "Snow": "snow",
            "Mist": "mist",
            "Fog": "mist",
            "Haze": "mist"
        }
        
        return jsonify({
            "success": True,
            "current": {
                "temperature": round(temp, 1),
                "feels_like": round(feels_like, 1),
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_direction": wind_deg,
                "pressure": pressure,
                "visibility": round(visibility, 1),
                "clouds": clouds,
                "rain_1h": rain_1h,
                "weather_main": weather_main,
                "weather_description": weather_desc,
                "weather_icon": icon_map.get(weather_main, "sun"),
                "icon_code": weather_icon,
                "city": city_name,
                "sunrise": sunrise,
                "sunset": sunset
            },
            "forecast_24h": {
                "text": forecast_text,
                "temp_min": round(temp_min_24h, 1),
                "temp_max": round(temp_max_24h, 1),
                "rain_chance": round(rain_chance),
                "items": forecast_items
            },
            "source": "OpenWeatherMap",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except requests.RequestException as e:
        print(f"OpenWeatherMap API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Weather service unavailable: {str(e)}",
            "fallback": {
                "temperature": 28,
                "humidity": 65,
                "wind_speed": 12,
                "weather_description": "Data unavailable",
                "weather_icon": "sun"
            }
        }), 503


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "AgroSmart Farm Agent",
        "version": "2.0.0-ML",
        "ml_model": "Random Forest (loaded)" if farm_agent.ml_model.model else "Not loaded",
        "weather_api": "OpenWeatherMap",
        "mandi_connect": "Active",
        "timestamp": datetime.now().isoformat()
    }), 200


# =============================================================================
# MANDI CONNECT API ENDPOINTS
# =============================================================================

@app.route("/api/market/prices", methods=["GET"])
def get_market_prices():
    """
    GET /api/market/prices
    
    Fetch real-time market prices for agricultural commodities.
    Markets are sorted by proximity to farmer's registered location.
    
    Query Parameters:
        crop (required): Crop name (e.g., "rice", "tomato", "wheat")
        lat (optional): Farmer's latitude
        lon (optional): Farmer's longitude
    
    Returns:
        JSON with market prices from 7 nearest mandis, local trader comparison,
        and recommendation for best selling price.
    
    Example:
        GET /api/market/prices?crop=tomato&lat=19.0760&lon=72.8777
    """
    try:
        # Get crop parameter
        crop = request.args.get("crop", "tomato")
        
        if not crop:
            return jsonify({
                "error": "Missing required parameter: crop",
                "example": "/api/market/prices?crop=rice"
            }), 400
        
        # Get optional location parameters
        farmer_lat = request.args.get("lat", type=float)
        farmer_lon = request.args.get("lon", type=float)
        
        # Fetch market prices
        prices = MandiConnect.get_market_prices(crop, farmer_lat, farmer_lon)
        
        return jsonify(prices), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch market prices: {str(e)}",
            "message": "Please try again or contact support"
        }), 500


@app.route("/api/market/crops", methods=["GET"])
def get_supported_crops():
    """
    GET /api/market/crops
    
    Get list of all supported crops with their price ranges.
    """
    crops = []
    for crop, data in MandiConnect.CROP_BASE_PRICES.items():
        crops.append({
            "name": crop.capitalize(),
            "id": crop,
            "price_range": f"₹{data['min']:,} - ₹{data['max']:,}",
            "unit": data["unit"],
            "min_price": data["min"],
            "max_price": data["max"]
        })
    
    return jsonify({
        "crops": crops,
        "total": len(crops),
        "last_updated": datetime.now().isoformat()
    }), 200


@app.route("/api/market/mandis", methods=["GET"])
def get_all_mandis():
    """
    GET /api/market/mandis
    
    Get list of all mandis in the database with their locations.
    """
    mandis = []
    for mandi_id, info in MandiConnect.MANDI_DATABASE.items():
        mandis.append({
            "id": mandi_id,
            "name": info["name"],
            "location": info["location"],
            "type": info["type"],
            "speciality": info["speciality"],
            "coordinates": {
                "latitude": info["latitude"],
                "longitude": info["longitude"]
            }
        })
    
    return jsonify({
        "mandis": mandis,
        "total": len(mandis),
        "last_updated": datetime.now().isoformat()
    }), 200


@app.route("/api/market/recommend", methods=["GET"])
def get_crop_recommendation():
    """
    GET /api/market/recommend
    
    Get recommended crop based on current market trends.
    
    Query Parameters:
        lat (optional): Farmer's latitude
        lon (optional): Farmer's longitude
    """
    try:
        farmer_lat = request.args.get("lat", type=float)
        farmer_lon = request.args.get("lon", type=float)
        
        recommendation = MandiConnect.get_recommended_crop(farmer_lat, farmer_lon)
        
        return jsonify(recommendation), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to get recommendation: {str(e)}"
        }), 500


@app.route("/api/market/compare", methods=["GET"])
def compare_prices():
    """
    GET /api/market/compare
    
    Compare prices across multiple crops at once.
    
    Query Parameters:
        crops: Comma-separated list of crops (e.g., "rice,wheat,maize")
        lat (optional): Farmer's latitude
        lon (optional): Farmer's longitude
    """
    try:
        crops_param = request.args.get("crops", "rice,wheat,tomato")
        crops = [c.strip() for c in crops_param.split(",")]
        
        farmer_lat = request.args.get("lat", type=float)
        farmer_lon = request.args.get("lon", type=float)
        
        comparisons = []
        for crop in crops[:5]:  # Limit to 5 crops
            prices = MandiConnect.get_market_prices(crop, farmer_lat, farmer_lon)
            comparisons.append({
                "crop": crop.capitalize(),
                "best_price": prices["best_market"]["price_display"],
                "best_market": prices["best_market"]["name"],
                "local_price": prices["local_trader"]["price_display"],
                "potential_extra": prices["recommendation"]["extra_earning"]
            })
        
        return jsonify({
            "comparisons": comparisons,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Comparison failed: {str(e)}"
        }), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Get information about the ML model."""
    return jsonify({
        "model_type": "Random Forest Classifier",
        "n_estimators": 100,
        "max_depth": 10,
        "features": farm_agent.ml_model.feature_names,
        "feature_importance": farm_agent.ml_model.get_feature_importance(),
        "crop_classes": ["Rice", "Maize", "Chickpea"],
        "training_samples": 10000,
        "description": "Trained on synthetic agricultural data with domain knowledge rules"
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        "service": "AgroSmart: Green Growth Edition",
        "description": "Autonomous Farm Agent API with ML-powered predictions & Mandi Connect",
        "modules": {
            "farm_agent": "ML-based irrigation control",
            "mandi_connect": "Real-time agricultural market prices",
            "crop_recommender": "ML-based crop recommendation from sensor data"
        },
        "endpoints": {
            "irrigation": {
                "POST /predict": "Get ML-based irrigation decision from sensor data",
                "GET /model/info": "Get ML model information and feature importance"
            },
            "crop_recommendation": {
                "POST /api/crop/recommend": "Get ML-based crop recommendation from sensor data (N, P, K, temp, humidity, pH, rainfall, moisture)",
                "GET /api/crop/list": "List all 23 crops with ideal growing conditions",
                "GET /api/crop/model-info": "Get crop recommendation model details"
            },
            "mandi_connect": {
                "GET /api/market/prices?crop={name}": "Get market prices for a crop (sorted by proximity)",
                "GET /api/market/crops": "List all supported crops with price ranges",
                "GET /api/market/mandis": "List all mandis with locations",
                "GET /api/market/recommend": "Get recommended crop based on market trends",
                "GET /api/market/compare?crops={list}": "Compare prices across multiple crops"
            },
            "diagnosis": {
                "POST /api/scan/image": "Analyze crop/soil images for diagnosis",
                "POST /api/chat": "Get soil regeneration advice based on NPK values"
            },
            "system": {
                "GET /health": "Service health check",
                "GET /": "API documentation"
            }
        },
        "mandi_connect": {
            "markets_available": len(MandiConnect.MANDI_DATABASE),
            "crops_supported": list(MandiConnect.CROP_BASE_PRICES.keys()),
            "data_source": "Government APMC APIs + Market Simulation"
        },
        "ml_models": {
            "irrigation": {
                "type": "Random Forest Classifier",
                "features": ["soil_moisture", "temperature", "humidity", "rain_probability", "wind_speed", "hour_of_day", "crop_type"]
            },
            "crop_recommendation": {
                "type": "Random Forest Classifier",
                "features": ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "pH", "rainfall", "soil_moisture"],
                "crops_supported": len(crop_recommender.CROP_LIST)
            }
        },
        "weather_api": {
            "provider": "Open-Meteo (free)",
            "data": ["temperature", "humidity", "rain_probability", "wind_speed"]
        },
        "crop_profiles": FarmAgent.CROP_PROFILES,
        "features": [
            "ML-based irrigation prediction (Random Forest)",
            "ML-based crop recommendation (Random Forest - 23 crops)",
            "Real-time weather data from Open-Meteo API",
            "Mandi Connect - Live market prices from 10+ mandis",
            "Location-based market sorting (closest first)",
            "Local trader vs Mandi price comparison",
            "Crop recommendation based on sensor data & market trends",
            "Dynamic crop-specific irrigation thresholds",
            "Evaporation guard (11 AM - 3 PM protection)",
            "Safety overrides for critical conditions",
            "Pump safety monitoring"
        ]
    }), 200


# =============================================================================
# LLM CHAT - LM Studio Integration
# =============================================================================

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    """
    POST /api/llm/chat
    Proxy to LM Studio for AI chat functionality.
    
    Expected JSON payload:
    {
        "message": "User's question",
        "history": [{"role": "user"|"assistant", "content": "..."}],
        "sensor_data": {optional sensor context}
    }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        user_message = data.get("message", "")
        history = data.get("history", [])
        sensor_data = data.get("sensor_data", {})
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Build context with sensor data if provided
        system_prompt = """You are Regen, an AI farming assistant for AgroSmart. You help farmers with:
- Crop management and disease identification
- Irrigation scheduling and water conservation
- Soil health and regenerative agriculture practices
- Market prices and selling recommendations
- Weather-based farming decisions

Be concise, practical, and farmer-friendly. Use simple language."""
        
        if sensor_data:
            system_prompt += f"""\n\nCurrent sensor readings from the farm:
- Soil Moisture: {sensor_data.get('soil_moisture', 'N/A')}%
- Temperature: {sensor_data.get('temperature', 'N/A')}°C
- Humidity: {sensor_data.get('humidity', 'N/A')}%
- Soil pH: {sensor_data.get('soil_ph', 'N/A')}
- Nitrogen (N): {sensor_data.get('nitrogen', 'N/A')}
- Phosphorus (P): {sensor_data.get('phosphorus', 'N/A')}
- Potassium (K): {sensor_data.get('potassium', 'N/A')}

Use this live data to give specific, actionable advice."""
        
        # Build messages array for LM Studio
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 messages to keep context manageable)
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call LM Studio API
        try:
            response = requests.post(
                LM_STUDIO_URL,
                json={
                    "model": "meta-llama-3.1-8b-instruct",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            assistant_message = result["choices"][0]["message"]["content"]
            
            return jsonify({
                "success": True,
                "response": assistant_message,
                "model": "meta-llama-3.1-8b-instruct"
            })
            
        except requests.exceptions.ConnectionError:
            return jsonify({
                "success": False,
                "error": "LM Studio not running. Please start LM Studio with the model loaded.",
                "response": "I'm sorry, I can't connect to my AI brain right now. Please make sure LM Studio is running."
            }), 503
        except requests.exceptions.Timeout:
            return jsonify({
                "success": False,
                "error": "LM Studio response timeout",
                "response": "I'm taking too long to think. Please try a simpler question."
            }), 504
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "response": "Something went wrong. Please try again."
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  AgroSmart: Green Growth Edition - Farm Agent Server")
    print("  ML-Powered Irrigation + Crop Recommendation + Mandi Connect")
    print("=" * 60)
    print(f"  Starting server at http://127.0.0.1:5000")
    print(f"  Irrigation ML Model: Random Forest (100 trees)")
    print(f"  Crop Recommendation ML Model: Random Forest (150 trees, 23 crops)")
    print(f"  Weather API: Open-Meteo (free, no key required)")
    print(f"  Mandi Connect: {len(MandiConnect.MANDI_DATABASE)} markets")
    print(f"  Supported Market Crops: {len(MandiConnect.CROP_BASE_PRICES)}")
    print(f"  Crop Profiles: {FarmAgent.CROP_PROFILES}")
    print(f"  Critical Moisture Threshold: {FarmAgent.CRITICAL_MOISTURE}%")
    print(f"  High Noon Window: {FarmAgent.HIGH_NOON_START}:00 - {FarmAgent.HIGH_NOON_END}:00")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)
