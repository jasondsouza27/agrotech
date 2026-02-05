"""
AgroSmart: Green Growth Edition - Autonomous Farm Agent Server
A Flask-based prediction server with ML-powered irrigation control using Random Forest.
"""

from flask import Flask, request, jsonify
from datetime import datetime
import time
import random
import numpy as np
import requests
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


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


@app.route("/api/scan/image", methods=["POST"])
def scan_image():
    """
    POST /api/scan/image
    Mock endpoint for crop/soil image analysis.
    Simulates AI-powered diagnosis of plant health issues.
    
    Returns diagnosis after 1 second processing delay.
    """
    try:
        # Simulate AI processing time
        time.sleep(1)
        
        # Mock diagnoses pool
        diagnoses = [
            {
                "status": "Nitrogen Deficiency",
                "remedy": "Apply nitrogen-rich fertilizer or plant nitrogen-fixing crops like beans or legumes",
                "severity": "moderate",
                "confidence": 0.87
            },
            {
                "status": "Phosphorus Deficiency",
                "remedy": "Add bone meal or rock phosphate to soil. Consider mycorrhizal inoculants",
                "severity": "mild",
                "confidence": 0.92
            },
            {
                "status": "Potassium Deficiency",
                "remedy": "Apply wood ash or potassium sulfate. Mulch with banana peels",
                "severity": "moderate",
                "confidence": 0.85
            },
            {
                "status": "Healthy",
                "remedy": "No action needed. Continue current maintenance schedule",
                "severity": "none",
                "confidence": 0.95
            },
            {
                "status": "Fungal Infection Detected",
                "remedy": "Apply organic fungicide. Improve air circulation. Remove affected leaves",
                "severity": "high",
                "confidence": 0.89
            },
            {
                "status": "Pest Damage",
                "remedy": "Introduce beneficial insects. Apply neem oil spray. Check for aphids or caterpillars",
                "severity": "moderate",
                "confidence": 0.83
            },
            {
                "status": "Water Stress",
                "remedy": "Adjust irrigation schedule. Check soil drainage. Consider mulching to retain moisture",
                "severity": "moderate",
                "confidence": 0.91
            }
        ]
        
        # Select a random diagnosis (in production, this would be ML model output)
        diagnosis = random.choice(diagnoses)
        diagnosis["timestamp"] = datetime.now().isoformat()
        diagnosis["processing_time_ms"] = 1000
        
        return jsonify(diagnosis), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Image analysis failed: {str(e)}",
            "status": "Error",
            "remedy": "Please try again or contact support"
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
            recommendations.append("Add bone meal or rock phosphite")
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


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "AgroSmart Farm Agent",
        "version": "2.0.0-ML",
        "ml_model": "Random Forest (loaded)" if farm_agent.ml_model.model else "Not loaded",
        "weather_api": "Open-Meteo",
        "timestamp": datetime.now().isoformat()
    }), 200


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
        "description": "Autonomous Farm Agent API with ML-powered predictions",
        "endpoints": {
            "POST /predict": "Get ML-based irrigation decision from sensor data",
            "POST /api/scan/image": "Analyze crop/soil images for diagnosis",
            "POST /api/chat": "Get soil regeneration advice based on NPK values",
            "GET /model/info": "Get ML model information and feature importance",
            "GET /health": "Service health check"
        },
        "ml_model": {
            "type": "Random Forest Classifier",
            "features": ["soil_moisture", "temperature", "humidity", "rain_probability", "wind_speed", "hour_of_day", "crop_type"]
        },
        "weather_api": {
            "provider": "Open-Meteo (free)",
            "data": ["temperature", "humidity", "rain_probability", "wind_speed"]
        },
        "crop_profiles": FarmAgent.CROP_PROFILES,
        "features": [
            "ML-based irrigation prediction (Random Forest)",
            "Real-time weather data from Open-Meteo API",
            "Dynamic crop-specific irrigation thresholds",
            "Evaporation guard (11 AM - 3 PM protection)",
            "Safety overrides for critical conditions",
            "Pump safety monitoring"
        ]
    }), 200


if __name__ == "__main__":
    print("=" * 60)
    print("  AgroSmart: Green Growth Edition - Farm Agent Server")
    print("  ML-Powered Irrigation with Real Weather Data")
    print("=" * 60)
    print(f"  Starting server at http://127.0.0.1:5000")
    print(f"  ML Model: Random Forest Classifier (100 trees)")
    print(f"  Weather API: Open-Meteo (free, no key required)")
    print(f"  Crop Profiles: {FarmAgent.CROP_PROFILES}")
    print(f"  Critical Moisture Threshold: {FarmAgent.CRITICAL_MOISTURE}%")
    print(f"  High Noon Window: {FarmAgent.HIGH_NOON_START}:00 - {FarmAgent.HIGH_NOON_END}:00")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)
