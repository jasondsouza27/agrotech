"""
Leaf Doctor - Crop Disease Detection API
AgroSmart Backend Server

This server provides:
- Image-based disease detection via file upload (/api/scan/image)
- Real-time detection status for webcam feed (/api/realtime/status)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
import os
from datetime import datetime

# Computer Vision Integration
try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("⚠️  OpenCV not installed. Using mock predictions.")
    print("   Install with: pip install opencv-python")

# YOLO Integration (optional - for custom trained models)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ℹ️  YOLO not installed. Using color-based analysis.")
    print("   Install with: pip install ultralytics")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ============================================================================
# YOLO MODEL CONFIGURATION
# ============================================================================
# Change this to your trained leaf disease model path (e.g., 'best.pt')
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt')
model = None

def load_model():
    """Load the YOLO model (lazy loading)."""
    global model
    if model is None and YOLO_AVAILABLE:
        try:
            print(f"Loading YOLO model: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    return model


def analyze_leaf_colors(img: np.ndarray) -> dict:
    """
    Analyze leaf image using color-based computer vision.
    Returns diagnosis based on detected color patterns.
    
    Analysis approach:
    - Green hues (H: 35-85): Healthy chlorophyll
    - Yellow hues (H: 15-35): Nitrogen deficiency / chlorosis  
    - Brown/orange (H: 0-15, 165-180): Disease / necrosis
    - Dark spots: Fungal disease indicators
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Convert to LAB for better brown detection
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Get image dimensions
    height, width = img.shape[:2]
    total_pixels = height * width
    
    # Define color masks in HSV
    # Healthy green (H: 35-85, S: 40-255, V: 40-255)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Yellowing / chlorosis (H: 15-35)
    yellow_lower = np.array([15, 40, 40])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Brown / necrosis (low saturation browns and oranges)
    brown_lower1 = np.array([0, 30, 20])
    brown_upper1 = np.array([20, 200, 180])
    brown_mask1 = cv2.inRange(hsv, brown_lower1, brown_upper1)
    
    brown_lower2 = np.array([160, 30, 20])
    brown_upper2 = np.array([180, 200, 180])
    brown_mask2 = cv2.inRange(hsv, brown_lower2, brown_upper2)
    brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
    
    # Dark spots (potential fungal infection) - very low value
    dark_lower = np.array([0, 0, 0])
    dark_upper = np.array([180, 255, 50])
    dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
    
    # Calculate percentages
    green_percent = (cv2.countNonZero(green_mask) / total_pixels) * 100
    yellow_percent = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
    brown_percent = (cv2.countNonZero(brown_mask) / total_pixels) * 100
    dark_percent = (cv2.countNonZero(dark_mask) / total_pixels) * 100
    
    # Analyze texture for spots (using edge detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = (cv2.countNonZero(edges) / total_pixels) * 100
    
    # Calculate color variance (diseased leaves often have irregular patterns)
    hsv_std = np.std(hsv[:,:,0])  # Hue variance
    
    # Decision logic based on color analysis
    analysis = {
        "green_percent": round(green_percent, 1),
        "yellow_percent": round(yellow_percent, 1),
        "brown_percent": round(brown_percent, 1),
        "dark_percent": round(dark_percent, 1),
        "edge_density": round(edge_density, 1),
        "color_variance": round(hsv_std, 1)
    }
    
    # Determine diagnosis based on color ratios
    # Check for disease indicators first (brown/dark spots)
    if brown_percent > 12 or (dark_percent > 8 and edge_density > 6):
        # Significant brown/dark spots = Disease
        confidence = min(0.95, 0.70 + (brown_percent / 100) + (dark_percent / 100))
        if brown_percent > 20:
            diagnosis_id = "late_blight"
        elif edge_density > 10:
            diagnosis_id = "early_blight"  # Spots with rings
        else:
            diagnosis_id = "leaf_spot"
            
    elif yellow_percent > 18 and green_percent < 45:
        # Significant yellowing = Deficiency
        confidence = min(0.92, 0.65 + (yellow_percent / 100))
        if yellow_percent > 35:
            diagnosis_id = "severe_nitrogen_deficiency"
        else:
            diagnosis_id = "nitrogen_deficiency"
            
    elif yellow_percent > 8 and brown_percent > 4:
        # Mixed yellowing and browning = Early disease stage
        confidence = min(0.88, 0.60 + (yellow_percent / 200) + (brown_percent / 100))
        diagnosis_id = "early_blight"
        
    elif green_percent > 45 and yellow_percent < 18 and brown_percent < 12:
        # Predominantly green = Healthy
        confidence = min(0.96, 0.75 + (green_percent / 200))
        diagnosis_id = "healthy"
        
    elif green_percent > 25 and yellow_percent < 25:
        # Mostly green with some yellowing = Mild stress
        confidence = min(0.85, 0.60 + (green_percent / 200))
        diagnosis_id = "mild_stress"
        
    else:
        # Mixed signals - provide moderate confidence healthy or stress
        if green_percent > yellow_percent:
            diagnosis_id = "healthy"
            confidence = 0.65
        else:
            diagnosis_id = "nutrient_stress"
            confidence = 0.60
    
    analysis["diagnosis_id"] = diagnosis_id
    analysis["confidence"] = round(confidence, 2)
    
    return analysis

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    # Return empty response if no favicon file exists
    return '', 204

# Mock AI diagnosis results - All remedies are organic/sustainable (Green Growth)
DIAGNOSIS_RESULTS = {
    "healthy": {
        "id": "healthy",
        "status": "healthy",
        "confidence": 0.94,
        "diagnosis": "Healthy Crop",
        "description": "Your crop is healthy and shows strong chlorophyll presence with no signs of disease or nutrient deficiency.",
        "remedy": "Keep soil moisture at 60%. Continue current organic practices. Monitor weekly for any changes.",
        "severity": "none",
        "icon": "✅",
        "color": "#22c55e"
    },
    "mild_stress": {
        "id": "mild_stress",
        "status": "healthy",
        "confidence": 0.82,
        "diagnosis": "Mild Environmental Stress",
        "description": "Leaf shows minor stress indicators but is generally healthy. May be due to temperature or water fluctuations.",
        "remedy": "Ensure consistent watering schedule. Provide shade during peak heat. Apply compost mulch to regulate soil temperature.",
        "severity": "low",
        "icon": "✅",
        "color": "#22c55e"
    },
    "nitrogen_deficiency": {
        "id": "nitrogen_deficiency",
        "status": "deficiency",
        "confidence": 0.87,
        "diagnosis": "Nitrogen Deficiency",
        "description": "Yellowing detected in leaves, indicating low nitrogen levels in the soil. Older leaves affected first.",
        "remedy": "Plant beans or legumes nearby to fix nitrogen naturally. Apply compost tea or well-rotted manure as organic nitrogen sources. Consider adding blood meal or fish emulsion.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#eab308"
    },
    "severe_nitrogen_deficiency": {
        "id": "severe_nitrogen_deficiency",
        "status": "deficiency",
        "confidence": 0.91,
        "diagnosis": "Severe Nitrogen Deficiency",
        "description": "Significant yellowing and chlorosis detected. Leaves show advanced nitrogen starvation symptoms.",
        "remedy": "Immediate application of organic nitrogen: fish emulsion (1 tbsp/gallon) or blood meal. Add compost heavily. Consider cover crop rotation with legumes next season.",
        "severity": "high",
        "icon": "🔴",
        "color": "#ef4444"
    },
    "nutrient_stress": {
        "id": "nutrient_stress",
        "status": "deficiency",
        "confidence": 0.75,
        "diagnosis": "General Nutrient Stress",
        "description": "Leaf coloration suggests possible nutrient imbalance. Could be multiple micronutrient deficiencies.",
        "remedy": "Apply balanced organic fertilizer. Test soil pH (ideal: 6.0-7.0). Add compost and consider foliar feeding with seaweed extract.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#eab308"
    },
    "early_blight": {
        "id": "early_blight",
        "status": "disease",
        "confidence": 0.91,
        "diagnosis": "Early Blight (Alternaria solani)",
        "description": "Fungal spots detected with characteristic patterns on leaves. Common in tomatoes and potatoes.",
        "remedy": "Apply Neem oil spray (2-3 tbsp per gallon of water). Improve air circulation by proper spacing. Remove affected leaves immediately and practice crop rotation. Apply copper-based organic fungicide if severe.",
        "severity": "high",
        "icon": "🔴",
        "color": "#ef4444"
    },
    "late_blight": {
        "id": "late_blight",
        "status": "disease",
        "confidence": 0.89,
        "diagnosis": "Late Blight (Phytophthora infestans)",
        "description": "Severe fungal infection detected. Brown lesions with possible white fungal growth. Spreads rapidly in humid conditions.",
        "remedy": "Remove and destroy all affected plant material immediately. Apply copper hydroxide organic fungicide. Improve drainage and air circulation. Do not compost infected material. Practice 3-year crop rotation.",
        "severity": "critical",
        "icon": "🔴",
        "color": "#dc2626"
    },
    "leaf_spot": {
        "id": "leaf_spot",
        "status": "disease",
        "confidence": 0.85,
        "diagnosis": "Bacterial/Fungal Leaf Spot",
        "description": "Dark spots detected on leaf surface indicating bacterial or fungal infection. Multiple pathogens can cause similar symptoms.",
        "remedy": "Apply baking soda solution (1 tbsp + 1/2 tsp liquid soap per gallon). Remove affected leaves. Avoid overhead watering. Apply neem oil preventively. Ensure good air circulation.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#f97316"
    },
    "powdery_mildew": {
        "id": "powdery_mildew",
        "status": "disease",
        "confidence": 0.88,
        "diagnosis": "Powdery Mildew",
        "description": "White powdery coating detected on leaf surface. Common fungal disease in humid conditions.",
        "remedy": "Spray milk solution (40% milk, 60% water). Apply neem oil. Improve air circulation. Remove severely affected leaves. Apply sulfur-based organic fungicide for persistent cases.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#eab308"
    },
    "rust": {
        "id": "rust",
        "status": "disease",
        "confidence": 0.86,
        "diagnosis": "Rust Fungus",
        "description": "Orange/brown rust pustules detected on leaves. Fungal spores spread by wind and water splash.",
        "remedy": "Remove and destroy infected leaves. Apply sulfur-based organic fungicide. Avoid wetting foliage when watering. Increase plant spacing for better airflow. Consider resistant varieties.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#f97316"
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def map_yolo_to_diagnosis(class_name: str, confidence: float) -> dict:
    """
    Map YOLO detection class names to diagnosis results.
    Customize this mapping based on your trained model's classes.
    """
    # Mapping for common plant disease datasets (e.g., PlantVillage)
    disease_mapping = {
        # Healthy classes
        "healthy": "healthy",
        "Healthy": "healthy",
        # Deficiency classes
        "nitrogen_deficiency": "nitrogen_deficiency",
        "nutrient_deficiency": "nutrient_stress",
        # Disease classes
        "early_blight": "early_blight",
        "late_blight": "late_blight",
        "leaf_spot": "leaf_spot",
        "rust": "rust",
        "powdery_mildew": "powdery_mildew",
    }
    
    # Check if class name matches any known diagnosis
    for key, diagnosis_id in disease_mapping.items():
        if key.lower() in class_name.lower():
            # Get diagnosis and update confidence from actual detection
            result = DIAGNOSIS_RESULTS[diagnosis_id].copy()
            result["confidence"] = round(confidence, 2)
            return result
    
    # Default: If using standard YOLO model (not trained on plants),
    # return info message
    return {
        "id": "unknown_detection",
        "status": "info",
        "confidence": round(confidence, 2),
        "diagnosis": f"Detected: {class_name}",
        "description": f"Standard object detected: {class_name}. For accurate plant disease detection, use a model trained on agricultural datasets.",
        "remedy": "Upload a clear image of the leaf for better analysis, or train a custom YOLO model on plant disease data.",
        "severity": "none",
        "icon": "ℹ️",
        "color": "#3b82f6"
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "Leaf Doctor API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/api/scan/image', methods=['POST'])
def scan_image():
    """
    Scan a crop image for disease detection.
    
    Accepts: multipart/form-data with 'image' file
    Returns: JSON diagnosis result
    """
    # Check if image file is present in request
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided",
            "message": "Please upload an image file with the key 'image'"
        }), 400
    
    file = request.files['image']
    
    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected",
            "message": "Please select an image file to upload"
        }), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type",
            "message": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Read image data
    image_data = file.read()
    processing_start = time.time()
    
    diagnosis = None
    analysis_method = "unknown"
    color_analysis = None
    
    # Priority 1: Use color-based analysis (works for actual leaf images)
    if CV_AVAILABLE:
        try:
            # Convert image bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Run color-based leaf analysis
                color_analysis = analyze_leaf_colors(img)
                diagnosis_id = color_analysis["diagnosis_id"]
                
                # Get the diagnosis template and update with analysis confidence
                diagnosis = DIAGNOSIS_RESULTS[diagnosis_id].copy()
                diagnosis["confidence"] = color_analysis["confidence"]
                analysis_method = "color_analysis"
                
                print(f"📊 Color Analysis: green={color_analysis['green_percent']}%, "
                      f"yellow={color_analysis['yellow_percent']}%, "
                      f"brown={color_analysis['brown_percent']}% -> {diagnosis_id}")
        except Exception as e:
            print(f"Color analysis error: {e}")
            diagnosis = None
    
    # Priority 2: Try YOLO if color analysis failed and model is trained on plants
    if diagnosis is None and YOLO_AVAILABLE:
        yolo_model = load_model()
        # Only use YOLO if it's a custom trained plant disease model
        is_plant_model = "best.pt" in MODEL_PATH or "plant" in MODEL_PATH.lower() or "disease" in MODEL_PATH.lower()
        
        if yolo_model and is_plant_model:
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                results = yolo_model(img, verbose=False, conf=0.5)
                
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_idx = boxes.conf.argmax().item()
                    confidence = float(boxes.conf[best_idx])
                    class_id = int(boxes.cls[best_idx])
                    class_name = results[0].names[class_id]
                    
                    diagnosis = map_yolo_to_diagnosis(class_name, confidence)
                    analysis_method = "yolo_plant_model"
            except Exception as e:
                print(f"YOLO prediction error: {e}")
    
    # Priority 3: Fallback to healthy with low confidence if all else fails
    if diagnosis is None:
        diagnosis = DIAGNOSIS_RESULTS["healthy"].copy()
        diagnosis["confidence"] = 0.50
        diagnosis["description"] += " (Limited analysis - OpenCV required for accurate detection)"
        analysis_method = "fallback"
    
    processing_time = int((time.time() - processing_start) * 1000)
    
    # Build response
    response = {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "filename": file.filename,
        "scan_id": f"SCAN-{random.randint(10000, 99999)}",
        "result": {
            "id": diagnosis["id"],
            "status": diagnosis["status"],
            "confidence": diagnosis["confidence"],
            "diagnosis": diagnosis["diagnosis"],
            "description": diagnosis["description"],
            "remedy": diagnosis["remedy"],
            "severity": diagnosis["severity"],
            "icon": diagnosis["icon"],
            "color": diagnosis["color"]
        },
        "metadata": {
            "analysis_method": analysis_method,
            "processing_time_ms": processing_time,
            "green_growth_certified": True,
            "cv_available": CV_AVAILABLE,
            "yolo_available": YOLO_AVAILABLE
        }
    }
    
    # Include color analysis breakdown for transparency
    if color_analysis:
        response["metadata"]["color_analysis"] = {
            "green_percent": color_analysis["green_percent"],
            "yellow_percent": color_analysis["yellow_percent"],
            "brown_percent": color_analysis["brown_percent"],
            "dark_spots_percent": color_analysis["dark_percent"]
        }
    
    return jsonify(response), 200


@app.route('/api/scan/frame', methods=['POST'])
def scan_frame():
    """
    Scan a frame from webcam for disease detection.
    
    Accepts: JSON with base64 encoded image in 'frame' field
    Returns: JSON diagnosis result
    """
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({
                "success": False,
                "error": "No frame data provided",
                "message": "Please provide base64 encoded image in 'frame' field"
            }), 400
        
        # Decode base64 image
        import base64
        import re
        
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        frame_data = data['frame']
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]
        
        image_data = base64.b64decode(frame_data)
        processing_start = time.time()
        
        diagnosis = None
        analysis_method = "unknown"
        color_analysis = None
        
        # Priority 1: Use color-based analysis
        if CV_AVAILABLE:
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    color_analysis = analyze_leaf_colors(img)
                    diagnosis_id = color_analysis["diagnosis_id"]
                    diagnosis = DIAGNOSIS_RESULTS[diagnosis_id].copy()
                    diagnosis["confidence"] = color_analysis["confidence"]
                    analysis_method = "color_analysis"
            except Exception as e:
                print(f"Color analysis error: {e}")
                diagnosis = None
        
        # Fallback if analysis failed
        if diagnosis is None:
            diagnosis = DIAGNOSIS_RESULTS["healthy"].copy()
            diagnosis["confidence"] = 0.50
            diagnosis["description"] += " (Limited analysis)"
            analysis_method = "fallback"
        
        processing_time = int((time.time() - processing_start) * 1000)
        
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "scan_id": f"FRAME-{random.randint(10000, 99999)}",
            "result": {
                "id": diagnosis["id"],
                "status": diagnosis["status"],
                "confidence": diagnosis["confidence"],
                "diagnosis": diagnosis["diagnosis"],
                "description": diagnosis["description"],
                "remedy": diagnosis["remedy"],
                "severity": diagnosis["severity"]
            },
            "metadata": {
                "analysis_method": analysis_method,
                "processing_time_ms": processing_time,
                "source": "webcam_frame"
            }
        }
        
        if color_analysis:
            response["metadata"]["color_analysis"] = {
                "green_percent": color_analysis["green_percent"],
                "yellow_percent": color_analysis["yellow_percent"],
                "brown_percent": color_analysis["brown_percent"]
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to process frame"
        }), 500


@app.route('/api/scan/diagnoses', methods=['GET'])
def get_all_diagnoses():
    """Return all possible diagnoses (for reference/testing)."""
    return jsonify({
        "success": True,
        "diagnoses": list(DIAGNOSIS_RESULTS.values()),
        "note": "All remedies are organic and sustainable (Green Growth certified)"
    })


@app.route('/api/esp32/sensors', methods=['GET'])
def get_esp32_sensors():
    """
    Get ESP32 sensor data.
    Returns simulated sensor data for demo purposes.
    Connect real ESP32 devices for live readings.
    """
    # Generate simulated sensor data with slight variations
    import random
    base_time = time.time()
    
    simulated_device = {
        "device_id": "ESP32_DEMO_001",
        "soil_moisture": round(45 + random.uniform(-5, 10), 1),
        "temperature": round(26 + random.uniform(-2, 4), 1),
        "humidity": round(62 + random.uniform(-5, 8), 1),
        "pump_running": random.choice([True, False]),
        "pump_runtime": random.randint(0, 120),
        "auto_mode": True,
        "wifi_rssi": random.randint(-75, -45),
        "uptime": int(base_time % 86400),
        "last_update": datetime.utcnow().isoformat(),
        "is_simulated": True,
        "is_online": True
    }
    
    return jsonify({
        "success": True,
        "devices": [simulated_device],
        "count": 1,
        "note": "Simulated data. Connect ESP32 hardware for real sensor readings."
    })


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Return system capabilities and model info."""
    return jsonify({
        "success": True,
        "cv_available": CV_AVAILABLE,
        "yolo_available": YOLO_AVAILABLE,
        "model_path": MODEL_PATH if YOLO_AVAILABLE else None,
        "model_loaded": model is not None,
        "capabilities": {
            "image_scan": True,
            "color_analysis": CV_AVAILABLE,
            "realtime_detection": CV_AVAILABLE,
            "yolo_plant_model": YOLO_AVAILABLE
        },
        "analysis_method": "color_analysis" if CV_AVAILABLE else "fallback",
        "instructions": {
            "for_best_results": "Upload clear leaf images against contrasting background",
            "custom_model": "Set YOLO_MODEL_PATH environment variable to your trained plant disease model"
        }
    })


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🌿 LEAF DOCTOR - Crop Disease Detection API 🌿          ║
    ║                                                           ║
    ║   AgroSmart Backend Server                                ║
    ║   Green Growth Certified Remedies                         ║
    ║                                                           ║
    ║   Endpoints:                                              ║
    ║   • GET  /api/health         - Health check               ║
    ║   • POST /api/scan/image     - Scan crop image            ║
    ║   • POST /api/scan/frame     - Scan webcam frame          ║
    ║   • GET  /api/scan/diagnoses - List all diagnoses         ║
    ║   • GET  /api/esp32/sensors  - ESP32 sensor data          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)
