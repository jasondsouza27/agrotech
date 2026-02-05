"""
Leaf Doctor - Crop Disease Detection API
AgroSmart Backend Server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    # Return empty response if no favicon file exists
    return '', 204

# Mock AI diagnosis results - All remedies are organic/sustainable (Green Growth)
DIAGNOSIS_RESULTS = [
    {
        "id": "healthy",
        "status": "healthy",
        "confidence": 0.94,
        "diagnosis": "Healthy Crop",
        "description": "Your crop is healthy and shows no signs of disease or nutrient deficiency.",
        "remedy": "Keep soil moisture at 60%. Continue current organic practices.",
        "severity": "none",
        "icon": "✅",
        "color": "#22c55e"
    },
    {
        "id": "nitrogen_deficiency",
        "status": "deficiency",
        "confidence": 0.87,
        "diagnosis": "Nitrogen Deficiency",
        "description": "Yellowing detected in older leaves, indicating low nitrogen levels in the soil.",
        "remedy": "Plant beans or legumes nearby to fix nitrogen naturally. Apply compost tea or well-rotted manure as organic nitrogen sources.",
        "severity": "moderate",
        "icon": "⚠️",
        "color": "#eab308"
    },
    {
        "id": "early_blight",
        "status": "disease",
        "confidence": 0.91,
        "diagnosis": "Early Blight (Alternaria solani)",
        "description": "Fungal spots detected with characteristic concentric rings on leaves.",
        "remedy": "Apply Neem oil spray (2-3 tbsp per gallon of water). Improve air circulation by proper spacing. Remove affected leaves and practice crop rotation.",
        "severity": "high",
        "icon": "🔴",
        "color": "#ef4444"
    }
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    
    # Simulate AI processing time (2 seconds)
    time.sleep(2)
    
    # Mock AI: Randomly select a diagnosis result
    diagnosis = random.choice(DIAGNOSIS_RESULTS)
    
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
            "model_version": "leaf-doctor-v1.0-mock",
            "processing_time_ms": 2000,
            "green_growth_certified": True
        }
    }
    
    return jsonify(response), 200


@app.route('/api/scan/diagnoses', methods=['GET'])
def get_all_diagnoses():
    """Return all possible diagnoses (for reference/testing)."""
    return jsonify({
        "success": True,
        "diagnoses": DIAGNOSIS_RESULTS,
        "note": "All remedies are organic and sustainable (Green Growth certified)"
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
    ║   • GET  /api/health        - Health check                ║
    ║   • POST /api/scan/image    - Scan crop image             ║
    ║   • GET  /api/scan/diagnoses - List all diagnoses         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)
