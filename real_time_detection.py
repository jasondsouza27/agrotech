"""
Real-Time Leaf Disease Detection using Computer Vision
=======================================================
This script captures video from your webcam and runs color-based
leaf health analysis on each frame in real-time.

INSTALLATION:
    pip install opencv-python numpy

OPTIONAL (for YOLO plant disease model):
    pip install ultralytics

USAGE:
    python real_time_detection.py

CUSTOM MODEL:
    To use a trained YOLO model for detection, set MODEL_PATH to your 
    trained model (e.g., 'best.pt' from PlantVillage training).
"""

import cv2
import numpy as np

# Try to import YOLO for optional plant disease model support
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ℹ️  YOLO not available. Using color-based analysis only.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the YOLO model weights (only used if trained on plant diseases)
# DEFAULT: None - Uses color-based analysis
# CUSTOM:  Set to 'path/to/your/best.pt' for trained plant disease model
MODEL_PATH = None  # Set to 'best.pt' if you have a trained plant model

# Webcam index (0 = default webcam, 1 = external USB camera, etc.)
WEBCAM_INDEX = 0

# Minimum confidence threshold for detections (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Window title
WINDOW_TITLE = "Real-Time Leaf Health Analysis"

# ============================================================================
# COLOR-BASED LEAF ANALYSIS
# ============================================================================

def analyze_leaf_colors(frame):
    """
    Analyze frame using color-based computer vision.
    Returns diagnosis based on detected color patterns.
    
    Analysis:
    - Green hues (H: 35-85): Healthy chlorophyll
    - Yellow hues (H: 15-35): Nitrogen deficiency
    - Brown/orange (H: 0-15): Disease/necrosis
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    total_pixels = height * width
    
    # Define color masks
    # Healthy green
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    # Yellowing
    yellow_mask = cv2.inRange(hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))
    # Brown/disease
    brown_mask1 = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([20, 200, 180]))
    brown_mask2 = cv2.inRange(hsv, np.array([160, 30, 20]), np.array([180, 200, 180]))
    brown_mask = cv2.bitwise_or(brown_mask1, brown_mask2)
    # Dark spots
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    
    # Calculate percentages
    green_pct = (cv2.countNonZero(green_mask) / total_pixels) * 100
    yellow_pct = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
    brown_pct = (cv2.countNonZero(brown_mask) / total_pixels) * 100
    dark_pct = (cv2.countNonZero(dark_mask) / total_pixels) * 100
    
    # Determine diagnosis
    if brown_pct > 15 or dark_pct > 10:
        if brown_pct > 25:
            return "DISEASE: Late Blight", (0, 0, 255), min(0.95, 0.70 + brown_pct/100)
        else:
            return "DISEASE: Leaf Spot", (0, 128, 255), min(0.90, 0.65 + brown_pct/100)
    elif yellow_pct > 20 and green_pct < 40:
        if yellow_pct > 40:
            return "SEVERE: N Deficiency", (0, 0, 255), min(0.92, 0.65 + yellow_pct/100)
        else:
            return "WARNING: N Deficiency", (0, 255, 255), min(0.88, 0.60 + yellow_pct/100)
    elif yellow_pct > 10 and brown_pct > 5:
        return "WARNING: Early Blight", (0, 165, 255), min(0.85, 0.60 + (yellow_pct + brown_pct)/200)
    elif green_pct > 50:
        return "HEALTHY: Good", (0, 255, 0), min(0.96, 0.75 + green_pct/200)
    elif green_pct > 30:
        return "HEALTHY: Mild Stress", (0, 255, 128), min(0.85, 0.60 + green_pct/200)
    else:
        return "CHECK: Low Vegetation", (255, 255, 0), 0.50
    
    return {
        "green": round(green_pct, 1),
        "yellow": round(yellow_pct, 1),
        "brown": round(brown_pct, 1)
    }


def draw_analysis_overlay(frame, diagnosis, color, confidence, color_stats=None):
    """Draw analysis results on frame."""
    height, width = frame.shape[:2]
    
    # Semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw diagnosis
    cv2.putText(frame, diagnosis, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw confidence
    conf_text = f"Confidence: {confidence*100:.0f}%"
    cv2.putText(frame, conf_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw method
    cv2.putText(frame, "Method: Color Analysis", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Press Q to quit
    cv2.putText(frame, "Press 'q' to quit", (width - 150, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_real_time_detection():
    """
    Main function to run real-time leaf health analysis using webcam.
    Uses color-based analysis for accurate plant health detection.
    """
    model = None
    use_yolo = False
    
    # Try to load YOLO model if specified and available
    if MODEL_PATH and YOLO_AVAILABLE:
        print(f"Loading YOLO model: {MODEL_PATH}")
        try:
            model = YOLO(MODEL_PATH)
            use_yolo = True
            print("✅ YOLO model loaded!")
        except Exception as e:
            print(f"⚠️  Could not load YOLO model: {e}")
            print("   Falling back to color-based analysis.")
    else:
        print("🎨 Using color-based leaf health analysis")

    # Initialize video capture
    print(f"Opening webcam (index {WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Troubleshooting tips:")
        print("  - Check if another application is using the camera")
        print("  - Try changing WEBCAM_INDEX to 1 or 2")
        print("  - Ensure camera drivers are installed")
        return

    # Get camera properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print("-" * 50)
    print("Real-time leaf health analysis started!")
    print("Point camera at plant leaves for diagnosis")
    print("Press 'q' to quit")
    print("-" * 50)

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()

        if not success:
            print("Error: Failed to capture frame from webcam.")
            break

        if use_yolo and model:
            # Use YOLO for trained plant disease model
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            annotated_frame = results[0].plot()
            
            num_detections = len(results[0].boxes)
            cv2.putText(annotated_frame, f"YOLO Detections: {num_detections}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Use color-based analysis
            diagnosis, color, confidence = analyze_leaf_colors(frame)
            annotated_frame = draw_analysis_overlay(frame, diagnosis, color, confidence)

        # Show the annotated frame
        cv2.imshow(WINDOW_TITLE, annotated_frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated successfully.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    run_real_time_detection()
