"""
Real-Time Leaf Disease Detection using YOLOv8
==============================================
This script captures video from your webcam and runs YOLO object detection
on each frame in real-time.

INSTALLATION:
    pip install opencv-python ultralytics

USAGE:
    python real_time_detection.py

CUSTOM MODEL:
    To detect specific leaf diseases, you need a YOLO model trained on a
    plant disease dataset (e.g., PlantVillage). Once you have trained your
    model and exported it as 'best.pt', change the MODEL_PATH below.
"""

import cv2
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the YOLO model weights
# DEFAULT: 'yolov8n.pt' - A lightweight general-purpose model (auto-downloads)
# CUSTOM:  Replace with 'path/to/your/best.pt' for leaf disease detection
MODEL_PATH = 'yolov8n.pt'

# Webcam index (0 = default webcam, 1 = external USB camera, etc.)
WEBCAM_INDEX = 0

# Minimum confidence threshold for detections (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Window title
WINDOW_TITLE = "Real-Time Leaf Disease Detection"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_real_time_detection():
    """
    Main function to run real-time object/disease detection using webcam.
    """
    # Load the YOLO model
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists or check your internet connection.")
        return

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
    print("Real-time detection started!")
    print("Press 'q' to quit")
    print("-" * 50)

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()

        if not success:
            print("Error: Failed to capture frame from webcam.")
            break

        # Run YOLO inference on the frame
        # verbose=False suppresses per-frame logging
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        # Visualize the results on the frame
        # This draws bounding boxes, labels, and confidence scores
        annotated_frame = results[0].plot()

        # Display detection info on screen
        num_detections = len(results[0].boxes)
        cv2.putText(
            annotated_frame,
            f"Detections: {num_detections}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

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
