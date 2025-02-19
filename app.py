import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture('video.mp4')

# Set a fixed width and height for resizing (adjust as needed)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frames Ended")
        break

    
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    results = model(frame)
    # Render results on the frame (this will include bounding boxes and labels)
    frame_with_results = results[0].plot()  # Get the result from the first frame and plot the bounding boxes

    # Display the frame with predictions
    cv2.imshow("Real-time Object Detection", frame_with_results)

    # Exit on pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
