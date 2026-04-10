import cv2
import numpy as np

# Use /dev/video0 based on your v4l2-ctl output
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set the format to Y16
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

if not cap.isOpened():
    print("Cannot open Boson. Check if another process is using /dev/video0")
    exit()

print("Feed started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # The Jetson/OpenCV might return the 16-bit data as a 1D array
    # because of the 'Convert_RGB=0' flag. If so, reshape it.
    if len(frame.shape) == 1 or frame.shape[1] != 320:
        frame = frame.reshape((256, 320))

    # --- RADIOMETRIC PROCESSING ---
    # Convert to 8-bit for display using Min-Max scaling
    # This ensures the scene contrast is always visible
    norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply a high-contrast thermal colormap (MAGMA or INFERNO)
    color_mapped = cv2.applyColorMap(norm_frame, cv2.COLORMAP_MAGMA)

    # Display the feed
    big_frame = cv2.resize(color_mapped, (1080, 1020), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('FLIR Boson 320 - Big View', big_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
