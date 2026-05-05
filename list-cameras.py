# list_cameras.py
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"Camera index {i}: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Camera index {i}: opens but no frame")
    cap.release()
