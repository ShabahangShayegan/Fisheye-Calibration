import cv2
import time
import platform
from pathlib import Path
from datetime import datetime

# --- Configuration ---
DEVICE_ID = 2
WIDTH = 640
HEIGHT = 480
FPS = 30

OUTPUT_DIR = Path("test_videos")
RECORD_SECONDS = None  # set to None to record until q/ESC

def open_camera():
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(DEVICE_ID, backend)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera with DEVICE_ID={DEVICE_ID}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  Reported FPS: {actual_fps}")

    return cap, actual_width, actual_height


def record_video():
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"fisheye_test_{timestamp}.mp4"

    cap, width, height = open_camera()

    # MP4 codec. If this fails on your machine, switch to XVID + .avi below.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        FPS,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter. Try changing codec/container.")

    print("\nRecording started.")
    print(f"Saving to: {output_path}")
    print("Press q or ESC to stop.\n")

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        elapsed = time.time() - start_time

        # Write clean raw camera frame to file
        writer.write(frame)
        frame_count += 1

        # Display overlay only in preview, not in saved video
        preview = frame.copy()
        cv2.putText(
            preview,
            f"REC {elapsed:.1f}s | frames: {frame_count}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow("Recording Test Video", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

        if RECORD_SECONDS is not None and elapsed >= RECORD_SECONDS:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    duration = time.time() - start_time
    measured_fps = frame_count / duration if duration > 0 else 0

    print("\nRecording finished.")
    print(f"Saved video: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f}s")
    print(f"Measured FPS: {measured_fps:.2f}")


if __name__ == "__main__":
    record_video()
