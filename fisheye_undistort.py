import argparse
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path


def load_calibration(path):
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeff"]


def build_fisheye_undistort_maps(camera_matrix, dist_coeff, frame_size, balance, scale):
    width, height = frame_size
    dim = (width, height)

    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        dist_coeff,
        dim,
        np.eye(3),
        balance=balance,
        new_size=dim,
        fov_scale=scale,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        dist_coeff,
        np.eye(3),
        new_camera_matrix,
        dim,
        cv2.CV_16SC2,
    )

    return map1, map2, new_camera_matrix


def reencode_with_ffmpeg(temp_video_path, final_output_path, fps):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or add it to PATH.\n"
            "Your undistorted temp video was created, but final MP4 encoding failed."
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video_path),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_output_path),
    ]

    print("\nRe-encoding to web-compatible MP4:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ffmpeg re-encode failed.")

    print(f"Final web-compatible MP4 saved to: {final_output_path}")


def undistort_video(input_video, output_video, calib_file, balance, scale, preview, keep_temp):
    input_path = Path(input_video)
    output_path = Path(output_video)
    calib_path = Path(calib_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_name(output_path.stem + "_temp_uncompressed.avi")

    camera_matrix, dist_coeff = load_calibration(calib_path)

    print("Loaded calibration:")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeff.ravel())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    if not input_fps or input_fps <= 1:
        input_fps = 30.0

    print("\nInput video:")
    print(f"Path: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {input_fps:.2f}")

    map1, map2, new_camera_matrix = build_fisheye_undistort_maps(
        camera_matrix,
        dist_coeff,
        (width, height),
        balance,
        scale,
    )

    print("\nNew Camera Matrix:")
    print(new_camera_matrix)

    # Write a temporary high-quality intermediate file.
    # MJPG in AVI is usually easy for OpenCV to write/read on Windows.
    temp_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(temp_path),
        temp_fourcc,
        input_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open temporary VideoWriter: {temp_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    print("\nUndistorting video...")
    print(f"Temporary output: {temp_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        writer.write(undistorted)
        processed += 1

        if preview:
            preview_frame = np.hstack((frame, undistorted))

            cv2.putText(
                preview_frame,
                "Original",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                preview_frame,
                "Undistorted",
                (width + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            if frame_count > 0:
                cv2.putText(
                    preview_frame,
                    f"{processed}/{frame_count}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Fisheye Undistortion Preview", preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nUndistortion pass complete.")
    print(f"Processed frames: {processed}")

    reencode_with_ffmpeg(temp_path, output_path, input_fps)

    if not keep_temp:
        try:
            temp_path.unlink()
            print(f"Deleted temp file: {temp_path}")
        except OSError:
            print(f"Could not delete temp file: {temp_path}")

    print("\nDone.")
    print(f"Saved compatible MP4 to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Undistort a fisheye video and save a web-compatible H.264 MP4."
    )

    parser.add_argument(
        "input_video",
        help="Path to the distorted input test video.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the final compatible MP4.",
    )

    parser.add_argument(
        "-c",
        "--calib",
        default="fisheye_calib_data.npz",
        help="Path to fisheye calibration .npz file.",
    )

    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="0.0 = more cropped/less black border, 1.0 = wider FOV/more black border.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="FOV scale passed to estimateNewCameraMatrixForUndistortRectify.",
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable side-by-side preview window.",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary AVI file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input_video)

    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_undistorted_h264.mp4")
    else:
        output_path = Path(args.output)

    undistort_video(
        input_video=input_path,
        output_video=output_path,
        calib_file=args.calib,
        balance=args.balance,
        scale=args.scale,
        preview=not args.no_preview,
        keep_temp=args.keep_temp,
    )
