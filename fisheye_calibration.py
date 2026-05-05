import argparse
import cv2
import numpy as np
import time
import os
import platform
import subprocess
import shutil

# Windows-only beep support
try:
    import winsound
except ImportError:
    winsound = None

# --- Configuration ---
DEVICE_ID = 1

# IMPORTANT:
# These are the number of ChArUco chessboard squares, not inner corners.
# Your physical printed board must match this exactly.
SQUARES_X = 5
SQUARES_Y = 7

SQUARE_LENGTH = 0.030
MARKER_LENGTH = 0.022

COOLDOWN_TIME = 2.0
TARGET_CAPTURES = 60
MIN_CHARUCO_CORNERS = 15

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SAVE_PATH = "fisheye_calib_data.npz"

# 6 phases × 10 captures = 60 captures
CAPTURES_PER_PHASE = 10

CAPTURE_PHASES = [
    {
        "name": "CENTER FLAT",
        "instruction": "Board centered, mostly flat",
        "detail": "Fill middle area. Keep board fully visible.",
        "spoken": "Phase one. Center flat. Keep the board in the middle and mostly flat.",
    },
    {
        "name": "CENTER TILTED",
        "instruction": "Board centered, tilted",
        "detail": "Tilt up, down, left, and right. Vary angle each shot.",
        "spoken": "Phase two. Center tilted. Keep the board centered, but tilt it in different directions.",
    },
    {
        "name": "IMAGE EDGES",
        "instruction": "Move board near image edges",
        "detail": "Top, bottom, left, right. Keep enough corners visible.",
        "spoken": "Phase three. Image edges. Move the board near the top, right, bottom, and left edges.",
    },
    {
        "name": "IMAGE CORNERS",
        "instruction": "Move board near image corners",
        "detail": "Top-left, top-right, bottom-left, bottom-right.",
        "spoken": "Phase four. Image corners. Move the board near each corner of the image.",
    },
    {
        "name": "CLOSE RANGE",
        "instruction": "Move board close to camera",
        "detail": "Large board in image. Avoid cutting off too much.",
        "spoken": "Phase five. Close range. Move the board close to the camera while keeping enough corners visible.",
    },
    {
        "name": "FAR RANGE / MIX",
        "instruction": "Move board farther away + mixed angles",
        "detail": "Smaller board, varied tilt, revisit weak regions.",
        "spoken": "Phase six. Far range and mixed angles. Move the board farther away and vary the tilt.",
    },
]


class AudioGuide:
    """
    Blocking audio guide.

    On Windows this uses PowerShell + System.Speech.Synthesis for every line.
    That is more reliable than pyttsx3 for repeated phase instructions because
    each utterance runs in a fresh speech process and blocks until it finishes.
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.backend = None
        self.engine = None

        if not self.enabled:
            return

        system_name = platform.system().lower()

        if system_name == "windows":
            powershell = shutil.which("powershell") or shutil.which("powershell.exe")
            pwsh = shutil.which("pwsh") or shutil.which("pwsh.exe")
            self.powershell_exe = powershell or pwsh

            if self.powershell_exe is not None:
                self.backend = "powershell_sapi"
                print("Audio backend: Windows PowerShell SAPI")
                return

            print("PowerShell was not found. Falling back to pyttsx3.")

        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 155)
            self.engine.setProperty("volume", 1.0)
            self.backend = "pyttsx3"
            print("Audio backend: pyttsx3")
        except Exception as e:
            print(f"Audio guide disabled. Could not initialize audio backend: {e}")
            print("On Windows, PowerShell should already exist. Otherwise install pyttsx3 with: pip install pyttsx3")
            self.enabled = False
            self.backend = None
            self.engine = None

    @staticmethod
    def _powershell_quote(text):
        # Single-quoted PowerShell string. Escape single quotes by doubling them.
        return "'" + text.replace("'", "''") + "'"

    def _say_with_powershell_sapi(self, text):
        quoted_text = self._powershell_quote(text)
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speak.Rate = 0; "
            "$speak.Volume = 100; "
            f"$speak.Speak({quoted_text}); "
            "$speak.Dispose();"
        )

        subprocess.run(
            [
                self.powershell_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _say_with_pyttsx3(self, text):
        if self.engine is None:
            return

        self.engine.say(text)
        self.engine.runAndWait()

    def say_blocking(self, text):
        if not self.enabled or self.backend is None:
            return

        print(f"AUDIO: {text}")

        try:
            if self.backend == "powershell_sapi":
                self._say_with_powershell_sapi(text)
            elif self.backend == "pyttsx3":
                self._say_with_pyttsx3(text)
        except Exception as e:
            print(f"Audio guide error: {e}")

    def beep_capture(self):
        if not self.enabled:
            return

        if winsound is not None:
            try:
                winsound.Beep(1200, 120)
            except Exception:
                pass

    def beep_warning(self):
        if not self.enabled:
            return

        if winsound is not None:
            try:
                winsound.Beep(500, 200)
            except Exception:
                pass

    def close(self):
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass


def get_phase(captured_frames):
    phase_index = min(captured_frames // CAPTURES_PER_PHASE, len(CAPTURE_PHASES) - 1)
    phase_start = phase_index * CAPTURES_PER_PHASE
    phase_end = phase_start + CAPTURES_PER_PHASE
    phase_capture_index = captured_frames - phase_start + 1

    return phase_index, phase_start, phase_end, phase_capture_index, CAPTURE_PHASES[phase_index]


def draw_mini_pose_guide(frame, phase_index, phase_capture_index):
    h, w = frame.shape[:2]

    box_w = 210
    box_h = 160
    x0 = w - box_w - 15
    y0 = 15
    x1 = x0 + box_w
    y1 = y0 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (35, 35, 35), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)

    mini_x0 = x0 + 20
    mini_y0 = y0 + 35
    mini_w = 170
    mini_h = 95
    mini_x1 = mini_x0 + mini_w
    mini_y1 = mini_y0 + mini_h

    cv2.rectangle(frame, (mini_x0, mini_y0), (mini_x1, mini_y1), (80, 80, 80), 1)

    cv2.line(
        frame,
        (mini_x0 + mini_w // 2, mini_y0),
        (mini_x0 + mini_w // 2, mini_y1),
        (70, 70, 70),
        1,
    )
    cv2.line(
        frame,
        (mini_x0, mini_y0 + mini_h // 2),
        (mini_x1, mini_y0 + mini_h // 2),
        (70, 70, 70),
        1,
    )

    color = (0, 255, 255)

    if phase_index == 0:
        bx = mini_x0 + mini_w // 2 - 38
        by = mini_y0 + mini_h // 2 - 26
        bw = 76
        bh = 52
        guide_text = "flat"

    elif phase_index == 1:
        bx = mini_x0 + mini_w // 2 - 42
        by = mini_y0 + mini_h // 2 - 24
        bw = 84
        bh = 48
        guide_text = "tilt"

    elif phase_index == 2:
        edge_positions = [
            ("top", mini_x0 + mini_w // 2 - 35, mini_y0 + 6, 70, 42),
            ("right", mini_x1 - 76, mini_y0 + mini_h // 2 - 22, 70, 44),
            ("bottom", mini_x0 + mini_w // 2 - 35, mini_y1 - 48, 70, 42),
            ("left", mini_x0 + 6, mini_y0 + mini_h // 2 - 22, 70, 44),
        ]
        guide_text, bx, by, bw, bh = edge_positions[(phase_capture_index - 1) % 4]

    elif phase_index == 3:
        corner_positions = [
            ("top-left", mini_x0 + 6, mini_y0 + 6, 68, 42),
            ("top-right", mini_x1 - 74, mini_y0 + 6, 68, 42),
            ("bottom-right", mini_x1 - 74, mini_y1 - 48, 68, 42),
            ("bottom-left", mini_x0 + 6, mini_y1 - 48, 68, 42),
        ]
        guide_text, bx, by, bw, bh = corner_positions[(phase_capture_index - 1) % 4]

    elif phase_index == 4:
        bx = mini_x0 + 18
        by = mini_y0 + 12
        bw = mini_w - 36
        bh = mini_h - 24
        guide_text = "close"

    else:
        far_positions = [
            ("far center", mini_x0 + mini_w // 2 - 24, mini_y0 + mini_h // 2 - 16, 48, 32),
            ("far left", mini_x0 + 20, mini_y0 + mini_h // 2 - 16, 48, 32),
            ("far right", mini_x1 - 68, mini_y0 + mini_h // 2 - 16, 48, 32),
            ("far top", mini_x0 + mini_w // 2 - 24, mini_y0 + 10, 48, 32),
            ("far bottom", mini_x0 + mini_w // 2 - 24, mini_y1 - 42, 48, 32),
        ]
        guide_text, bx, by, bw, bh = far_positions[(phase_capture_index - 1) % 5]

    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

    if phase_index == 1:
        if phase_capture_index % 2 == 0:
            cv2.line(frame, (bx, by + bh), (bx + bw, by), color, 2)
        else:
            cv2.line(frame, (bx, by), (bx + bw, by + bh), color, 2)

    cv2.putText(
        frame,
        f"Guide: {guide_text}",
        (x0 + 15, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )


def draw_phase_overlay(frame, captured_frames):
    phase_index, phase_start, phase_end, phase_capture_index, phase = get_phase(captured_frames)

    phase_total = len(CAPTURE_PHASES)
    current_phase_num = phase_index + 1

    cv2.putText(
        frame,
        f"PHASE {current_phase_num}/{phase_total}: {phase['name']}",
        (10, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        phase["instruction"],
        (10, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        phase["detail"],
        (10, 192),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )

    cv2.putText(
        frame,
        f"Phase shot: {phase_capture_index}/{CAPTURES_PER_PHASE}",
        (10, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 255, 180),
        2,
    )

    bar_x = 10
    bar_y = 238
    bar_w = 300
    bar_h = 14

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    progress = (captured_frames - phase_start) / max(1, CAPTURES_PER_PHASE)
    progress = max(0.0, min(1.0, progress))
    fill_w = int(bar_w * progress)

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + fill_w, bar_y + bar_h),
        (0, 180, 255),
        -1,
    )

    draw_mini_pose_guide(frame, phase_index, phase_capture_index)


def create_charuco_tools():
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict,
        )
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        modern_cv2 = True
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict,
        )
        detector_params = cv2.aruco.DetectorParameters_create()
        detector = None
        modern_cv2 = False

    return aruco_dict, board, detector_params, detector, modern_cv2


def get_board_object_points(board):
    try:
        return board.getChessboardCorners()
    except AttributeError:
        return board.chessboardCorners


def detect_markers(gray, aruco_dict, detector_params, detector, modern_cv2):
    if modern_cv2:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params,
        )

    return corners, ids, rejected


def interpolate_charuco(corners, ids, gray, board):
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
    )

    return ret, charuco_corners, charuco_ids


def show_wait_screen(frame, message, actual_height):
    display_frame = frame.copy()
    cv2.putText(
        display_frame,
        message,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        "Capture is paused while audio plays.",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        "Press q/ESC after audio if you need to quit.",
        (10, actual_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.imshow("Fisheye ChArUco Calibration", display_frame)
    cv2.waitKey(1)


def speak_phase_and_wait(audio, frame, captured_frames, actual_height):
    phase_index, phase_start, phase_end, phase_capture_index, phase = get_phase(captured_frames)

    display_frame = frame.copy()
    draw_phase_overlay(display_frame, captured_frames)

    cv2.putText(
        display_frame,
        "PHASE AUDIO - calibration paused",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        "Listen first, then position the board.",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Fisheye ChArUco Calibration", display_frame)
    cv2.waitKey(1)

    audio.say_blocking(phase["spoken"])

    # Real pause between phases, after the spoken line finishes.
    time.sleep(1.0)


def auto_fisheye_calibrate(audio_enabled=False):
    aruco_dict, board, detector_params, detector, modern_cv2 = create_charuco_tools()

    audio = AudioGuide(enabled=audio_enabled)

    all_corners = []
    all_ids = []
    img_shape = None

    cap = cv2.VideoCapture(DEVICE_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        audio.say_blocking("Error. Could not open camera.")
        audio.close()
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 60)
    print("TRUE FISHEYE CHARUCO CALIBRATION STARTED")
    print("=" * 60)
    print(f"Device ID: {DEVICE_ID}")
    print(f"Requested resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Actual resolution:    {actual_width}x{actual_height}")
    print(f"Board squares:        {SQUARES_X}x{SQUARES_Y}")
    print(f"Square length:        {SQUARE_LENGTH} m")
    print(f"Marker length:        {MARKER_LENGTH} m")
    print(f"Target captures:      {TARGET_CAPTURES}")
    print(f"Cooldown:             {COOLDOWN_TIME} seconds")
    print(f"Min ChArUco corners:  {MIN_CHARUCO_CORNERS}")
    print(f"Audio guide:          {'enabled' if audio_enabled else 'disabled'}")
    print("=" * 60)

    captured_frames = 0
    last_capture_time = time.time()
    last_spoken_phase = -1
    last_bad_view_prompt_time = 0.0

    try:
        # Grab one frame before any audio so the window can show the pause state.
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            audio.say_blocking("Frame grab failed.")
            return

        show_wait_screen(frame, "STARTING CALIBRATION - audio first", actual_height)
        audio.say_blocking("Calibration started.")

        # Force phase 1 instruction immediately, before detection/capture can happen.
        speak_phase_and_wait(audio, frame, captured_frames, actual_height)
        last_spoken_phase = 0
        last_capture_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed.")
                audio.say_blocking("Frame grab failed.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if img_shape is None:
                img_shape = gray.shape[::-1]

            display_frame = frame.copy()

            current_time = time.time()
            time_since_capture = current_time - last_capture_time

            phase_index, phase_start, phase_end, phase_capture_index, phase = get_phase(captured_frames)

            # If we just entered a new phase, pause before doing detection/capture.
            if phase_index != last_spoken_phase:
                speak_phase_and_wait(audio, frame, captured_frames, actual_height)
                last_spoken_phase = phase_index
                last_capture_time = time.time()
                current_time = time.time()
                time_since_capture = current_time - last_capture_time

            corners, ids, rejected = detect_markers(
                gray,
                aruco_dict,
                detector_params,
                detector,
                modern_cv2,
            )

            marker_count = 0 if ids is None else len(ids)
            charuco_count = 0
            can_capture = False
            charuco_corners = None
            charuco_ids = None

            if ids is not None and marker_count > 0:
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

                if marker_count >= 4:
                    _, charuco_corners, charuco_ids = interpolate_charuco(
                        corners,
                        ids,
                        gray,
                        board,
                    )

                    if charuco_corners is not None and charuco_ids is not None:
                        charuco_count = len(charuco_corners)

                        cv2.aruco.drawDetectedCornersCharuco(
                            display_frame,
                            charuco_corners,
                            charuco_ids,
                            (0, 255, 0),
                        )

                        if charuco_count >= MIN_CHARUCO_CORNERS:
                            can_capture = True

            if time_since_capture < COOLDOWN_TIME:
                remaining = COOLDOWN_TIME - time_since_capture
                cv2.putText(
                    display_frame,
                    f"MOVE BOARD - next scan in {remaining:.1f}s",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                )
            else:
                if can_capture:
                    cv2.putText(
                        display_frame,
                        "GOOD VIEW - capturing",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)

                    captured_frames += 1
                    last_capture_time = time.time()

                    captured_phase_index, _, _, captured_phase_shot, captured_phase = get_phase(
                        captured_frames - 1
                    )

                    print(
                        f"Captured pose {captured_frames}/{TARGET_CAPTURES} "
                        f"| Phase: {captured_phase['name']} "
                        f"| Phase shot: {captured_phase_shot}/{CAPTURES_PER_PHASE} "
                        f"| ChArUco corners: {charuco_count} "
                        f"| Markers: {marker_count}"
                    )

                    audio.beep_capture()

                    flash = np.zeros_like(frame)
                    flash[:] = (0, 255, 0)
                    display_frame = cv2.addWeighted(display_frame, 0.5, flash, 0.5, 0)

                    if captured_frames >= TARGET_CAPTURES:
                        audio.say_blocking("All captures complete. Calculating calibration.")
                        break
                else:
                    cv2.putText(
                        display_frame,
                        "SCANNING - need better board view",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                    )

                    # Keep this rare and blocking; otherwise pyttsx3 can queue oddly on Windows.
                    if audio_enabled and current_time - last_bad_view_prompt_time > 20.0:
                        audio.say_blocking(
                            "Need a better board view. Keep more ChArUco corners visible."
                        )
                        last_bad_view_prompt_time = time.time()
                        last_capture_time = time.time()

            cv2.putText(
                display_frame,
                f"Captured: {captured_frames}/{TARGET_CAPTURES}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                f"Markers: {marker_count} | ChArUco corners: {charuco_count}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            draw_phase_overlay(display_frame, captured_frames)

            cv2.putText(
                display_frame,
                "Press q/ESC to quit.",
                (10, actual_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Fisheye ChArUco Calibration", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                audio.say_blocking("Calibration stopped.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(all_corners) < 10:
        print(f"\nNot enough valid captures: {len(all_corners)}")
        print("Need at least 10, preferably 40-60 for fisheye.")
        audio.say_blocking("Not enough valid captures. Calibration stopped.")
        audio.close()
        return

    print("\nCalculating fisheye intrinsics. Please wait.")

    board_obj_pts = get_board_object_points(board)

    objpoints = []
    imgpoints = []

    for i in range(len(all_ids)):
        current_ids = all_ids[i]
        current_corners = all_corners[i]

        obj_pts = np.array(
            [board_obj_pts[idx[0]] for idx in current_ids],
            dtype=np.float32,
        )

        objpoints.append(obj_pts)
        imgpoints.append(current_corners)

    objpoints_fe = [
        np.reshape(pts, (pts.shape[0], 1, 3)).astype(np.float64)
        for pts in objpoints
    ]

    imgpoints_fe = [
        np.reshape(pts, (pts.shape[0], 1, 2)).astype(np.float64)
        for pts in imgpoints
    ]

    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    dist_coeff = np.zeros((4, 1), dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_CHECK_COND
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )

    try:
        rms, camera_matrix, dist_coeff, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_fe,
            imgpoints_fe,
            img_shape,
            camera_matrix,
            dist_coeff,
            flags=flags,
            criteria=criteria,
        )
    except cv2.error as e:
        print("\nCalibration failed.")
        print("This usually means the captured poses are poorly conditioned.")
        print("Capture more views with stronger edge/corner coverage and varied tilt.")
        print(e)
        audio.say_blocking(
            "Calibration failed. Capture more views with stronger edge and corner coverage."
        )
        audio.close()
        return

    print("\nFISHEYE CALIBRATION RESULTS")
    print("=" * 60)
    print(f"RMS reprojection error: {rms}")
    print(f"Image shape: {img_shape}")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coeffs k1, k2, k3, k4:")
    print(dist_coeff.T)

    np.savez(
        SAVE_PATH,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
        image_width=img_shape[0],
        image_height=img_shape[1],
        rms=rms,
        squares_x=SQUARES_X,
        squares_y=SQUARES_Y,
        square_length=SQUARE_LENGTH,
        marker_length=MARKER_LENGTH,
    )

    print(f"\nSaved calibration to: {SAVE_PATH}")
    audio.say_blocking("Calibration complete. File saved.")
    audio.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fisheye ChArUco calibration capture tool.")
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable spoken audio guidance and capture beeps.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEVICE_ID,
        help=f"Camera device ID. Default: {DEVICE_ID}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=SAVE_PATH,
        help=f"Output calibration .npz path. Default: {SAVE_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DEVICE_ID = args.device
    SAVE_PATH = args.output

    auto_fisheye_calibrate(audio_enabled=args.audio)
