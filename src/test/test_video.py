import cv2
import os
import sys
import random
from pathlib import Path

DATA_DIR = Path("src/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "input.mp4"
TARGET_FPS = 30


def main(video_path):
    print("\n[INFO] Starting video preparation script")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Input video: {video_path}")
    print(f"[INFO] Resolution : {width}x{height}")
    print(f"[INFO] FPS        : {fps:.2f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Handle existing input.mp4
    if OUTPUT_FILE.exists():
        rand = random.randint(1000, 9999)
        renamed = DATA_DIR / f"input_{rand}.mp4"
        OUTPUT_FILE.rename(renamed)

        print(f"[INFO] Existing input.mp4 found")
        print(f"[INFO] Renamed previous file -> {renamed}")

    # Determine frame skipping
    if fps > TARGET_FPS:
        skip = round(fps / TARGET_FPS)
        print(f"[INFO] FPS greater than {TARGET_FPS}")
        print(f"[INFO] Frame sampling ratio: 1/{skip}")
    else:
        skip = 1
        print("[INFO] FPS already suitable. No downsampling needed.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, TARGET_FPS, (width, height))

    frame_idx = 0
    saved = 0

    print("[INFO] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            out.write(frame)
            saved += 1

        frame_idx += 1

    cap.release()
    out.release()

    print("\n[INFO] Processing complete")
    print(f"[INFO] Total frames read : {frame_idx}")
    print(f"[INFO] Frames saved      : {saved}")
    print(f"[INFO] Output saved to   : {OUTPUT_FILE}\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python -m src.test.test_video.py <path_to_video>\n")
        sys.exit(1)

    video_path = sys.argv[1]
    main(video_path)