import os
import cv2
import numpy as np

from src.inference.trt_object_engine import TRTObjectInferenceEngine
from src.utils.image import letterbox, scale_boxes

DEBUG_OUTPUT_PATH = "src/data/yolo_trt_debug.mp4"

class TRTDebugRunner:

    def __init__(self, engine_path: str):
        self.engine = TRTObjectInferenceEngine(engine_path)

    def run_on_video(self, mp4_path: str):
        assert os.path.exists(mp4_path), f"[ERROR] MP4 file does not exist: {mp4_path}"

        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError("[ERROR] Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            DEBUG_OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H)
        )

        print(f"[INFO] Running debug inference on video: {mp4_path}")
        print(f"[INFO] Original resolution = {W}x{H}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # -----------------------------
                # 1. LETTERBOX to 256x256
                # -----------------------------
                img256 = letterbox(frame, size=256)

                # -----------------------------
                # 2. Run TensorRT inference
                # -----------------------------
                raw = self.engine.infer(img256)

                # -----------------------------
                # 3. Scale boxes back to original resolution
                # -----------------------------
                dets_scaled = scale_boxes(raw, (H, W), size=256)

                # -----------------------------
                # 4. Draw boxes
                # -----------------------------
                for x1, y1, x2, y2, conf, cls_id in dets_scaled:
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"{cls_id}:{conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

            except Exception as e:
                print(f"[ERROR] Inference error: {e}")

            writer.write(frame)

        cap.release()
        writer.release()

        print(f"[INFO] Debug video written to {DEBUG_OUTPUT_PATH}")


if __name__ == "__main__":
    runner = TRTDebugRunner("src/weights/yolo/yolo.engine")
    runner.run_on_video("src/data/input.mp4")