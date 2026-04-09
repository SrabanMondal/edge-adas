import cv2
import numpy as np
import time
import os
import threading
import json
import uvicorn
import socket
import urllib.request
from urllib.parse import urlparse, urlunparse

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return ip


class HttpJpegCamera:
    def __init__(self, shot_url: str, timeout_sec: float = 2.0):
        self.shot_url = shot_url
        self.timeout_sec = timeout_sec
        self._opened = False
        self._last_frame = None
        self._width = 0
        self._height = 0

        frame = self._fetch_frame()
        if frame is not None:
            self._last_frame = frame
            self._height, self._width = frame.shape[:2]
            self._opened = True

    def _fetch_frame(self):
        try:
            with urllib.request.urlopen(self.shot_url, timeout=self.timeout_sec) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def isOpened(self):
        return self._opened

    def grab(self):
        if not self._opened:
            return False
        frame = self._fetch_frame()
        if frame is None:
            return False
        self._last_frame = frame
        self._height, self._width = frame.shape[:2]
        return True

    def retrieve(self):
        if self._last_frame is None:
            return False, None
        return True, self._last_frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def release(self):
        self._opened = False


def _to_ipwebcam_shot_url(stream_url: str) -> str:
    parsed = urlparse(stream_url)
    path = parsed.path or "/"
    normalized_path = path.rstrip("/")

    if normalized_path.endswith("/video"):
        new_path = normalized_path[:-6] + "/shot.jpg"
    elif normalized_path in ("", "/"):
        new_path = "/shot.jpg"
    else:
        new_path = normalized_path

    return urlunparse(parsed._replace(path=new_path, query=""))


def _normalize_camera_source(raw_source: str):
    src = str(raw_source).strip()
    if src.isdigit():
        return int(src)

    # IP Webcam URLs are often provided as "ip:port/video" without scheme.
    if "://" not in src:
        src = f"http://{src}"
    return src


def _open_camera_with_fallbacks(source):
    if isinstance(source, int):
        backend_candidates = [
            ("CAP_DSHOW", "DSHOW"),
            ("CAP_MSMF", "MSMF"),
            ("CAP_ANY", "ANY"),
        ]
    else:
        backend_candidates = [
            ("CAP_FFMPEG", "FFMPEG"),
            ("CAP_ANY", "ANY"),
        ]

    for backend_attr, backend_name in backend_candidates:
        if not hasattr(cv2, backend_attr):
            continue
        backend = getattr(cv2, backend_attr)
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"[INFO] Camera opened using backend={backend_name}, source={source}")
            return cap
        cap.release()

    # Final fallback lets OpenCV choose defaults directly.
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        print(f"[INFO] Camera opened using backend=DEFAULT, source={source}")
        return cap
    cap.release()

    # Fallback for IP Webcam style HTTP streams when OpenCV backend cannot decode /video.
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        shot_url = _to_ipwebcam_shot_url(source)
        jpeg_cam = HttpJpegCamera(shot_url)
        if jpeg_cam.isOpened():
            print(f"[INFO] Camera opened using backend=HTTP-JPEG, source={shot_url}")
            return jpeg_cam

    return None

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
 
from src.inference.openvino_engine import InferenceEngine
from src.utils.image import letterbox, unletterbox, scale_boxes
from src.adas.perception.road.segmentation import clean_road_mask
from src.adas.control.mpcv2 import CenterlineMPC
from src.adas.perception.road.road_v2 import RoadPerception
from src.inference.object_engine import ObjectInferenceEngine
from src.adas.perception.object.object_brake import ObjectPerception

YOLOP_MODEL_PATH = os.getenv("YOLOP_MODEL_PATH", "src/weights/yolop/ov/yolop.xml")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "src/weights/yolo/ov/yolo26n.xml")
IMG_SIZE = 256
CAMERA_IP = os.getenv("CAMERA_IP", "0")
# Global telemetry state
telemetry = {
    "steer": 0.0,
    "brake": 0.0,
    "fps": 0.0,
    "latency": 0.0
}
telemetry_lock = threading.Lock()

# Global running flag
is_running = False
camera_thread = None

def inference_loop():
    global is_running, telemetry

    camera_source = _normalize_camera_source(CAMERA_IP)
    cap = _open_camera_with_fallbacks(camera_source)
    if cap is None:
        print(f"[ERROR] Could not open camera source: {CAMERA_IP}")
        if isinstance(camera_source, str):
            print("[HINT] For IP Webcam, use full URL like http://<phone-ip>:8080/video")
            print("[HINT] If /video fails on your OpenCV build, try a local USB camera index (CAMERA_IP=0)")
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Camera initialized at {w}x{h}")

    # Initialize engines
    engine = InferenceEngine(YOLOP_MODEL_PATH, device="CPU")
    road_engine = RoadPerception()
    object_engine = ObjectInferenceEngine(YOLO_MODEL_PATH, device="CPU")
    object_perception = ObjectPerception(w, h)
    mpc = CenterlineMPC(w, h)
    
    print("[INFO] ADAS Models Initialized. Starting inference loop.")

    frame_idx = 0
    t_last_inference = time.perf_counter()

    while is_running:
        # We read from camera continuously but run inference on every 4th frame
        # to loosely match an ~7.5 FPS processing rate on a 30 FPS camera.
        ret = cap.grab()
        if not ret:
            time.sleep(0.01)
            continue
            
        frame_idx += 1
        if frame_idx % 3 != 0:
            continue
            
        ret, frame = cap.retrieve()
        if not ret:
            continue
            
        t_start = time.perf_counter()
        
        # Preprocessing
        boxed = letterbox(frame)
        
        # Inference
        drive_logits = engine.infer(boxed)
        object_outputs = object_engine.infer(boxed)
        # print(f"[DEBUG] Drive Logits Shape: {drive_logits.shape}")
        # Postprocessing Road
        if drive_logits.shape[0] == 1:
            drive_mask_320 = (drive_logits[0] > 0).astype(np.uint8)
        else:
            drive_mask_320 = (drive_logits[1] > drive_logits[0]).astype(np.uint8)
        # print(f"[DEBUG] Drive Mask Unique Values: {np.unique(drive_mask_320)}")
        drive_mask = unletterbox(drive_mask_320, frame.shape[:2])
        out = road_engine.process(drive_mask)
        center_pts = out["center_points"]
        # print(f"[DEBUG] Center Points: {len(center_pts)} - Sample: {center_pts[:5]}")
        # Control MPC
        steer, traj = mpc.compute(
            road_mask=drive_mask,
            center_points=center_pts,
            gps_bias=0
        )
        # print("Raw Steer Value:", steer)
        
        # Postprocessing Objects
        unletterboxed_objs = scale_boxes(object_outputs, frame.shape[:2])
        brake, dist = object_perception.filter_and_control(unletterboxed_objs, 10)
        
        t_end = time.perf_counter()
        latency = (t_end - t_start) * 1000
        fps = 1.0 / (t_end - t_last_inference) if (t_end - t_last_inference) > 0 else 0
        t_last_inference = t_end
        
        # Update global telemetry with simple float conversion
        with telemetry_lock:
            telemetry["steer"] = float(steer)
            telemetry["brake"] = float(brake)
            telemetry["fps"] = float(fps)
            telemetry["latency"] = float(latency)

    cap.release()
    print("[INFO] Camera released. Inference thread stopped.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # your Jetson Nano frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def start_background_thread():
    global is_running, camera_thread
    is_running = True
    camera_thread = threading.Thread(target=inference_loop, daemon=True)
    camera_thread.start()

@app.on_event("shutdown")
def stop_background_thread():
    global is_running, camera_thread
    is_running = False
    if camera_thread:
        camera_thread.join()

@app.get("/api/telemetry/stream")
def telemetry_stream():

    def event_generator():
        while True:
            with telemetry_lock:
                data = telemetry.copy()

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/telemetry")
def get_telemetry():
    with telemetry_lock:
        return telemetry

# Mount static folder for frontend
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("\n[ADAS] Server Started")
    print(f"[ADAS] Local Dashboard   : http://localhost:8000")
    print(f"[ADAS] Network Dashboard : http://{local_ip}:8000\n")
    uvicorn.run("src.camera_api_cpu:app", host="0.0.0.0", port=8000, reload=False)
