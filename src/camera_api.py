import cv2
import numpy as np
import time
import os
import threading
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import uvicorn
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return ip

from src.inference.tensorrt_engine import InferenceEngine
from src.utils.image import letterbox, unletterbox, scale_boxes
from src.adas.perception.road.segmentation import clean_road_mask
from src.adas.control.mpcv2 import CenterlineMPC
from src.adas.perception.road.road_v2 import RoadPerception
from src.inference.trt_object_engine import TRTObjectInferenceEngine
from src.adas.perception.object.object_brake import ObjectPerception

YOLOP_MODEL_PATH = "src/weights/yolop/yolop.engine"
YOLO_MODEL_PATH = "src/weights/yolo/yolo.engine"
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
    import pycuda.driver as cuda
    cuda.init()
    ctx = cuda.Device(0).make_context()
    global is_running, telemetry
    
    cap = cv2.VideoCapture(CAMERA_IP)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at IP: {CAMERA_IP}")
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Camera initialized at {w}x{h}")

    # Initialize engines
    engine = InferenceEngine(YOLOP_MODEL_PATH)
    road_engine = RoadPerception()
    object_engine = TRTObjectInferenceEngine(YOLO_MODEL_PATH)
    object_perception = ObjectPerception(w, h)
    mpc = CenterlineMPC(w,h)
    
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
    ctx.pop()

app = FastAPI()

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
    uvicorn.run("src.camera_api:app", host="0.0.0.0", port=8000, reload=False)
