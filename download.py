import os
import argparse
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--new", action="store_true", help="Download YOLOv26 instead of YOLOv8")
args = parser.parse_args()

WEIGHTS = {
    "yolov8n.onnx": "https://github.com/SrabanMondal/edge-adas/releases/download/v1.0.0/yolov8n.onnx",
    "yolov26n.onnx": "https://github.com/SrabanMondal/edge-adas/releases/download/v1.0.0/yolov26n.onnx",
    "yolopv2.onnx": "https://github.com/SrabanMondal/edge-adas/releases/download/v1.0.0/yolopv2.onnx",
}

TARGET_DIR = "src/weights"
os.makedirs(TARGET_DIR, exist_ok=True)

# Decide which YOLO model to download
selected_models = ["yolov26n.onnx"] if args.new else ["yolov8n.onnx"]

# Always include yolopv2
selected_models.append("yolopv2.onnx")

for name in selected_models:
    url = WEIGHTS[name]
    dest = os.path.join(TARGET_DIR, name)

    if not os.path.exists(dest):
        print(f"[INFO] Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"[INFO] {name} downloaded successfully and saved to {dest}.")
    else:
        print(f"[INFO] {name} already exists.")