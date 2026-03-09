# Theory Documentation: EDGE ADAS Pipeline

This document explains how the ADAS backend works at a conceptual and systems level, based on the source code as the primary reference.

## 1. Project Overview

### What this system is

This project is a real-time monocular ADAS backend. It processes a single camera stream and produces control telemetry for:

- Lateral control: steering command from road geometry
- Longitudinal safety: brake command from object proximity

The runtime server is implemented with FastAPI and exposes telemetry endpoints for a frontend dashboard.

### What problem it solves

The system addresses a practical autonomy stack problem: convert raw camera frames into stable, low-latency driving cues without requiring expensive sensor fusion.

It does this by combining:

- Semantic road segmentation for drivable space
- Object detection for obstacle-aware braking
- A model-predictive style trajectory selector for steering

### Main architecture

At a high level, the architecture has five layers:

1. Input layer: camera frame capture (`cv2.VideoCapture`)
2. Inference layer: road segmentation + object detection (OpenVINO or TensorRT)
3. Perception layer: centerline extraction and object risk filtering
4. Control layer: trajectory scoring and steering output
5. Serving layer: telemetry exposed via HTTP/SSE through FastAPI

Primary entrypoints:

- `src/camera_api.py`: TensorRT-oriented runtime
- `src/camera_api_cpu.py`: OpenVINO-oriented runtime

## 2. System Workflow

### Lifecycle flow

1. FastAPI app starts.
2. Startup hook launches a background inference thread.
3. The thread opens camera stream using `CAMERA_IP` (default `"0"`, local camera index).
4. Models and processing components are created once.
5. Main loop repeatedly grabs frames and performs inference every 3rd frame.
6. Outputs are converted into steering/brake telemetry.
7. Telemetry is published through:
   - `/api/telemetry`: latest snapshot (pull)
   - `/api/telemetry/stream`: server-sent events stream (push)
8. Shutdown hook stops thread and releases resources.

### Data flow per processed frame

1. Acquire frame from camera
    - Uses `cap.grab()` each loop, then `cap.retrieve()` on selected frames.
    - Frame skipping (`frame_idx % 3`) reduces compute load while still reading stream continuously.

2. Preprocess image
    - `letterbox(frame)` resizes with aspect ratio preservation into a square model input (`size=256` in current path).

3. Inference
    - Road branch: `InferenceEngine.infer(...)` -> segmentation logits.
    - Object branch: object engine infer -> detection boxes.

4. Road mask generation
    - If binary channel output: threshold on channel 0.
    - If 2-channel output: class comparison (`channel1 > channel0`).
    - `unletterbox(...)` maps the mask back to the original frame resolution.

5. Road perception
    - `RoadPerception.process(...)` extracts a centerline from the binary drivable mask.
    - Returns `center_points` + a confidence proxy based on number of valid rows.

6. Steering control
    - `CenterlineMPC.compute(...)` evaluates precomputed candidate trajectories.
    - Selects lowest-cost valid trajectory and applies steering EMA smoothing.

7. Object postprocessing and braking
    - `scale_boxes(...)` maps detection boxes from model space back to original frame size.
    - `ObjectPerception.filter_and_control(...)` filters detections by ROI and confidence, estimates closest distance, and computes brake force.

8. Telemetry update
    - Shared dictionary is updated under a thread lock:
    - `steer`
    - `brake`
    - `fps`
    - `latency`

## 3. Module Explanation

This section covers the modules imported by `camera_api.py` and `camera_api_cpu.py`.

### `src/camera_api.py`

Role:

- Main TensorRT-flavored server runtime.

Responsibilities:

- Creates FastAPI app and background camera thread.
- Owns the frame loop and orchestrates all pipeline stages.
- Publishes telemetry and serves static frontend.

Notable details:

- Initializes CUDA context explicitly in the thread (`pycuda.driver`).
- Uses TensorRT engines for both road (`tensorrt_engine.py`) and object detection (`trt_object_engine.py`).
- Imports `clean_road_mask` but does not currently apply it in this file.

### `src/camera_api_cpu.py`

Role:

- OpenVINO-flavored variant intended for non-TensorRT path.

Responsibilities:

- Same orchestration pattern as `camera_api.py`.
- Uses OpenVINO road model and OpenVINO object model wrappers.

Notable details as currently implemented:

- Also creates a CUDA context in `inference_loop` despite being CPU/OpenVINO variant.
- Uses `ObjectInferenceEngine` with `object_engine.infer(...)` call, but the class currently exposes `get_perception(...)`.
- `if __name__ == "__main__"` launches `src.camera_api:app` instead of `src.camera_api_cpu:app`.

These are implementation mismatches to be aware of when running this variant.

### `src/utils/image.py`

Role:

- Geometry and coordinate transforms between original frame space and model input space.

Key functions:

- `letterbox(img, size=256)`:
  - Scales image to fit square input while preserving aspect ratio.
  - Pads with black borders.
- `unletterbox(mask, orig_shape, size=256)`:
  - Removes padding region and rescales mask to original resolution.
  - Includes guard for empty crop edge cases.
- `scale_boxes(boxes, orig_shape, size=256)`:
  - Maps detection coordinates from letterboxed model space back to original frame size.
  - Clips to frame boundaries.

Why this matters:

- It guarantees all downstream logic (ROI tests, MPC, UI overlays) runs in original camera geometry.

### `src/adas/perception/road/segmentation.py`

Role:

- Optional binary mask cleanup for road segmentation.

Key function:

- `clean_road_mask(mask)`:
  - Morphological close then open with 3x3 kernels.
  - Fills small holes and suppresses noise.

Note:

- Imported in both camera APIs but not currently used in those two entrypoints.

### `src/adas/perception/road/road_v2.py`

Role:

- Fast centerline extraction from a binary drivable-area mask.

Core design:

- Samples only lower 60% of the frame.
- Processes every `row_step` rows for speed.
- For each sampled row:
  - Finds leftmost and rightmost drivable pixel.
  - Enforces minimum road width.
- Applies EMA only to left/right geometry arrays across frames.

Output:

- `center_points`: list of (x, y) centerline points.
- `confidence`: simple score proportional to number of extracted points.

### `src/adas/control/mpcv2.py`

Role:

- Computes steering via a lightweight MPC-inspired candidate search.

Core design:

- Precomputes trajectories for fixed steering candidates at initialization.
- Uses bicycle-model kinematics in image-coordinate space.
- Runtime evaluation checks each candidate for:
  - Road feasibility (trajectory points remain on drivable mask)
  - Centerline tracking error
  - GPS-bias alignment error
  - Steering smoothness penalty

Outputs:

- Smoothed steering command
- Chosen trajectory points (for optional visualization)

### `src/adas/perception/object/object_brake.py`

Role:

- Converts detections into brake actuation signal.

Pipeline:

- For each detection:
  - Filter by confidence threshold (`conf >= 0.25` in code comment context).
  - Take bottom-center point of box.
  - Keep only objects inside a trapezoidal forward ROI.
  - Estimate distance from inverse box height (`distance = 1000 / box_h`).
- Uses nearest object for braking logic.

Braking output:

- If nearest distance < `safe_distance` (15m), apply linearly increasing brake force toward `emergency_stop` (3m).
- Returns `(brake_force, closest_distance)`.

### `src/inference/openvino_engine.py`

Role:

- OpenVINO backend for road segmentation model.

Key implementation points:

- Uses OpenVINO `PrePostProcessor` to embed preprocessing in the graph:
  - Input tensor expected as NHWC uint8 BGR
  - Converts to RGB, float, scale /255
  - Model layout NCHW
- Compiles model with latency performance hint.
- Uses model output port named `"drive_area"`.

### `src/inference/object_engine.py`

Role:

- OpenVINO backend for object model.

Key implementation points:

- Similar PPP-based preprocessing pipeline.
- Method `infer(...)` returns raw detections from output tensor.
- Postprocess decodes (nms) YOLO-style output `(1, 84, N)` if needed. (Explained further below in `trt_object_engine.py`)

### `src/inference/tensorrt_engine.py`

Role:

- TensorRT backend for road segmentation model.

Core behavior:

- Loads serialized TRT engine and allocates host/device buffers.
- Preprocesses frame manually to match model expectations:
  - BGR -> RGB
  - float normalization /255
  - NHWC -> NCHW
- Executes async inference on CUDA stream.
- Returns segmentation logits without batch dimension.

### `src/inference/trt_object_engine.py`

Role:

- TensorRT backend for object detection model, including decoding/NMS.

Core behavior:

- Manual preprocessing and TensorRT execution.
- Postprocess decodes YOLO-style output `(1, 84, N)` if needed:
  - Picks best class per anchor
  - Applies confidence threshold
  - Converts `xywh -> xyxy`
  - Performs class-aware NMS using OpenCV
- Returns final detections in `(M, 6)` format:
  - `[x1, y1, x2, y2, confidence, class_id]`

## 4. Core Algorithms and Processing Logic

### A. Letterbox-normalized multi-task perception

The pipeline standardizes arbitrary camera frames into fixed-size model input while preserving aspect ratio. This allows stable model inference and deterministic reverse mapping.

Conceptually:

1. Original frame -> letterbox square
2. Model inference in square space
3. Unletterbox / scale back to original geometry

This geometry consistency is crucial for:

- ROI-based distance logic
- Pixel-accurate trajectory validity checks
- UI overlays

### B. Road centerline extraction from segmentation

`RoadPerception` intentionally avoids heavy postprocessing to prioritize real-time throughput.

Algorithm steps:

1. Restrict analysis to bottom region (where near-field driving signal is strongest).
2. For each sampled row:
   - Find left and right drivable extents.
3. Reject rows that are too narrow (noise/unreliable structure).
4. Smooth left/right boundaries temporally (EMA).
5. Midpoint of smoothed boundaries forms centerline points.

Why this works:

- Row-wise span geometry is robust and cheap.
- EMA on geometry reduces jitter without blurring the segmentation mask itself.

### C. Candidate-trajectory steering (MPC-style)

`CenterlineMPC` uses a practical approximation of MPC:

- finite candidate steering set
- forward simulation per candidate
- cost minimization with constraints

Algorithm steps:

1. Offline at init: precompute trajectories for candidate steering angles.
2. Online per frame:
   - Transform trajectory offsets into image coordinates.
   - Discard candidates leaving drivable mask.
   - Score remaining candidates:
     - centerline deviation
     - GPS bias alignment
     - steering change penalty
3. Pick minimum-cost candidate.
4. Apply output EMA for control smoothness.

Benefits:

- Predictable compute cost.
- Fast enough for embedded/edge constraints.
- Easier to tune than full nonlinear MPC solvers.

### D. Object-based braking logic

Object detections are filtered to likely in-lane threats using a trapezoidal ROI. Distance proxy is estimated from box height, and brake force scales with perceived proximity.

Design intent:

- Focus only on forward driving corridor.
- Use lightweight heuristics to produce immediate safety actuation.
- Keep logic interpretable and tunable.

## 5. CPU vs GPU Design

### Shared architecture

Both implementations follow the same orchestration pattern:

- camera capture
- frame subsampling
- dual inference branches
- road + object perception
- steering + braking control
- telemetry publication

### Backend differences

`camera_api.py` path:

- Road model backend: TensorRT (`tensorrt_engine.py`)
- Object model backend: TensorRT (`trt_object_engine.py`)
- Model artifacts: `.engine`

`camera_api_cpu.py` path:

- Road model backend: OpenVINO (`openvino_engine.py`)
- Object model backend: OpenVINO (`object_engine.py`)
- Model artifacts: `.xml`

### Operational implications

TensorRT path generally targets NVIDIA GPU deployment with explicit CUDA memory management and custom postprocess.

OpenVINO path targets Intel-oriented acceleration and graph-integrated preprocessing through OpenVINO PPP.

## 6. Benchmark Results

> **Test Hardware**

- CPU: Intel Core i5 (2.4 GHz)
- GPU: Intel Iris Xe Graphics
- Edge Device: Jetson Nano

### Average Processing Time per Frame

| Device / Hardware | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Write (ms) | Total (ms) | FPS |
|-------------------|-----------------|----------------|------------------|------------|------------|-----|
| Intel Iris Xe GPU (i5 2.4 GHz) | 10.01 | 22.80 | 8.22 | 0.00 | **41.03** | **24.37** |
| Intel Core i5 CPU Only (2.4 GHz) | 5.39 | 84.65 | 4.38 | 0.00 | **94.42** | **10.59** |
| Jetson Nano | 14.26 | 94.79 | 19.79 | 0.00 | **128.85** | **7.76** |

### Performance Comparison

| Hardware | FPS | Relative Speed |
|----------|-----|---------------|
| Intel Iris Xe GPU | 24.37 | **2.3× faster than CPU** |
| Intel CPU Only | 10.59 | Baseline |
| Jetson Nano | 7.76 | 0.73× CPU |

## 7. System Design Insights

### Why frame skipping is used

The loop decouples camera acquisition frequency from inference frequency. This lowers compute pressure while keeping the stream fresh, which is often preferable to processing stale queued frames.

### Why geometry EMA is used instead of mask EMA

Smoothing extracted geometry is cheaper and avoids temporal lag introduced by repeatedly filtering full masks. It stabilizes steering while preserving responsiveness.

### Why candidate trajectories are precomputed

Precomputation moves expensive kinematic simulation outside the frame loop. Runtime then becomes mostly vectorized checks and lightweight cost accumulation.

### Why perception and control stay heuristic-friendly

The stack emphasizes explainable and debuggable logic:

- deterministic transformations
- interpretable costs
- explicit thresholds

This is useful for rapid iteration and embedded deployment where predictability matters.

### Why telemetry is lock-protected shared state

The inference thread and API thread read/write the same telemetry object. Mutex protection prevents torn reads and inconsistent multi-field snapshots.

## 8. Conclusion

This project implements a compact, real-time ADAS pipeline that combines segmentation-driven steering and detection-driven braking in a service-friendly FastAPI runtime.

Conceptually, its strength is in balancing practical control quality with low computational complexity:

- robust geometric transformations
- lightweight but effective perception
- MPC-inspired steering selection
- simple safety-oriented braking policy

For future developers, the key mental model is:

- image geometry consistency -> reliable perception -> constrained trajectory scoring -> stable control telemetry.

If this invariant is preserved, backend swaps (OpenVINO, TensorRT, future runtimes) and model updates can be integrated with minimal architectural changes.
