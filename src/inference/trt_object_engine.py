import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from src.utils.nms import decode_if_needed

class TRTObjectInferenceEngine:
    """
    Clean, production-safe TensorRT inference engine.
    Fully compatible with OpenVINO output format.
    """

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))

            if self.engine.binding_is_input(idx):
                self.input_shape = shape  # (1, 3, 320, 320)
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_shape = shape  # e.g. (1, 84, 2100)
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess for TensorRT engine.
        """
        _, _, H, W = self.input_shape
        img = cv2.resize(frame, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(np.expand_dims(img, 0))

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns (1, 300, 6) detections exactly like OpenVINO.
        """
        img = self._preprocess(frame)
        np.copyto(self.host_inputs[0], img.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        # Expect engine output shape = (1, 84, 2100)
        arr = self.host_outputs[0].reshape(self.output_shape)
        return decode_if_needed(arr)

    def _postprocess(self, raw, conf_thres=0.25, iou_thres=0.7):
        """
        Decode raw YOLOv8 ONNX output into final detections.

        Parameters
        ----------
        raw : np.ndarray
            Raw model output, shape (1, 84, N) where 84 = 4 box + 80 classes
            and N = number of anchors (e.g. 2100 for 320×320 input).
            Box coords (cx, cy, w, h) are in input pixel space.
        conf_thres : float
            Minimum class confidence to keep a detection.
        iou_thres : float
            IoU threshold for NMS.

        Returns
        -------
        np.ndarray, shape (M, 6)
            Each row: [x1, y1, x2, y2, confidence, class_id]
            Coordinates are in the model's input pixel space (e.g. 320×320).
            Returns empty (0, 6) array if no detections.
        """
        # (1, 84, N) → (N, 84)
        pred = np.squeeze(raw).T

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]

        # Best class per anchor
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(pred)), class_ids]

        # Confidence filter
        mask = confidences > conf_thres
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # xywh → xyxy  (coords already in pixel space)
        boxes = np.empty((len(boxes_xywh), 4), dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2   # x1
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2   # y1
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2   # x2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2   # y2

        # Class-aware NMS
        keep = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]

            idxs = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(),
                cls_scores.tolist(),
                conf_thres,
                iou_thres,
            )
            if len(idxs) > 0:
                keep.extend(np.where(cls_mask)[0][idxs.flatten()])

        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float32)

        keep = np.array(keep)
        # Stack into (M, 6): [x1, y1, x2, y2, conf, class_id]
        detections = np.column_stack([
            boxes[keep],
            confidences[keep],
            class_ids[keep].astype(np.float32),
        ])
        return detections