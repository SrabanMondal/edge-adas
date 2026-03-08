import numpy as np
import cv2

def decode_if_needed(output):
    """
    Receives (1, 84, N) raw output or already postprocessed (1,M,6) detections.
    Normalize model outputs to (M,6)
    """
    arr = np.asarray(output)
    arr = np.squeeze(arr)

    # Already NMSed
    if arr.ndim == 2 and arr.shape[1] == 6:
        return arr.astype(np.float32)

    # Raw YOLO head (84,N)
    if arr.ndim == 2 and arr.shape[0] > 10:
        return postprocess_nms(arr)

    raise ValueError(f"[ERROR] Unsupported output shape: {arr.shape}")

def postprocess_nms(raw, conf_thres=0.25, iou_thres=0.7):
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
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = raw.squeeze(0)  # (84,N)

        pred = raw.T  # (N,84)
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