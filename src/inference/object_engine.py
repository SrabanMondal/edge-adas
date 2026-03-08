import numpy as np
import openvino as ov

from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino import Type, Layout

from src.utils.nms import decode_if_needed

class ObjectInferenceEngine:
    def __init__(self, yolo26_path, device="GPU"):
        self.core = ov.Core()
        
        # 1. Read the INT8 Model
        raw_model = self.core.read_model(yolo26_path)
        
        # 2. Add PrePostProcessor (PPP)
        # This removes the need for manual transpose/scale in your infer function
        ppp = PrePostProcessor(raw_model)
        
        # What you give: (1, 320, 320, 3), UINT8, BGR (from cv2)
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)
            
        # What the model expects: NCHW, Float (PPP handles the INT8 mapping internally)
        ppp.input().model().set_layout(Layout('NCHW'))
        
        # The Math: BGR->RGB, Scale to 0-1
        ppp.input().preprocess() \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .scale(255.0)
            
        # 3. Compile with Latency Hint
        self.yolo26 = self.core.compile_model(
            ppp.build(), 
            device, 
            {"PERFORMANCE_HINT": "LATENCY"}
        )
        self.yolo26_request = self.yolo26.create_infer_request()
        
        # Save output layer info
        self.output_layer = self.yolo26.output(0)
        print(f"[INFO] YOLO26 INT8 Engine Ready on {device}")

    def infer(self, img_256: np.ndarray):
        """
        Args: frame (BGR image from cv2)
        Output: (M ,6)
        """
        # 1. Resize only (PPP handles the rest)
        input_tensor = np.expand_dims(img_256, 0)
        
        # 2. Inference (No manual /255.0 needed!)
        results = self.yolo26_request.infer({0: input_tensor})
        
        # 3. Process Detections
        detections = decode_if_needed(results[self.output_layer])
        return detections
