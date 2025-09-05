import io
import base64
import numpy as np
from PIL import Image

class DocumentDetector:
    def __init__(self, model_path: str, label_jsonl_path: str):
        # Load the TFLite model using the TensorFlow Lite Interpreter
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get model input details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.input_dtype = self.input_details[0]['dtype']

        # Parse JSONL for labels
        self.labels = []
        with open(label_jsonl_path, 'r') as f:
            label_set = set()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                import json
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'boundingBoxAnnotations' in data:
                    for ann in data['boundingBoxAnnotations']:
                        if 'displayName' in ann:
                            label_set.add(ann['displayName'])
        self.labels = sorted(label_set)

    def detect_base64(self, base64_str: str):
        # Decode the base64 image
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        orig_width, orig_height = img.size

        # Resize to model input
        img_resized = img.resize((self.input_width, self.input_height))
        img_array = np.array(img_resized)

        # Normalize if needed
        if self.input_dtype == np.float32:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(self.input_dtype)

        input_data = np.expand_dims(img_array, axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
        classes = np.squeeze(self.interpreter.get_tensor(self.output_details[1]['index']))
        scores = np.squeeze(self.interpreter.get_tensor(self.output_details[2]['index']))
        count = int(np.squeeze(self.interpreter.get_tensor(self.output_details[3]['index'])))

        results = []
        CONF_THRESH = 0.5
        for i in range(count):
            score = float(scores[i])
            if score < CONF_THRESH:
                continue
            class_idx = int(classes[i])
            if class_idx < 0 or class_idx >= len(self.labels):
                continue
            label = self.labels[class_idx]

            # Convert normalized coords to pixel values
            y_min, x_min, y_max, x_max = boxes[i]
            x_min_pixel = int(x_min * orig_width)
            y_min_pixel = int(y_min * orig_height)
            x_max_pixel = int(x_max * orig_width)
            y_max_pixel = int(y_max * orig_height)

            results.append({
                'label': label,
                'score': round(score, 4),
                'box': [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]
            })
        return results


# Example usage:
# detector = DocumentDetector("model.tflite", "data.jsonl")
# with open("test_img.b64", "r") as f:
#     base64_image = f.read().strip()
# detections = detector.detect_base64(base64_image)
# print(detections)
