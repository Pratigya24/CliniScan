
import numpy as np


def run_detection(det_model, image_np):
    results = det_model.predict(image_np, conf=0.25, verbose=False)
    annotated_img = results[0].plot()

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = det_model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"class": cls_name, "confidence": conf})

    return annotated_img, detections
