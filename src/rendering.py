from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Any

COLORS = {
    "qr_code": (0, 220, 180),
    "barcode_zone": (255, 180, 0),
    "stamp_zone": (255, 80, 80),
    "signature_zone": (120, 220, 255),
    "text_zone": (140, 120, 255),
}

def annotate_document(image_bgr: np.ndarray, findings: Dict[str, List[Dict[str, Any]]]) -> np.ndarray:
    canvas = image_bgr.copy()
    all_items = []
    for key in ["qr_codes", "barcode_zones", "stamp_zones", "signature_zones", "text_zones"]:
        all_items.extend(findings.get(key, []))
    for item in all_items:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
        label = item["label"]
        color = COLORS.get(label, (255,255,255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        tag = label.replace("_", " ")
        if "score" in item:
            tag += f" {item['score']:.2f}"
        cv2.rectangle(canvas, (x, max(0, y - 28)), (x + max(160, len(tag)*8), y), color, -1)
        cv2.putText(canvas, tag, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 20), 2, cv2.LINE_AA)
    return canvas
