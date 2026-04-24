from __future__ import annotations
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np

def _box_dict(x, y, w, h, label, score=None, extra=None):
    payload = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "label": label}
    if score is not None:
        payload["score"] = float(score)
    if extra:
        payload.update(extra)
    return payload

def detect_qr_codes(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    detector = cv2.QRCodeDetector()
    outputs = []
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image_bgr)
        if ok and points is not None:
            for data, pts in zip(decoded_info, points):
                pts = np.array(pts, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                outputs.append(_box_dict(x, y, w, h, "qr_code", 0.95, {"data": data or ""}))
            if outputs:
                return outputs
    except Exception:
        pass
    data, pts, _ = detector.detectAndDecode(image_bgr)
    if pts is not None and len(pts) > 0:
        pts = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        outputs.append(_box_dict(x, y, w, h, "qr_code", 0.92, {"data": data or ""}))
    return outputs

def detect_barcode_candidates(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    H, W = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / max(h, 1)
        area = w * h
        if area > W * H * 0.005 and ratio > 2.0 and h > 50:
            results.append(_box_dict(x, y, w, h, "barcode_zone", min(0.88, 0.5 + ratio / 10.0)))
    results = sorted(results, key=lambda r: r["w"] * r["h"], reverse=True)[:3]
    return results

def detect_stamp_regions(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # red masks
    red1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 60), (180, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 50), (135, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    H, W = mask.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < W * H * 0.002:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        if circ > 0.2 and 0.5 < w / max(h, 1) < 1.8:
            score = min(0.94, 0.55 + circ / 1.4)
            results.append(_box_dict(x, y, w, h, "stamp_zone", score))
    return sorted(results, key=lambda r: r["score"], reverse=True)[:4]

def detect_signature_regions(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    roi = gray[int(H * 0.58):, :]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 185, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y += int(H * 0.58)
        area = w * h
        ratio = w / max(h, 1)
        if area > W * H * 0.005 and ratio > 2.4 and h < H * 0.12:
            score = min(0.91, 0.55 + min(ratio, 8) / 20)
            results.append(_box_dict(x, y, w, h, "signature_zone", score))
    return sorted(results, key=lambda r: r["score"], reverse=True)[:3]

def detect_text_zones(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 12))
    morph = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > W * H * 0.002 and w > 180 and h > 40:
            results.append(_box_dict(x, y, w, h, "text_zone", 0.7))
    results = sorted(results, key=lambda r: r["w"] * r["h"], reverse=True)[:14]
    return results

def analyze_document(image_bgr: np.ndarray) -> Dict[str, Any]:
    qr = detect_qr_codes(image_bgr)
    barcodes = detect_barcode_candidates(image_bgr)
    stamps = detect_stamp_regions(image_bgr)
    signatures = detect_signature_regions(image_bgr)
    text = detect_text_zones(image_bgr)

    score = 100
    if not qr:
        score -= 25
    if not stamps:
        score -= 20
    if not signatures:
        score -= 15
    if len(text) < 3:
        score -= 10
    if not barcodes:
        score -= 5
    risk_score = max(5, min(95, 100 - score if score < 100 else 8))
    if risk_score <= 20:
        status = "verified"
    elif risk_score <= 50:
        status = "review"
    else:
        status = "flagged"

    return {
        "qr_codes": qr,
        "barcode_zones": barcodes,
        "stamp_zones": stamps,
        "signature_zones": signatures,
        "text_zones": text,
        "risk_score": int(risk_score),
        "status": status,
    }
