import cv2
from src.detectors import analyze_document

def test_invoice_has_basic_regions():
    img = cv2.imread("sample_media/sample_invoice_full.png")
    findings = analyze_document(img)
    assert len(findings["text_zones"]) >= 3
    assert findings["risk_score"] >= 0
