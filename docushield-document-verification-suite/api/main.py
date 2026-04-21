from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
import tempfile, os
from src.document_parser import load_document_pages, pil_to_cv
from src.detectors import analyze_document

app = FastAPI(title="DocuShield Vision API")

@app.get("/health")
def health():
    return {"status":"ok","service":"docushield-vision-api"}

@app.post("/detect/document")
async def detect_document(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        pages = load_document_pages(tmp_path)
        page_results = []
        for page_num, pil_img in pages:
            findings = analyze_document(pil_to_cv(pil_img))
            page_results.append({
                "page": page_num,
                "status": findings["status"],
                "risk_score": findings["risk_score"],
                "counts": {
                    "qr_codes": len(findings["qr_codes"]),
                    "barcode_zones": len(findings["barcode_zones"]),
                    "stamp_zones": len(findings["stamp_zones"]),
                    "signature_zones": len(findings["signature_zones"]),
                    "text_zones": len(findings["text_zones"]),
                },
                "qr_payloads": [item.get("data", "") for item in findings["qr_codes"]],
            })
        return {"filename": file.filename, "pages": page_results}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
