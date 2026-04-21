# DocuShield Vision — Document Verification Command Center

A client-facing computer vision portfolio project focused on **documents**, not generic object detection.

## What it does

DocuShield Vision audits uploaded **receipts, invoices, PDF pages, approval copies, and QR-bearing documents** and highlights five verification surfaces:

- **QR code detection and decode**
- **Barcode zone localization**
- **Stamp region detection**
- **Signature block detection**
- **Text zone mapping**

It then computes a review-oriented **risk score** and presents the result in a polished Streamlit dashboard plus a FastAPI backend.

## Why this direction is stronger

Instead of pretending that a generic YOLO model can understand every random image, this project is intentionally **domain-specific**. It is built around document verification workflows that are believable for real operations:

- invoice intake
- receipt review
- stamped approval copies
- QR-based trace checks
- signature presence validation
- batch audit evidence export

## Stack

- Python
- Streamlit
- OpenCV
- PyMuPDF
- Plotly
- FastAPI

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Optional API:

```bash
python -m uvicorn api.main:app --reload
```

## Included sample files

- `sample_media/sample_invoice_full.png`
- `sample_media/sample_invoice_no_qr.png`
- `sample_media/sample_receipt_qr.png`
- `sample_media/sample_invoice_full.pdf`

## Notes

- QR decoding is real and uses OpenCV's QR detector.
- Barcode handling is a **candidate-zone locator**, not a full barcode decoder.
- Stamp, signature, and text-zone detection are **visual heuristics** designed for a portfolio-grade workflow and should be treated as rule-based verification layers.
- This is more credible for portfolio use than a generic object detector pasted onto random receipts.

## Portfolio positioning

This project can be presented as:

- **Document Verification Command Center**
- **Receipt & Invoice Audit Vision Suite**
- **QR / Stamp / Signature Forensics Dashboard**
- **Apostille / administrative document intake assistant**

## Screenshots to capture

- Executive Overview
- Document Studio with annotated invoice
- Document Studio with receipt
- Batch Audit
- Zone Analytics
- API Console
