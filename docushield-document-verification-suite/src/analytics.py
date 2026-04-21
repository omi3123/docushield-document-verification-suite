from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.config import HISTORY_PATH

def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=["timestamp","asset_name","pages","qr_decoded","barcode_candidates","stamp_regions","signature_regions","text_zones","risk_score","status"])

def append_history(record: dict) -> None:
    df = load_history()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)

def compute_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "runs": 0,
            "verified_share": 0.0,
            "avg_risk": 0.0,
            "qr_hits": 0,
            "stamp_hits": 0,
            "signature_hits": 0,
        }
    return {
        "runs": int(len(df)),
        "verified_share": float((df["status"] == "verified").mean() * 100),
        "avg_risk": float(df["risk_score"].mean()),
        "qr_hits": int((df["qr_decoded"] > 0).sum()),
        "stamp_hits": int((df["stamp_regions"] > 0).sum()),
        "signature_hits": int((df["signature_regions"] > 0).sum()),
    }
