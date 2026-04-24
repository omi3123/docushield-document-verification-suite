from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import append_history, compute_summary, load_history
from src.config import SAMPLE_DIR, ASSETS_DIR
from src.document_parser import cv_to_pil, load_document_pages, pil_to_cv
from src.detectors import analyze_document
from src.rendering import annotate_document

st.set_page_config(
    page_title="DocuShield Vision — Document Verification Command Center",
    page_icon="🛡️",
    layout="wide",
)

PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID = "rgba(148, 163, 184, 0.18)"
TEXT = "#dbe7ff"
MUTED = "#9db1d5"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp{
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, .10), transparent 22%),
                radial-gradient(circle at top right, rgba(99, 102, 241, .12), transparent 24%),
                linear-gradient(180deg,#040b18 0%, #07152d 44%, #07182e 100%);
            color:#eaf2ff;
        }
        .block-container{padding-top:1.25rem;padding-bottom:2rem;max-width:1350px;}
        [data-testid="stSidebar"]{
            background:linear-gradient(180deg, rgba(4,10,24,.96), rgba(7,20,40,.98));
            border-right:1px solid rgba(120,160,255,.12);
        }
        [data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
        h1,h2,h3,h4{color:#eaf2ff;}
        .hero-shell{
            background:linear-gradient(135deg, rgba(9,22,46,.95), rgba(10,34,72,.92));
            border:1px solid rgba(120,160,255,.16);
            border-radius:26px;
            padding:28px 28px 24px 28px;
            box-shadow:0 20px 60px rgba(0,0,0,.28);
            margin-bottom:18px;
        }
        .eyebrow{color:#5ed6ff;font-size:.88rem;font-weight:700;letter-spacing:.22rem;text-transform:uppercase;margin-bottom:.5rem;}
        .hero-title{font-size:2.35rem;line-height:1.06;font-weight:800;margin:0 0 .7rem 0;max-width:950px;}
        .hero-copy{color:#aabddc;font-size:1.06rem;line-height:1.7;max-width:920px;margin-bottom:1rem;}
        .pill-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:.25rem;}
        .pill{background:rgba(83,195,255,.10);border:1px solid rgba(83,195,255,.20);color:#dff6ff;padding:.48rem .8rem;border-radius:999px;font-size:.9rem;font-weight:600;}
        .panel{
            background:linear-gradient(180deg, rgba(10,22,45,.95), rgba(9,21,43,.92));
            border:1px solid rgba(120,160,255,.14);
            border-radius:22px;padding:20px 20px 18px 20px;
            box-shadow:0 10px 40px rgba(0,0,0,.18);margin-bottom:16px;
        }
        .panel.soft{background:linear-gradient(180deg, rgba(10,22,45,.72), rgba(9,21,43,.64));}
        .metric-card{
            background:linear-gradient(180deg, rgba(11,29,58,.88), rgba(10,25,49,.82));
            border:1px solid rgba(120,160,255,.16);
            border-radius:20px;padding:18px 18px 14px 18px;min-height:118px;
        }
        .metric-label{font-size:.82rem;text-transform:uppercase;letter-spacing:.16rem;color:#7eb4ff;font-weight:700;margin-bottom:14px;}
        .metric-value{font-size:2rem;font-weight:800;color:#f6fbff;line-height:1;margin-bottom:10px;}
        .metric-sub{font-size:.92rem;color:#aabddc;line-height:1.45;}
        .flag-box{border-radius:18px;padding:14px 16px;margin-bottom:10px;border:1px solid transparent;}
        .flag-box.good{background:rgba(56,211,159,.08);border-color:rgba(56,211,159,.24);color:#dffcf3;}
        .flag-box.warn{background:rgba(255,206,106,.08);border-color:rgba(255,206,106,.22);color:#fff4d4;}
        .flag-box.bad{background:rgba(255,123,138,.08);border-color:rgba(255,123,138,.22);color:#ffe8ec;}
        .zone-chip{display:inline-block;padding:.42rem .72rem;border-radius:999px;border:1px solid rgba(120,160,255,.16);background:rgba(120,160,255,.08);margin:0 .45rem .55rem 0;color:#eaf2ff;font-size:.9rem;font-weight:600;}
        .side-brand{background:linear-gradient(180deg, rgba(12,28,56,.95), rgba(10,22,43,.95));border:1px solid rgba(120,160,255,.14);border-radius:24px;padding:18px;margin-bottom:18px;}
        .side-brand h2{font-size:1.4rem;margin:0 0 8px 0;}
        .side-brand p{color:#aabddc;line-height:1.6;margin:0;}
        .side-mini{background:rgba(83,195,255,.08);border:1px solid rgba(83,195,255,.16);border-radius:18px;padding:14px;color:#e6faff;font-size:.94rem;line-height:1.55;margin-top:14px;}
        .section-title{font-size:1.25rem;font-weight:800;color:#f3f8ff;margin-bottom:.3rem;}
        .section-copy{color:#aabddc;line-height:1.65;margin-bottom:0;}
        .small-note{color:#aabddc;font-size:.92rem;line-height:1.6;}
        .stTabs [data-baseweb="tab-list"]{gap:10px;background:rgba(6,17,34,.66);border:1px solid rgba(120,160,255,.10);border-radius:18px;padding:6px;}
        .stTabs [data-baseweb="tab"]{background:transparent;border-radius:14px;color:#c9d8ef;padding:10px 16px;font-weight:700;}
        .stTabs [aria-selected="true"]{background:linear-gradient(180deg, rgba(83,195,255,.18), rgba(122,125,255,.18)) !important;color:white !important;}
        div[data-testid="stMetric"]{background:linear-gradient(180deg, rgba(10,22,45,.92), rgba(9,20,39,.9));border:1px solid rgba(120,160,255,.14);padding:14px 16px;border-radius:18px;}
        div[data-testid="stDataFrame"], div[data-testid="stTable"]{border-radius:18px;overflow:hidden;border:1px solid rgba(120,160,255,.14);}
        </style>
        """,
        unsafe_allow_html=True,
    )


def fig_style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    return fig


def html_metric(label: str, value: str, sub: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def hero() -> None:
    c1, c2 = st.columns([1.55, 0.9], gap="large")
    with c1:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="eyebrow">Document authenticity command center</div>
                <div class="hero-title">DocuShield Vision for QR, stamp, signature, barcode, and layout verification</div>
                <div class="hero-copy">
                    Built for receipts, invoices, apostille-style pages, approval copies, and verification workflows
                    where visual trust signals matter. The interface is tuned for client demos, audit reviews,
                    evidence export, and fast operational triage.
                </div>
                <div class="pill-row">
                    <span class="pill">QR decode</span>
                    <span class="pill">Barcode zoning</span>
                    <span class="pill">Stamp region scan</span>
                    <span class="pill">Signature block check</span>
                    <span class="pill">Text layout mapping</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.image(str(ASSETS_DIR / "hero-preview.png"), use_container_width=True)


def sidebar():
    st.sidebar.markdown(
        """
        <div class="side-brand">
            <div class="eyebrow">DocuShield Vision</div>
            <h2>Verification workflow</h2>
            <p>Use bundled document samples or your own upload. This build is tuned for receipts, invoices, stamped pages, and QR-bearing documents.</p>
            <div class="side-mini">Best portfolio demo flow: sample receipt with QR → annotated evidence → risk summary → analytics → API contract preview.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source = st.sidebar.radio("Input source", ["Use bundled sample", "Upload my own file"])
    if source == "Use bundled sample":
        selected = st.sidebar.selectbox(
            "Bundled file",
            ["sample_invoice_full.png", "sample_invoice_no_qr.png", "sample_receipt_qr.png", "sample_invoice_full.pdf"],
        )
        file_path = str(SAMPLE_DIR / selected)
        return {"mode": "sample", "file_path": file_path, "display_name": selected}

    upload = st.sidebar.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"])
    if upload is not None:
        suffix = Path(upload.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(upload.read())
        tmp.flush()
        return {"mode": "upload", "file_path": tmp.name, "display_name": upload.name}
    return {"mode": "none", "file_path": None, "display_name": None}


def findings_counts(findings):
    return {
        "qr": len(findings["qr_codes"]),
        "barcode": len(findings["barcode_zones"]),
        "stamp": len(findings["stamp_zones"]),
        "signature": len(findings["signature_zones"]),
        "text": len(findings["text_zones"]),
    }


def summary_strip(history_df: pd.DataFrame) -> None:
    sm = compute_summary(history_df)
    cols = st.columns(6)
    cards = [
        ("Audit runs", f"{sm['runs']}", "Total verification sessions logged in the command center."),
        ("Verified share", f"{sm['verified_share']:.0f}%", "Documents that passed the main visual trust checks."),
        ("Avg risk", f"{sm['avg_risk']:.1f}", "Composite portfolio-level risk signal across processed documents."),
        ("QR hits", f"{sm['qr_hits']}", "Runs with at least one decodable QR trace."),
        ("Stamp hits", f"{sm['stamp_hits']}", "Runs where a likely stamp region was surfaced."),
        ("Signature hits", f"{sm['signature_hits']}", "Runs with a signature candidate detected."),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(html_metric(label, value, sub), unsafe_allow_html=True)


def issue_boxes(issue_flags: list[str], findings: dict) -> None:
    if issue_flags:
        severity = "warn" if findings["risk_score"] <= 50 else "bad"
        title = "Review signals" if severity == "warn" else "Flagged signals"
        body = "<br>".join([f"• {msg}" for msg in issue_flags])
        st.markdown(f'<div class="flag-box {severity}"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="flag-box good"><strong>Verified profile</strong><br>Main verification surfaces were found and the page passed the current heuristic checks.</div>', unsafe_allow_html=True)


def risk_badge(risk_score: int, status: str) -> None:
    tone = "good" if status == "verified" else ("warn" if status == "review" else "bad")
    st.markdown(
        f"""
        <div class="flag-box {tone}">
            <strong>{status.upper()}</strong><br>
            Risk score: <strong>{risk_score}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_panel(input_payload):
    st.markdown('<div class="section-title">Document Studio</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Run a page through the document verification pipeline, inspect annotations, decode QR traces, and export evidence rows.</p>', unsafe_allow_html=True)

    if input_payload["mode"] == "none":
        st.markdown('<div class="flag-box warn"><strong>No file loaded yet</strong><br>Choose a bundled sample or upload your own document from the sidebar to start the audit workflow.</div>', unsafe_allow_html=True)
        return

    pages = load_document_pages(input_payload["file_path"])
    st.markdown(f"""
        <div class="panel soft">
            <div class="eyebrow">Loaded asset</div>
            <div class="section-title" style="margin-bottom:.15rem;">{input_payload['display_name']}</div>
            <p class="section-copy">Parsed <strong>{len(pages)}</strong> page(s) for visual trust-signal extraction and evidence review.</p>
        </div>
    """, unsafe_allow_html=True)

    for page_num, pil_img in pages:
        st.markdown(f"### Page {page_num}")
        img_bgr = pil_to_cv(pil_img)
        findings = analyze_document(img_bgr)
        annotated = annotate_document(img_bgr, findings)
        counts = findings_counts(findings)

        c1, c2 = st.columns([1.18, 0.82], gap="large")
        with c1:
            st.image(cv_to_pil(annotated), caption=f"Annotated verification view · page {page_num}", use_container_width=True)

        with c2:
            risk_badge(findings["risk_score"], findings["status"])
            metric_cols = st.columns(5)
            metric_cols[0].metric("QR", counts["qr"])
            metric_cols[1].metric("Barcode", counts["barcode"])
            metric_cols[2].metric("Stamp", counts["stamp"])
            metric_cols[3].metric("Sign", counts["signature"])
            metric_cols[4].metric("Text", counts["text"])

            st.markdown('<div class="panel soft">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Detected surfaces</div>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="zone-chip">{label}</span>' for label in [f"QR payloads: {counts['qr']}", f"Barcode zones: {counts['barcode']}", f"Stamp regions: {counts['stamp']}", f"Signature blocks: {counts['signature']}", f"Text zones: {counts['text']}"]]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            issue_flags = []
            if not findings["qr_codes"]:
                issue_flags.append("Missing QR trace or QR not decodable")
            if not findings["stamp_zones"]:
                issue_flags.append("Approval stamp not found")
            if not findings["signature_zones"]:
                issue_flags.append("Signature block not found")
            if len(findings["text_zones"]) < 3:
                issue_flags.append("Document layout looks sparse for a standard invoice or receipt")
            issue_boxes(issue_flags, findings)

            with st.expander("QR payloads", expanded=bool(findings["qr_codes"])):
                if findings["qr_codes"]:
                    for item in findings["qr_codes"]:
                        st.code(item.get("data", ""), language=None)
                else:
                    st.write("No QR payload decoded on this page.")

            export_rows = []
            for bucket_name in ["qr_codes", "barcode_zones", "stamp_zones", "signature_zones", "text_zones"]:
                for item in findings[bucket_name]:
                    export_rows.append({"type": item["label"], **{k: v for k, v in item.items() if k != "label"}})

            if export_rows:
                out_df = pd.DataFrame(export_rows)
                st.dataframe(out_df, use_container_width=True, hide_index=True, height=220)
                st.download_button("Download page evidence CSV", out_df.to_csv(index=False).encode("utf-8"), file_name=f"page_{page_num}_evidence.csv", mime="text/csv", key=f"evidence_download_{page_num}", use_container_width=True)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asset_name": input_payload["display_name"],
            "pages": len(pages),
            "qr_decoded": counts["qr"],
            "barcode_candidates": counts["barcode"],
            "stamp_regions": counts["stamp"],
            "signature_regions": counts["signature"],
            "text_zones": counts["text"],
            "risk_score": findings["risk_score"],
            "status": findings["status"],
        }
        append_history(record)


def batch_panel():
    st.markdown('<div class="section-title">Batch Audit</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Review the seeded run history to show how the workflow behaves when multiple documents are processed in sequence.</p>', unsafe_allow_html=True)
    history = load_history().sort_values("timestamp", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in ledger", len(history))
    c2.metric("Flagged runs", int((history["status"] == "flagged").sum()) if not history.empty else 0)
    c3.metric("Review runs", int((history["status"] == "review").sum()) if not history.empty else 0)
    st.dataframe(history, use_container_width=True, hide_index=True, height=430)
    st.download_button("Download seeded run history", history.to_csv(index=False).encode("utf-8"), file_name="docushield_run_history.csv", mime="text/csv", key="download_history", use_container_width=True)


def analytics_panel():
    st.markdown('<div class="section-title">Zone Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Monitor verification mix, risk movement, and how frequently each trust surface is being detected.</p>', unsafe_allow_html=True)
    history = load_history()
    if history.empty:
        st.markdown('<div class="flag-box warn"><strong>No analytics available yet</strong><br>Process at least one document to generate chart-ready history.</div>', unsafe_allow_html=True)
        return

    c1, c2 = st.columns(2)
    status_counts = history["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig1 = px.bar(status_counts, x="status", y="count", color="status", title="Audit status mix", color_discrete_map={"verified": "#38d39f", "review": "#ffce6a", "flagged": "#ff7b8a"})
    fig1.update_traces(marker_line_width=0)
    fig_style(fig1, 320)
    c1.plotly_chart(fig1, use_container_width=True, key="status_mix_chart")

    hist = history.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    fig2 = px.line(hist, x="timestamp", y="risk_score", markers=True, title="Risk score over recent audits")
    fig2.update_traces(line_color="#53c3ff", marker_color="#7a7dff")
    fig_style(fig2, 320)
    c2.plotly_chart(fig2, use_container_width=True, key="risk_trend_chart")

    zone_df = pd.DataFrame({"zone": ["QR hits", "Barcode zones", "Stamp regions", "Signature regions", "Text zones"], "avg_count": [history["qr_decoded"].mean(), history["barcode_candidates"].mean(), history["stamp_regions"].mean(), history["signature_regions"].mean(), history["text_zones"].mean()]})
    fig3 = px.bar(zone_df, x="zone", y="avg_count", title="Average detections by verification surface", color="zone")
    fig3.update_traces(marker_line_width=0)
    fig_style(fig3, 350)
    st.plotly_chart(fig3, use_container_width=True, key="zone_avg_chart")


def api_panel():
    st.markdown('<div class="section-title">API Console</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Show the backend integration story for deployment workflows where the frontend and detection layer are separated.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### POST `/detect/document`")
        st.code('curl -X POST "http://127.0.0.1:8000/detect/document" \\\n  -F "file=@sample_media/sample_invoice_full.png"', language="bash")
    with c2:
        st.markdown("#### GET `/health`")
        st.code('{"status": "ok", "service": "docushield-vision-api"}', language="json")
        st.markdown('<p class="small-note">Run <code>python -m uvicorn api.main:app --reload</code> and open <code>/docs</code> to inspect the live API schema.</p>', unsafe_allow_html=True)


def tabs_ui(input_payload):
    tabs = st.tabs(["Executive Overview", "Document Studio", "Batch Audit", "Zone Analytics", "API Console"])
    with tabs[0]:
        history = load_history()
        summary_strip(history)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.1, 0.9], gap="large")
        with c1:
            st.markdown("""
                <div class="panel">
                    <div class="eyebrow">Use case fit</div>
                    <div class="section-title">Built for document verification instead of generic object detection</div>
                    <p class="section-copy">
                        This project is intentionally focused on receipts, invoices, stamped pages, QR-bearing documents,
                        and approval workflows. That makes the output far more believable for admin-tech, compliance,
                        verification, and fraud-screening use cases.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            recent = history.sort_values("timestamp", ascending=False).head(8)
            st.dataframe(recent, use_container_width=True, hide_index=True, height=280)
        with c2:
            st.image(str(ASSETS_DIR / "workflow-preview.png"), use_container_width=True)
            st.markdown("""
                <div class="panel soft">
                    <div class="eyebrow">Client-facing highlights</div>
                    <div class="section-title">Why this portfolio piece works</div>
                    <p class="section-copy">
                        QR decoding, layout zoning, stamp and signature checks, evidence export, run history, and API readiness
                        are all surfaced in one product-style interface.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    with tabs[1]:
        process_panel(input_payload)
    with tabs[2]:
        batch_panel()
    with tabs[3]:
        analytics_panel()
    with tabs[4]:
        api_panel()


def main():
    inject_css()
    hero()
    payload = sidebar()
    tabs_ui(payload)


if __name__ == "__main__":
    main()
