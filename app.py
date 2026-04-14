"""
Streamlit UI — Return Fraud Validator
Real-time testing with image upload, JSON output, latency metrics.
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
import time
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Return Fraud Validator",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0d1b2a; }
    .stApp { background: #0d1b2a; }
    .result-yes { background:#1a4731; border-left:4px solid #2ecc71;
                  padding:12px; border-radius:8px; color:#2ecc71; font-weight:bold; }
    .result-no  { background:#4a1a1a; border-left:4px solid #e74c3c;
                  padding:12px; border-radius:8px; color:#e74c3c; font-weight:bold; }
    .metric-box { background:#162b3e; border-radius:10px; padding:16px;
                  text-align:center; border:1px solid #1e3a52; }
    .metric-val { font-size:1.8rem; font-weight:700; color:#00b4d8; }
    .metric-lbl { font-size:0.8rem; color:#8899aa; margin-top:4px; }
    .field-row  { background:#162b3e; border-radius:6px; padding:8px 14px;
                  margin:4px 0; display:flex; justify-content:space-between; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Return Fraud Validator")
st.caption("InternVL2.5-4B MPO + LoRA — Multimodal Return Validation")

# ── Sidebar — Metadata ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Vendor Metadata")
    product  = st.text_input("Product",  value="Blue Denim Jacket")
    brand    = st.text_input("Brand",    value="Levi's")
    color    = st.text_input("Color",    value="blue")
    design   = st.text_input("Design",   value="plain denim")
    quantity = st.selectbox("Quantity",  ["1", "2", "3+"], index=0)

    st.divider()
    st.header("⚙️ Settings")
    use_lora    = st.toggle("Use LoRA weights", value=False)
    use_4bit    = st.toggle("4-bit quantization", value=True)
    use_cache   = st.toggle("Enable result cache", value=True)
    show_latency = st.toggle("Show latency breakdown", value=True)

    st.divider()
    if st.button("🗑 Clear Cache", use_container_width=True):
        from utils.cache_manager import result_cache, image_cache
        result_cache.invalidate_all()
        image_cache.clear()
        st.success("Cache cleared")

# ── Image Upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Delivery Images (Customer Return)")
    delivery_files = st.file_uploader(
        "Upload 4 delivery images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="delivery",
    )
    if delivery_files:
        cols = st.columns(min(4, len(delivery_files)))
        for i, f in enumerate(delivery_files[:4]):
            with cols[i]:
                st.image(Image.open(f), caption=f"D{i+1}", use_container_width=True)

with col2:
    st.subheader("🏭 Vendor Images (Original Product)")
    vendor_files = st.file_uploader(
        "Upload 4 vendor images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="vendor",
    )
    if vendor_files:
        cols = st.columns(min(4, len(vendor_files)))
        for i, f in enumerate(vendor_files[:4]):
            with cols[i]:
                st.image(Image.open(f), caption=f"V{i+1}", use_container_width=True)

st.divider()

# ── Validate Button ────────────────────────────────────────────────────────────
validate_btn = st.button("🚀 Validate Return", type="primary",
                          use_container_width=True,
                          disabled=not (delivery_files and vendor_files))

if validate_btn:
    metadata = {
        "product": product, "brand": brand,
        "color": color, "design": design, "quantity": quantity,
    }

    # Save uploaded files to temp paths
    def save_temp(files):
        paths = []
        for f in files[:4]:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(f.read())
            tmp.close()
            paths.append(tmp.name)
        # Pad to 4 if fewer uploaded
        while len(paths) < 4:
            paths.append(paths[-1])
        return paths

    delivery_paths = save_temp(delivery_files)
    vendor_paths   = save_temp(vendor_files)

    with st.spinner("Analyzing images..."):
        # Check result cache first
        result = None
        if use_cache:
            from utils.cache_manager import result_cache
            result = result_cache.get(delivery_paths, vendor_paths, metadata)
            if result:
                st.info("⚡ Result served from cache")

        if result is None:
            from model.internvl_lora import ReturnValidator
            validator = ReturnValidator(use_lora=use_lora, load_in_4bit=use_4bit)
            result    = validator.validate(delivery_paths, vendor_paths, metadata)

    # ── Decision Banner ────────────────────────────────────────────────────────
    accept = result.get("accept_return", "no") == "yes"
    if accept:
        st.markdown('<div class="result-yes">✅ RETURN ACCEPTED — All conditions satisfied</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-no">❌ RETURN REJECTED — One or more conditions failed</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Field Results ──────────────────────────────────────────────────────────
    res_col, json_col = st.columns([1, 1])

    with res_col:
        st.subheader("📋 Validation Fields")
        field_labels = {
            "product_match":   "Product Match",
            "design_match":    "Design Match",
            "color_match":     "Color Match",
            "quantity_is_one": "Quantity = 1",
            "is_damaged":      "Is Damaged",
            "is_used":         "Is Used",
            "accept_return":   "Accept Return",
        }
        for field, label in field_labels.items():
            val = result.get(field, "no")
            # For damage/used: "no" is good (green), "yes" is bad (red)
            if field in ("is_damaged", "is_used"):
                color_cls = "🟢" if val == "no" else "🔴"
            elif field == "accept_return":
                color_cls = "✅" if val == "yes" else "❌"
            else:
                color_cls = "🟢" if val == "yes" else "🔴"
            st.markdown(
                f'<div class="field-row"><span style="color:#cad3e0">{label}</span>'
                f'<span>{color_cls} <b>{val.upper()}</b></span></div>',
                unsafe_allow_html=True,
            )

    with json_col:
        st.subheader("📄 Raw JSON Output")
        display_result = {k: v for k, v in result.items() if not k.startswith("_")}
        st.code(json.dumps(display_result, indent=2), language="json")

    # ── Latency Metrics ────────────────────────────────────────────────────────
    if show_latency and "_latency_ms" in result:
        st.divider()
        st.subheader("⚡ Performance Metrics")
        from utils.latency_tracker import tracker
        stats = tracker.get_stats()

        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = [
            (m1, f"{result.get('_latency_ms', 0):.0f} ms", "This Request"),
            (m2, f"{stats.get('avg_total_ms', 0):.0f} ms", "Avg Total"),
            (m3, f"{stats.get('avg_preprocessing_ms', 0):.0f} ms", "Avg Preprocessing"),
            (m4, f"{stats.get('avg_inference_ms', 0):.0f} ms", "Avg Inference"),
            (m5, f"{stats.get('cached_requests', 0)}", "Cache Hits"),
        ]
        for col, val, lbl in metrics:
            with col:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        # Latency breakdown bar
        if stats.get("last_10"):
            import pandas as pd
            import plotly.graph_objects as go
            last = stats["last_10"][-1]
            fig = go.Figure(go.Bar(
                x=["Preprocessing", "Inference", "Post-processing"],
                y=[last["preprocessing_ms"], last["inference_ms"], last["postprocessing_ms"]],
                marker_color=["#3498db", "#e74c3c", "#27ae60"],
                text=[f"{v:.0f}ms" for v in [last["preprocessing_ms"],
                      last["inference_ms"], last["postprocessing_ms"]]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Latency Breakdown (Last Request)",
                template="plotly_dark",
                paper_bgcolor="#0d1b2a",
                plot_bgcolor="#162b3e",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Cleanup temp files
    import os
    for p in delivery_paths + vendor_paths:
        try: os.unlink(p)
        except: pass

# ── Cache Stats (always visible) ──────────────────────────────────────────────
with st.expander("📊 Cache & Performance Stats"):
    from utils.cache_manager import image_cache, result_cache
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Image Cache**")
        st.json(image_cache.stats())
    with c2:
        st.markdown("**Result Cache**")
        st.json(result_cache.stats())
