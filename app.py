import json
import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.demo import config
from src.demo.gatekeeper_service import GatekeeperService
from src.demo.wear_service import WearService
from src.demo.visualization import draw_boxes, result_card_html, metric_card_html
from src.demo.utils import (
    GATEKEEPER_METRICS,
    WEAR_METRICS_V2,
    GATEKEEPER_CKPT,
    GATEKEEPER_V3_CKPT,
    WEAR_V2_CKPT,
    load_image,
    check_checkpoint,
)

st.set_page_config(
    page_title="Mining Truck Tyre Inspection",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Startup validation: verify checkpoints exist
ckpt_gk = config.GATEKEEPER_CHECKPOINTS[config.GATEKEEPER_MODEL]
ckpt_wear = str(WEAR_V2_CKPT)
missing = []
if not os.path.isfile(ckpt_gk):
    missing.append(f"Gatekeeper checkpoint: {ckpt_gk}")
if not os.path.isfile(ckpt_wear):
    missing.append(f"Wear checkpoint: {ckpt_wear}")
if missing:
    st.error(f"Missing checkpoint(s):\n" + "\n".join(missing))
    st.stop()

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0;
        font-size: 1rem;
        opacity: 0.8;
    }
    div[data-testid="stImage"] {
        border: 2px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stImage"] img {
        max-height: 450px !important;
        object-fit: contain !important;
    }
    .result-metric {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .result-metric .label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-metric .value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #212529;
        margin-top: 2px;
    }
    .result-metric .value.good { color: #28a745; }
    .result-metric .value.bad { color: #dc3545; }
    .result-metric .value.info { color: #17a2b8; }
    .result-metric .value.pass { color: #28a745; }
    .result-metric .value.fail { color: #dc3545; }
    div[data-testid="stSidebar"] {
        background: #f0f2f6;
    }
    .download-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_gatekeeper():
    return GatekeeperService()


@st.cache_resource
def init_wear():
    return WearService()


EXAMPLE_DIR = Path("datasets/annotated")
EXAMPLE_GOOD_DIR = EXAMPLE_DIR / "good"
EXAMPLE_BAD_DIR = EXAMPLE_DIR / "bad"
EXAMPLE_NEG_DIR = EXAMPLE_DIR / "negative"

EXAMPLE_GOOD = sorted(EXAMPLE_GOOD_DIR.glob("*.jpg"))[:3]
EXAMPLE_BAD = sorted(EXAMPLE_BAD_DIR.glob("*.jpg"))[:3]
EXAMPLE_NEG = sorted(EXAMPLE_NEG_DIR.glob("*.jpg"))[:3]


def load_example_image(category):
    if category == "Good Tyre":
        files = EXAMPLE_GOOD
    elif category == "Bad Tyre":
        files = EXAMPLE_BAD
    else:
        files = EXAMPLE_NEG
    if files:
        return Image.open(files[0]).convert("RGB")
    return None


def resize_display_image(image, max_width=800, max_height=450):
    w, h = image.size
    ratio = min(max_width / w, max_height / h, 1.0)
    if ratio < 1.0:
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image


def process_image(image):
    gatekeeper = init_gatekeeper()
    wear = init_wear()

    gate_dets = gatekeeper.predict(image)
    gate_decision = gatekeeper.decide(gate_dets)

    gate_vis = image.copy()
    if gate_dets:
        gate_vis = draw_boxes(gate_vis, gate_dets)

    wear_result = None
    wear_vis = None
    final_class = "Non-Tire"
    confidence = 0.0

    if gate_decision["is_tire"]:
        boxes, scores, labels = wear.predict(image)
        wear_result = wear.classify_output(boxes, scores, labels)
        final_class = wear_result["final_class"]
        confidence = max(d["confidence"] for d in wear_result["detections"]) if wear_result["detections"] else 0.0
        wear_vis = image.copy()
        wear_vis = draw_boxes(wear_vis, wear_result["detections"])

    return {
        "gatekeeper_result": "Tyre" if gate_decision["is_tire"] else "Non-Tyre",
        "gatekeeper_confidence": gate_decision.get("confidence", 0.0),
        "gatekeeper_reason": gate_decision.get("reason", ""),
        "gatekeeper_detections": gate_dets,
        "gatekeeper_vis": gate_vis,
        "wear_result": wear_result,
        "wear_vis": wear_vis,
        "final_class": final_class,
        "confidence": confidence,
    }


def image_to_bytes(img, format="PNG"):
    buf = BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def result_metric_html(label, value, css_class=""):
    return f"""
    <div class="result-metric">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
    </div>
    """


def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚛 Mining Truck Tyre Inspection System</h1>
        <p>AI-based Tyre Verification and Wear Detection</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 📸 Example Images")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Good\nTyre", use_container_width=True):
                img = load_example_image("Good Tyre")
                if img:
                    st.session_state["uploaded_image"] = img
                    st.session_state["image_source"] = "example_good"
                    st.session_state["run_scan"] = False
                    st.rerun()
        with col2:
            if st.button("❌ Bad\nTyre", use_container_width=True):
                img = load_example_image("Bad Tyre")
                if img:
                    st.session_state["uploaded_image"] = img
                    st.session_state["image_source"] = "example_bad"
                    st.session_state["run_scan"] = False
                    st.rerun()
        with col3:
            if st.button("🚫 Non-\nTyre", use_container_width=True):
                img = load_example_image("Non Tyre")
                if img:
                    st.session_state["uploaded_image"] = img
                    st.session_state["image_source"] = "example_neg"
                    st.session_state["run_scan"] = False
                    st.rerun()

        st.markdown("---")
        st.markdown("## 🤖 Model Information")
        st.markdown(
            f"**Gatekeeper:**<br>V3 (Production)"
            f"<br><span style='font-size:0.8rem;color:#666'>({config.GATEKEEPER_MODEL.upper()})</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**Wear Detection:**<br>Current (Faster R-CNN V1)", unsafe_allow_html=True)

        show_paths = st.checkbox("Show Model Paths", value=False)
        if show_paths:
            st.code(f"Gatekeeper: {config.GATEKEEPER_CHECKPOINTS[config.GATEKEEPER_MODEL]}")
            st.code(f"Wear: {WEAR_V2_CKPT}")

        st.markdown("---")
        st.markdown("## 📈 Performance")
        with st.expander("Gatekeeper Performance", expanded=True):
            gk = GATEKEEPER_METRICS
            st.markdown(metric_card_html("Accuracy", gk.get("Accuracy", "N/A")),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("Precision", gk["Precision"]),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("Recall", gk["Recall"]),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("F1 Score", gk["F1"]),
                        unsafe_allow_html=True)

        with st.expander("Wear Detection Performance", expanded=True):
            wd = WEAR_METRICS_V2
            st.markdown(metric_card_html("Overall Accuracy", wd["OverallAccuracy"]),
                        unsafe_allow_html=True)
            st.markdown("**Tire**")
            st.markdown(metric_card_html("Precision", wd["Tire"]["Precision"]),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("Recall", wd["Tire"]["Recall"]),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("F1", wd["Tire"]["F1"]),
                        unsafe_allow_html=True)
            st.markdown("**Non-Tire**")
            st.markdown(metric_card_html("Precision", wd["Non-Tire"]["Precision"]),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("Recall", wd["Non-Tire"]["Recall"]),
                        unsafe_allow_html=True)
            st.markdown(metric_card_html("F1", wd["Non-Tire"]["F1"]),
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.caption(
            "Mining Truck Tyre Inspection System v3.0"
        )

    uploaded_file = st.file_uploader(
        "Upload a tyre image",
        type=["jpg", "jpeg", "png"],
        help="Drag & drop or click to upload a mining truck tyre image",
    )

    if uploaded_file is not None:
        try:
            image = load_image(uploaded_file)
            st.session_state["uploaded_image"] = image
            st.session_state["image_source"] = "upload"
            st.session_state["run_scan"] = False
        except Exception as e:
            st.error(f"Invalid image file: {e}")
            st.stop()

    if "uploaded_image" not in st.session_state:
        st.info(
            "👆 Upload an image or click an example button in the sidebar to begin.")
        st.markdown("### 📖 How It Works")
        with st.expander("Pipeline Explanation", expanded=True):
            st.markdown("""
            **1. Tyre Verification (Gatekeeper)**  
            A Faster R-CNN model checks if the image contains a mining truck tyre.  
            If no tyre is detected, the pipeline stops.

            **2. Wear Detection**  
            A second Faster R-CNN model inspects the tyre for cuts, wear, and damage.  
            Detections are classified as: **Cut**, **Tire**, or **Non-Tire**.

            **3. Final Decision**  
            - **Cut** detected → **Bad-Tire** ❌  
            - **Tire** only → **Good-Tire** ✅  
            - **Non-Tire** → **Non-Tire** ℹ️
            """)
        st.stop()

    image = st.session_state["uploaded_image"]

    scan_col1, scan_col2 = st.columns([1, 3])
    with scan_col1:
        scan_clicked = st.button("🔍 Start Scan", type="primary", use_container_width=True)
    with scan_col2:
        st.caption("Click to begin tyre verification and wear detection pipeline")

    if scan_clicked:
        st.session_state["run_scan"] = True

    if not st.session_state.get("run_scan"):
        disp_img = resize_display_image(image)
        st.image(disp_img, caption="Input Image", use_container_width=True)
        st.info("👆 Press **Start Scan** to run the inspection pipeline.")
        st.stop()

    try:
        result = process_image(image)
    except FileNotFoundError as e:
        st.error(f"🚨 Missing checkpoint: {e}")
        st.warning("Please ensure all model checkpoints are available.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Inference failed: {e}")
        st.stop()

    # ===== SIDE-BY-SIDE IMAGES =====
    left_col, right_col = st.columns(2)

    with left_col:
        disp_original = resize_display_image(image)
        st.image(disp_original, caption="Uploaded Image", use_container_width=True)

    with right_col:
        detection_img = result["wear_vis"] if result.get("wear_vis") is not None else result["gatekeeper_vis"]
        if detection_img is not None:
            disp_detection = resize_display_image(detection_img)
            st.image(disp_detection, caption="Detection Result", use_container_width=True)

    # ===== RESULT CARDS =====
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    final_class = result["final_class"]
    confidence = result["confidence"]
    gatekeeper_passed = result["gatekeeper_result"] == "Tyre"

    if final_class == "Good-Tire":
        status_text = "✅ GOOD"
        status_css = "good"
    elif final_class == "Bad-Tire":
        status_text = "❌ BAD"
        status_css = "bad"
    else:
        status_text = "ℹ️ NON-TYRE"
        status_css = "info"

    gatekeeper_text = "✅ PASSED" if gatekeeper_passed else "❌ REJECTED"
    gatekeeper_css = "pass" if gatekeeper_passed else "fail"

    wear_detail = result["wear_result"]["reason"] if result["wear_result"] else "Not inspected (non-tyre)"

    cols = st.columns(4)
    with cols[0]:
        st.markdown(result_metric_html("Tyre Status", status_text, status_css),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(result_metric_html("Confidence", f"{confidence:.1%}"),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(result_metric_html("Gatekeeper", gatekeeper_text, gatekeeper_css),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(result_metric_html("Wear Detail", wear_detail),
                    unsafe_allow_html=True)

    # ===== COLLAPSIBLE TECHNICAL DETAILS =====
    with st.expander("Technical Details", expanded=False):
        det_tabs = st.tabs(["Gatekeeper Raw", "Wear Raw", "Full JSON"])
        with det_tabs[0]:
            st.json({
                "detections": [
                    {
                        "class": d["class"],
                        "confidence": round(d["confidence"], 4),
                        "bbox": [round(b, 2) for b in d["bbox"]],
                    }
                    for d in result["gatekeeper_detections"]
                ],
                "decision": {
                    "is_tire": result["gatekeeper_result"] == "Tyre",
                    "confidence": round(result["gatekeeper_confidence"], 4),
                    "reason": result["gatekeeper_reason"],
                },
            })
        with det_tabs[1]:
            if result["wear_result"]:
                st.json({
                    "final_class": result["wear_result"]["final_class"],
                    "reason": result["wear_result"]["reason"],
                    "detections": [
                        {
                            "class": d["class"],
                            "confidence": round(d["confidence"], 4),
                            "bbox": [round(b, 2) for b in d["bbox"]],
                        }
                        for d in result["wear_result"]["detections"]
                    ],
                })
            else:
                st.info("Wear detection was not executed.")
        with det_tabs[2]:
            st.json({
                "gatekeeper_result": result["gatekeeper_result"],
                "wear_result": result["wear_result"]["final_class"] if result["wear_result"] else None,
                "final_class": result["final_class"],
                "confidence": round(result["confidence"], 4),
            })

    # ===== DOWNLOAD BUTTONS =====
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)

    annotated = result.get("wear_vis") or result.get("gatekeeper_vis")
    if annotated is not None:
        img_bytes = image_to_bytes(annotated)
        with dl_col1:
            st.download_button(
                label="📷 Download Annotated Image",
                data=img_bytes,
                file_name="detection_result.png",
                mime="image/png",
                use_container_width=True,
            )

    export_data = {
        "gatekeeper_result": result["gatekeeper_result"],
        "wear_result": result["wear_result"]["final_class"] if result["wear_result"] else None,
        "final_class": result["final_class"],
        "confidence": round(result["confidence"], 4),
        "gatekeeper_confidence": round(result["gatekeeper_confidence"], 4),
        "gatekeeper_reason": result["gatekeeper_reason"],
    }
    json_str = json.dumps(export_data, indent=2)
    with dl_col2:
        st.download_button(
            label="📋 Download Inspection JSON",
            data=json_str,
            file_name="inspection_result.json",
            mime="application/json",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== SIDEBAR EXPORT =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💾 Export")
    st.sidebar.download_button(
        label="Download inspection_result.json",
        data=json_str,
        file_name="inspection_result.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 📖 How It Works")
    with st.expander("Pipeline Explanation", expanded=True):
        st.markdown("""
        **1. Tyre Verification**  
        A Faster R-CNN (MobileNetV3-Large-FPN) model detects whether the image contains a mining truck tyre.
        If a tyre is not detected with sufficient confidence, the pipeline stops and reports "Non-Tyre".

        **2. Wear Detection**  
        A second Faster R-CNN model inspects detected tyres for surface damage (cuts/wear).
        It outputs three detection classes: **Tire** (good surface), **Cut** (damage), **Non-Tire**.

        **3. Final Decision**  
        - **Cut** detected → **Bad-Tire** ❌ → Requires maintenance  
        - **Tire** only → **Good-Tire** ✅ → Passes inspection  
        - **Non-Tire** → **Non-Tire** ℹ️ → Not a tyre image
        """)


if __name__ == "__main__":
    main()
