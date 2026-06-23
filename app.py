import json
import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.demo.gatekeeper_service import GatekeeperService
from src.demo.wear_service import WearService
from src.demo.visualization import draw_boxes, result_card_html, metric_card_html
from src.demo.utils import (
    GATEKEEPER_METRICS,
    WEAR_METRICS_V2,
    load_image,
)

st.set_page_config(
    page_title="Mining Truck Tyre Inspection",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .stage-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .stage-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #1a1a2e;
    }
    .status-pass {
        color: #28a745;
        font-weight: 600;
    }
    .status-fail {
        color: #dc3545;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] {
        background: #f0f2f6;
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


def process_image(image):
    gatekeeper = init_gatekeeper()
    wear = init_wear()

    stage1_col, stage2_col = st.columns(2)

    with st.spinner("Stage 1: Tyre Verification in progress..."):
        gate_dets = gatekeeper.predict(image)
        gate_decision = gatekeeper.decide(gate_dets)

    with stage1_col:
        st.markdown('<div class="stage-box">', unsafe_allow_html=True)
        st.markdown('<div class="stage-title">🔍 Stage 1: Tyre Verification</div>',
                    unsafe_allow_html=True)

        if gate_decision["is_tire"]:
            st.markdown(f'<div class="status-pass">✅ Tyre Detected</div>',
                        unsafe_allow_html=True)
            st.metric("Confidence", f"{gate_decision['confidence']:.2%}")
        else:
            st.markdown(f'<div class="status-fail">❌ Non-Tyre Detected</div>',
                        unsafe_allow_html=True)
            st.metric("Reason", gate_decision.get("reason", ""))

        st.markdown('</div>', unsafe_allow_html=True)

    vis_img = image.copy()
    if gate_dets:
        vis_img = draw_boxes(vis_img, gate_dets)

    with stage2_col:
        st.markdown('<div class="stage-box">', unsafe_allow_html=True)
        st.markdown('<div class="stage-title">🛞 Stage 2: Wear Detection</div>',
                    unsafe_allow_html=True)

        if gate_decision["is_tire"]:
            with st.spinner("Running wear analysis..."):
                boxes, scores, labels = wear.predict(image)
                wear_result = wear.classify_output(boxes, scores, labels)

            final_class = wear_result["final_class"]
            confidence = max(d["confidence"] for d in wear_result["detections"]) if wear_result["detections"] else 0.0

            st.markdown(f'<div class="stage-title">Result: {final_class}</div>',
                        unsafe_allow_html=True)

            wear_vis = image.copy()
            wear_vis = draw_boxes(wear_vis, wear_result["detections"])
        else:
            final_class = "Non-Tire"
            confidence = 0.0
            wear_result = None
            wear_vis = None
            st.markdown(
                '<div style="color: #dc3545; font-weight: 600;">⛔ Inspection stopped at Stage 1</div>',
                unsafe_allow_html=True)
            st.caption("Wear detection was not executed because the image is not a tyre.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Final Result")
    st.markdown(result_card_html(final_class, confidence, wear_result["reason"] if wear_result else "Non-Tyre image"),
                unsafe_allow_html=True)

    st.markdown("### 🖼️ Detection Visualization")
    vis_tabs = st.tabs(["Gatekeeper", "Wear Detection"])
    with vis_tabs[0]:
        st.image(vis_img, caption="Gatekeeper Detections", use_column_width=True)
    with vis_tabs[1]:
        if wear_vis is not None:
            st.image(wear_vis, caption="Wear Detection", use_column_width=True)
        else:
            st.info("No wear detection visualization available (non-tyre image).")

    return {
        "gatekeeper_result": "Tyre" if gate_decision["is_tire"] else "Non-Tyre",
        "gatekeeper_confidence": gate_decision.get("confidence", 0.0),
        "gatekeeper_reason": gate_decision.get("reason", ""),
        "wear_result": wear_result,
        "final_class": final_class,
        "confidence": confidence,
        "gatekeeper_detections": gate_dets,
    }


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
        st.markdown("## ⚙️ Debug")
        show_debug = st.checkbox("Show Technical Details", value=False)

        st.markdown("---")
        st.markdown("## 📈 Performance")
        with st.expander("Gatekeeper Performance", expanded=True):
            gk = GATEKEEPER_METRICS
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
            "Mining Truck Tyre Inspection System v2.0"
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
    st.image(image, caption="Input Image", use_column_width=True)

    scan_col1, scan_col2 = st.columns([1, 3])
    with scan_col1:
        scan_clicked = st.button("🔍 Start Scan", type="primary", use_container_width=True)
    with scan_col2:
        st.caption("Click to begin tyre verification and wear detection pipeline")

    if scan_clicked:
        st.session_state["run_scan"] = True

    if st.session_state.get("run_scan"):
        try:
            result = process_image(image)
        except FileNotFoundError as e:
            st.error(f"🚨 Missing checkpoint: {e}")
            st.warning("Please ensure all model checkpoints are available.")
            st.stop()
        except Exception as e:
            st.error(f"🚨 Inference failed: {e}")
            st.stop()
    else:
        st.info("👆 Press **Start Scan** to run the inspection pipeline.")
        st.stop()

    if show_debug:
        st.markdown("### 🔧 Technical Details")
        debug_tabs = st.tabs(["Gatekeeper Raw", "Wear Raw", "Full JSON"])
        with debug_tabs[0]:
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
        with debug_tabs[1]:
            if result["wear_result"]:
                st.json({
                    "final_class": result["wear_result"]["final_class"],
                    "reason": result["wear_result"]["reason"],
                    "detections": [
                        {
                            "class": d["class"],
                            "confidence": round(d["confidence"], 4),
                        }
                        for d in result["wear_result"]["detections"]
                    ],
                })
            else:
                st.info("Wear detection was not executed.")
        with debug_tabs[2]:
            st.json({
                "gatekeeper_result": result["gatekeeper_result"],
                "wear_result": result["wear_result"]["final_class"] if result["wear_result"] else None,
                "final_class": result["final_class"],
                "confidence": round(result["confidence"], 4),
            })

    export_data = {
        "gatekeeper_result": result["gatekeeper_result"],
        "wear_result": result["wear_result"]["final_class"] if result["wear_result"] else None,
        "final_class": result["final_class"],
        "confidence": round(result["confidence"], 4),
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💾 Export")
    json_str = json.dumps(export_data, indent=2)
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
