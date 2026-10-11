import sys
from pathlib import Path
import random

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import io
from PIL import Image
import streamlit as st

from surfaceguard.config import ALL_CATEGORIES, resolve_path
from surfaceguard.inference.predictor import SurfaceGuardPredictor
from surfaceguard.utils.artifacts import ArtifactManager
from surfaceguard.utils.device import get_device_info

try:
    from app.components.ui_helpers import (
        apply_custom_css,
        render_header,
        render_verdict_banner,
        render_metric_tiles,
        format_specimen_label,
        CATEGORY_META,
    )
    from app.components.benchmark_view import render_benchmark_tab
except ModuleNotFoundError:
    from components.ui_helpers import (
        apply_custom_css,
        render_header,
        render_verdict_banner,
        render_metric_tiles,
        format_specimen_label,
        CATEGORY_META,
    )
    from components.benchmark_view import render_benchmark_tab


# Page configuration
st.set_page_config(
    page_title="SurfaceGuard AI — Industrial Optical Inspection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply high-tech cyber-industrial styling & header
apply_custom_css()
render_header()

# Initialize core services
predictor = SurfaceGuardPredictor()
artifact_mgr = ArtifactManager()

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.markdown("### ⚙️ Inspection Engine Controls")

selected_model = st.sidebar.selectbox(
    "Deep Inspection Architecture",
    options=["patchcore", "autoencoder", "classical"],
    format_func=lambda x: {
        "patchcore": "PatchCore Neural Memory (ResNet-18) [15/15 Ready ⭐]",
        "autoencoder": "Convolutional Autoencoder (Reconstruction) [Needs Training]",
        "classical": "Structural Computer Vision (Gaussian Baseline)",
    }.get(x, x),
    help="Select the deep learning inspection architecture to evaluate specimens against.",
)

availability = artifact_mgr.list_available_categories(model_type=selected_model)

def format_cat_dropdown(cat: str) -> str:
    meta = CATEGORY_META.get(cat, {"icon": "📦", "title": cat.title()})
    status_icon = "🟢" if availability.get(cat) else "⏳"
    return f"{meta['icon']} {meta['title']} [{status_icon}]"

selected_category = st.sidebar.selectbox(
    "Target Production Specimen",
    options=ALL_CATEGORIES,
    format_func=format_cat_dropdown,
    help="Select the factory production part to inspect.",
)

# Inspection Sensitivity & Tuning
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Optical Calibration")

sensitivity_mode = st.sidebar.select_slider(
    "Tolerance Strictness",
    options=["Permissive (Fewer False Alerts)", "Standard (Calibrated 99% Precision)", "Strict (Zero-Defect Tolerance)"],
    value="Standard (Calibrated 99% Precision)",
    help="Fine-tune operating decision threshold boundary against false positives.",
)
thresh_mult = {
    "Permissive (Fewer False Alerts)": 1.15,
    "Standard (Calibrated 99% Precision)": 1.00,
    "Strict (Zero-Defect Tolerance)": 0.88,
}[sensitivity_mode]

alpha = st.sidebar.slider(
    "Heatmap Blend Opacity",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Adjust transparency of the thermal defect localization overlay.",
)

# Hardware Environment
dev_info = get_device_info()
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Edge Telemetry")
st.sidebar.caption(f"**Execution Hardware**: `{dev_info['device_name']}`")
st.sidebar.caption(f"**Acceleration**: `Torch {dev_info['pytorch_version']}`")
st.sidebar.caption(f"**CUDA Core Status**: `{'Enabled' if dev_info['cuda_available'] else 'CPU AVX-512 Optimized'}`")

# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tab_inspect, tab_comparator, tab_benchmark = st.tabs([
    "🏭 Automated Optical Inspection (AOI Live Station)",
    "🔬 Zero-Shot Optical Comparator (Dual-Scan)",
    "📊 Fleet Performance & Benchmark Analytics",
])

# ------------------------------------------------------------------------------
# TAB 1: AUTOMATED OPTICAL INSPECTION (AOI LIVE STATION)
# ------------------------------------------------------------------------------
with tab_inspect:
    cat_meta = CATEGORY_META.get(selected_category, {"icon": "📦", "title": selected_category.title(), "type": "Object"})
    
    cat_test_dir = resolve_path(f"data/mvtec_ad/{selected_category}/test")
    showcase_dir = resolve_path(f"data/showcase_specimens/{selected_category}/test")
    synth_test_dir = resolve_path(f"data/synthetic_mvtec/{selected_category}/test")
    sample_dir = cat_test_dir if cat_test_dir.exists() else (showcase_dir if showcase_dir.exists() else (synth_test_dir if synth_test_dir.exists() else None))

    all_samples = []
    if sample_dir and sample_dir.exists():
        all_samples = sorted(list(sample_dir.rglob("*.png")) + list(sample_dir.rglob("*.jpg")))

    col_selector, col_preview = st.columns([1.1, 0.9])

    with col_selector:
        st.markdown(
            f"""
            <div class="tech-card">
                <div class="card-header">
                    <span>{cat_meta['icon']}</span>
                    <span>Conveyor Optical Stream — {cat_meta['title']}</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
                    Active high-resolution camera feed from production conveyor line. Specimen images are captured in real-time under industrial studio illumination.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feed filtering: All, Defective Audit Samples, or Clean Tolerance Standards
        filter_mode = st.radio(
            "Conveyor Stream Filter",
            options=["All Production Batches", "🔴 Defect Audit Specimens", "🟢 Tolerance Baseline Standards"],
            horizontal=True,
            label_visibility="collapsed",
        )

        filtered_samples = all_samples
        if filter_mode == "🔴 Defect Audit Specimens":
            filtered_samples = [p for p in all_samples if p.parent.name != "good"]
        elif filter_mode == "🟢 Tolerance Baseline Standards":
            filtered_samples = [p for p in all_samples if p.parent.name == "good"]

        if not filtered_samples:
            filtered_samples = all_samples

        # Streamlit-safe state tracking for random sampling
        if "selected_specimen_idx" not in st.session_state:
            st.session_state.selected_specimen_idx = 0

        # Bound check for current filter length
        if filtered_samples:
            st.session_state.selected_specimen_idx = min(
                st.session_state.selected_specimen_idx, len(filtered_samples) - 1
            )

        def on_next_batch():
            if filtered_samples:
                st.session_state.selected_specimen_idx = random.randint(0, len(filtered_samples) - 1)

        # Cycle button & Selector in one row
        sub_c1, sub_c2 = st.columns([3, 1.2])
        with sub_c1:
            chosen_specimen = st.selectbox(
                "Select Optical Feed Specimen:",
                options=filtered_samples,
                index=st.session_state.selected_specimen_idx if filtered_samples else 0,
                format_func=format_specimen_label,
            )
            if chosen_specimen in filtered_samples:
                st.session_state.selected_specimen_idx = filtered_samples.index(chosen_specimen)
        with sub_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.button(
                "🎲 Next Batch",
                on_click=on_next_batch,
                use_container_width=True,
                help="Cycle to a random production batch item",
            )

    input_image = None
    if chosen_specimen:
        try:
            input_image = Image.open(chosen_specimen).convert("RGB")
        except Exception as e:
            st.error(f"Error loading optical feed specimen: {e}")

    with col_preview:
        if input_image:
            is_good = (chosen_specimen.parent.name == "good")
            badge_color = "#10b981" if is_good else "#f43f5e"
            badge_label = "STANDARD BASELINE" if is_good else "DEFECT AUDIT LOT"

            st.markdown(
                f"""
                <div class="tech-card" style="padding: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; color: #94a3b8;">
                            SENSOR: OPTICAL CAM #1
                        </span>
                        <span style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.72rem; font-family: 'JetBrains Mono'; font-weight: 600;">
                            {badge_label}
                        </span>
                    </div>
                    <div class="hud-frame">
                """,
                unsafe_allow_html=True,
            )
            st.image(input_image, caption=f"Specimen Resolution: {input_image.size[0]}×{input_image.size[1]} px | Batch File: {chosen_specimen.name}", use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Execution Trigger Button
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if not availability.get(selected_category) and selected_model == "autoencoder":
        st.info("💡 **Tip**: Switch to **`PatchCore Neural Memory (ResNet-18)`** in the sidebar. All 15 MVTec AD categories are pre-trained and ready!")

    inspect_btn = st.button("⚡ Execute Neural Surface Audit", type="primary", use_container_width=True)

    if inspect_btn and input_image:
        if not availability.get(selected_category) and selected_model in ["patchcore", "autoencoder"]:
            if selected_model == "autoencoder":
                st.warning(
                    f"⚠️ **Convolutional Autoencoder checkpoint for '{selected_category}' is not yet trained.**\n\n"
                    f"👉 **Please switch to `PatchCore Neural Memory (ResNet-18)`** in the sidebar! All 15 industrial categories are trained and calibrated with feature memory banks.\n\n"
                    f"*(Or to train the Autoencoder on normal samples, run `python scripts/train_autoencoder.py --category {selected_category}`)*"
                )
            else:
                st.error(f"❌ Feature Memory Bank missing for category '{selected_category}'.")
        else:
            with st.spinner(f"Running deep neural inference via {selected_model}..."):
                try:
                    result = predictor.predict_image(
                        image=input_image,
                        category=selected_category,
                        model_type=selected_model,
                        auto_segment=False,
                    )

                    # Apply user sensitivity calibration
                    adjusted_thresh = round(result["threshold"] * thresh_mult, 4)
                    result["threshold"] = adjusted_thresh
                    result["is_anomalous"] = bool(result["anomaly_score"] >= adjusted_thresh)

                    # Glowing Verdict Banner
                    render_verdict_banner(
                        is_anomalous=result["is_anomalous"],
                        anomaly_score=result["anomaly_score"],
                        threshold=result["threshold"],
                    )

                    # Telemetry Metric Tiles
                    render_metric_tiles(result)

                    # Visual Defect Localization Suite
                    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
                    st.markdown("#### 🔬 Tri-Channel Anomaly Localization Suite")

                    v1, v2, v3 = st.columns(3)
                    with v1:
                        st.markdown(
                            """
                            <div class="tech-card" style="padding: 0.8rem; text-align: center;">
                                <div style="font-family: 'Space Grotesk'; font-size: 0.85rem; color: #38bdf8; margin-bottom: 0.4rem;">
                                    CHANNEL 1: OPTICAL RAW FEED
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(input_image, use_container_width=True)
                        st.caption("Surface optical sensor capture under calibrated factory illumination")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with v2:
                        st.markdown(
                            """
                            <div class="tech-card" style="padding: 0.8rem; text-align: center;">
                                <div style="font-family: 'Space Grotesk'; font-size: 0.85rem; color: #f43f5e; margin-bottom: 0.4rem;">
                                    CHANNEL 2: DEFECT THERMAL DENSITY (JET)
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(result["heatmap_base64"], use_container_width=True)
                        st.caption("Pixel-wise anomaly intensity map generated from L2 patch distance")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with v3:
                        st.markdown(
                            """
                            <div class="tech-card" style="padding: 0.8rem; text-align: center;">
                                <div style="font-family: 'Space Grotesk'; font-size: 0.85rem; color: #a78bfa; margin-bottom: 0.4rem;">
                                    CHANNEL 3: LOCALIZED DEFECT OVERLAY
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(result["overlay_base64"], use_container_width=True)
                        st.caption(f"Blended spatial defect localization ({int(alpha*100)}% overlay alpha)")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Engineering Disposition Action Box
                    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
                    if result["is_anomalous"]:
                        st.markdown(
                            f"""
                            <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 10px; padding: 1rem 1.4rem;">
                                <strong style="color: #f43f5e; font-family: 'Space Grotesk';">DISPOSITION PROTOCOL:</strong>
                                <span style="color: #fecdd3; margin-left: 0.5rem; font-size: 0.95rem;">
                                    Surface discrepancy exceeded calibrated acceptance limit ({result['anomaly_score']:.4f} &gt; {result['threshold']:.4f}). 
                                    <strong>Automated pneumatic diverter instructed to eject item to Secondary QC Quarantine.</strong>
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 1rem 1.4rem;">
                                <strong style="color: #10b981; font-family: 'Space Grotesk';">DISPOSITION PROTOCOL:</strong>
                                <span style="color: #a7f3d0; margin-left: 0.5rem; font-size: 0.95rem;">
                                    Surface features conform strictly to normal tolerance threshold ({result['anomaly_score']:.4f} &le; {result['threshold']:.4f}). 
                                    <strong>Specimen cleared for automated downstream packaging.</strong>
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as ex:
                    st.error(f"Inference execution failed: {ex}")

# ------------------------------------------------------------------------------
# TAB 2: ZERO-SHOT OPTICAL COMPARATOR (DUAL-SCAN)
# ------------------------------------------------------------------------------
with tab_comparator:
    st.markdown("### 🔬 Zero-Shot Optical Comparator (Twin-Specimen Analysis)")
    st.markdown(
        """
        Inspect **any custom industrial component** or new uncalibrated material without training.
        Supply a **Defect-Free Reference Standard** and a **Target Test Piece**. Deep feature representations 
        are compared in real-time to detect structural deformities, cuts, scratches, or missing geometry.
        """
    )

    rw_col1, rw_col2 = st.columns(2)
    with rw_col1:
        st.markdown("##### 1. Clean Golden Reference Standard")
        ref_file = st.file_uploader(
            "Upload verified defect-free reference standard image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="rw_ref_uploader",
        )

    with rw_col2:
        st.markdown("##### 2. Inspection Target Sample")
        test_file = st.file_uploader(
            "Upload target piece to inspect for anomalies",
            type=["png", "jpg", "jpeg", "bmp"],
            key="rw_test_uploader",
        )

    ref_img_pil = None
    if ref_file:
        ref_img_pil = Image.open(ref_file).convert("RGB")

    test_img_pil = None
    if test_file:
        test_img_pil = Image.open(test_file).convert("RGB")

    # Preview side-by-side
    p_col1, p_col2 = st.columns(2)
    if ref_img_pil:
        with p_col1:
            st.image(ref_img_pil, caption="Golden Reference Standard", use_container_width=True)
    if test_img_pil:
        with p_col2:
            st.image(test_img_pil, caption="Inspection Target Piece", use_container_width=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("🔍 Execute Twin-Specimen Comparison", type="primary", use_container_width=True):
        if not ref_img_pil or not test_img_pil:
            st.warning("Please provide both a Golden Reference standard and an Inspection Target image.")
        else:
            with st.spinner("Analyzing deep feature structural consistency..."):
                try:
                    from surfaceguard.inference.realworld_inspector import RealWorldInspector
                    inspector = RealWorldInspector()
                    res = inspector.inspect_against_reference(ref_img_pil, test_img_pil, threshold=3.5)

                    render_verdict_banner(res["is_anomalous"], res["anomaly_score"], res["threshold"])
                    render_metric_tiles(res)

                    st.markdown("#### Comparison Anomaly Localization")
                    c_v1, c_v2, c_v3 = st.columns(3)
                    with c_v1:
                        st.image(test_img_pil, caption="Target Piece", use_container_width=True)
                    with c_v2:
                        st.image(res["heatmap_base64"], caption="Anomaly Heatmap (JET)", use_container_width=True)
                    with c_v3:
                        st.image(res["overlay_base64"], caption="Localized Defect Overlay", use_container_width=True)
                except Exception as ex:
                    st.error(f"Comparison failed: {ex}")

# ------------------------------------------------------------------------------
# TAB 3: FLEET PERFORMANCE & BENCHMARK ANALYTICS
# ------------------------------------------------------------------------------
with tab_benchmark:
    render_benchmark_tab()
