from pathlib import Path
import pandas as pd
import streamlit as st
from surfaceguard.config import resolve_path, ALL_CATEGORIES, OBJECT_CATEGORIES, TEXTURE_CATEGORIES
from surfaceguard.utils.artifacts import ArtifactManager


def render_benchmark_tab():
    st.markdown("### 📊 Fleet Performance & Multi-Category Benchmark Analytics")
    st.markdown(
        """
        Comprehensive empirical evaluation across all 15 industrial categories.
        Unsupervised one-class inspection models trained strictly on defect-free reference specimens.
        """
    )

    mgr = ArtifactManager()
    patchcore_status = mgr.list_available_categories(model_type="patchcore")
    ae_status = mgr.list_available_categories(model_type="autoencoder")

    # Benchmark CSV if available
    summary_csv = resolve_path("results/reports/benchmark_summary.csv")
    if summary_csv.exists():
        try:
            results_df = pd.read_csv(summary_csv)

            # High level metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                val = results_df['Image AUROC'].mean() if "Image AUROC" in results_df else 0
                st.metric("Macro Image AUROC", f"{val * 100:.1f}%", help="Area under the ROC curve for anomaly classification")
            with col2:
                val_pix = results_df['Pixel AUROC'].mean() if "Pixel AUROC" in results_df else 0
                st.metric("Macro Pixel AUROC", f"{val_pix * 100:.1f}%", help="Spatial anomaly localization accuracy")
            with col3:
                val_lat = results_df['Mean Latency (ms)'].mean() if "Mean Latency (ms)" in results_df else 0
                st.metric("Average CPU Latency", f"{val_lat:.1f} ms", help="Per-specimen inference time on local CPU")
            with col4:
                val_fps = results_df['FPS'].mean() if "FPS" in results_df else 0
                st.metric("Conveyor Throughput", f"{val_fps:.1f} FPS", help="Frames inspected per second")

            st.markdown("---")

            # Comparative Chart
            chart_path = resolve_path("results/figures/category_auroc_comparison.png")
            if chart_path.exists():
                st.markdown("#### 📈 Multi-Category AUROC Distribution")
                st.image(str(chart_path), caption="Image vs Pixel AUROC by Industrial Category", use_container_width=True)

            st.markdown("#### 📋 Detailed Per-Category Inspection Metrics")
            st.dataframe(
                results_df.style.format({
                    "Image AUROC": "{:.4f}",
                    "Image AP": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "Pixel AUROC": "{:.4f}",
                    "Pixel AP": "{:.4f}",
                    "Pixel IoU": "{:.4f}",
                    "Mean Latency (ms)": "{:.1f}",
                    "FPS": "{:.1f}",
                }),
                use_container_width=True,
                height=450,
            )

        except Exception as e:
            st.warning(f"Could not parse benchmark summary: {e}")

    # Status Grid
    st.markdown("---")
    st.markdown("#### 📦 Feature Memory Bank & Checkpoint Fleet Matrix")
    status_data = []
    for cat in ALL_CATEGORIES:
        status_data.append({
            "Industrial Specimen": cat.replace("_", " ").title(),
            "Classification": "Surface Texture" if cat in TEXTURE_CATEGORIES else "Manufactured Object",
            "PatchCore Memory Bank": "✅ Calibrated & Ready" if patchcore_status.get(cat) else "⏳ Pending Build",
            "Conv-Autoencoder": "✅ Trained" if ae_status.get(cat) else "⏳ Pending Train",
            "Classical Structural Baseline": "✅ Ready (Zero-Shot)",
        })
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
