from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from surfaceguard.config import resolve_path


def main():
    reports_dir = resolve_path("results/reports")
    figures_dir = resolve_path("results/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "benchmark_summary.csv"
    if not csv_path.exists():
        print(f"No benchmark summary CSV found at {csv_path}. Run scripts/run_benchmark.py first.")
        return

    df = pd.read_csv(csv_path)

    # Generate Markdown Report
    report_md_path = reports_dir / "benchmark_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# SurfaceGuard AI — Benchmark & Evaluation Report\n\n")
        f.write("## Overview\n")
        f.write("This report aggregates empirical evaluation results across MVTec AD categories.\n")
        f.write("Unsupervised one-class visual inspection: trained exclusively on normal, defect-free images.\n\n")

        f.write("## Per-Category Metrics\n\n")
        f.write(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
        f.write("\n\n")

        f.write("## Macro Averages\n\n")
        if "Image AUROC" in df.columns:
            f.write(f"- **Macro Image AUROC**: {df['Image AUROC'].mean():.4f}\n")
        if "Pixel AUROC" in df.columns:
            f.write(f"- **Macro Pixel AUROC**: {df['Pixel AUROC'].mean():.4f}\n")
        if "Mean Latency (ms)" in df.columns:
            f.write(f"- **Mean Latency**: {df['Mean Latency (ms)'].mean():.2f} ms\n")
        if "FPS" in df.columns:
            f.write(f"- **Mean Throughput**: {df['FPS'].mean():.1f} FPS\n")

    print(f"Generated benchmark report at: {report_md_path}")

    # Generate Bar Chart Plot
    if "Category" in df.columns and "Image AUROC" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["Category"], df["Image AUROC"], color="#6366f1", edgecolor="#312e81")
        ax.set_ylabel("Image AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title("SurfaceGuard AI — Image Anomaly Detection Performance (AUROC)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        chart_path = figures_dir / "category_auroc_comparison.png"
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"Saved comparison chart at: {chart_path}")


if __name__ == "__main__":
    main()
