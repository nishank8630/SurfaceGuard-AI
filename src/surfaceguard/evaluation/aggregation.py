from typing import Any, Dict, List, Optional
import pandas as pd
from surfaceguard.config import OBJECT_CATEGORIES, TEXTURE_CATEGORIES


def aggregate_benchmark_results(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregates per-category evaluation metrics into a pandas DataFrame."""
    rows = []
    for res in results_list:
        if "error" in res:
            continue

        cat = res["category"]
        img_m = res.get("image_metrics", {})
        pix_m = res.get("pixel_metrics", {})
        lat_m = res.get("latency", {})

        cat_type = "Texture" if cat in TEXTURE_CATEGORIES else "Object"

        row = {
            "Category": cat,
            "Type": cat_type,
            "Model": res.get("model_type", "Unknown"),
            "Image AUROC": img_m.get("auroc"),
            "Image AP": img_m.get("average_precision"),
            "F1 Score": img_m.get("f1_score"),
            "Pixel AUROC": pix_m.get("pixel_auroc"),
            "Pixel AP": pix_m.get("pixel_ap"),
            "Pixel IoU": pix_m.get("pixel_iou"),
            "Mean Latency (ms)": lat_m.get("mean_ms"),
            "FPS": lat_m.get("throughput_fps"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_macro_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes equal-weight macro averages across evaluated categories."""
    if df.empty:
        return {}

    numeric_cols = [
        "Image AUROC",
        "Image AP",
        "F1 Score",
        "Pixel AUROC",
        "Pixel AP",
        "Pixel IoU",
        "Mean Latency (ms)",
        "FPS",
    ]

    macro_avg = {}
    for col in numeric_cols:
        if col in df.columns:
            valid_vals = df[col].dropna()
            macro_avg[col] = float(valid_vals.mean()) if not valid_vals.empty else None

    # Object vs Texture breakdown
    obj_df = df[df["Type"] == "Object"]
    tex_df = df[df["Type"] == "Texture"]

    return {
        "macro_average": macro_avg,
        "object_average_auroc": float(obj_df["Image AUROC"].dropna().mean()) if not obj_df.empty else None,
        "texture_average_auroc": float(tex_df["Image AUROC"].dropna().mean()) if not tex_df.empty else None,
        "total_categories_evaluated": len(df),
    }
