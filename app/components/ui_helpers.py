import streamlit as st
from typing import Dict, Any


CATEGORY_META = {
    "bottle": {"icon": "🍾", "title": "Glass Containers & Bottles", "type": "Object", "tolerance": "High"},
    "cable": {"icon": "🔌", "title": "Multi-Core Wiring Bundles", "type": "Object", "tolerance": "Critical"},
    "capsule": {"icon": "💊", "title": "Pharmaceutical Gel Capsules", "type": "Object", "tolerance": "Critical"},
    "carpet": {"icon": "🧶", "title": "Woven Textile Fabrics", "type": "Texture", "tolerance": "Standard"},
    "grid": {"icon": "🔲", "title": "Structural Precision Grids", "type": "Texture", "tolerance": "High"},
    "hazelnut": {"icon": "🌰", "title": "Organic Agricultural Goods", "type": "Object", "tolerance": "Standard"},
    "leather": {"icon": "👜", "title": "Tanned Leather Materials", "type": "Texture", "tolerance": "High"},
    "metal_nut": {"icon": "🔩", "title": "Hexagonal Machine Nuts", "type": "Object", "tolerance": "Critical"},
    "pill": {"icon": "⚪", "title": "Compressed Medicine Tablets", "type": "Object", "tolerance": "Critical"},
    "screw": {"icon": "🪛", "title": "Precision Fastener Screws", "type": "Object", "tolerance": "Critical"},
    "tile": {"icon": "🧱", "title": "Architectural Ceramic Tiles", "type": "Texture", "tolerance": "High"},
    "toothbrush": {"icon": "🪥", "title": "Micro-Bristle Assemblies", "type": "Object", "tolerance": "Standard"},
    "transistor": {"icon": "⚡", "title": "Semiconductor Electronics", "type": "Object", "tolerance": "Critical"},
    "wood": {"icon": "🪵", "title": "Engineered Wood Veneers", "type": "Texture", "tolerance": "Standard"},
    "zipper": {"icon": "🤐", "title": "Mechanical Zipper Fasteners", "type": "Object", "tolerance": "Standard"},
}


def apply_custom_css():
    """Injects high-end cyber-industrial glassmorphism aesthetic CSS."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

        /* Root color palette */
        :root {
            --bg-primary: #070a13;
            --bg-card: rgba(15, 23, 42, 0.72);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-glow: rgba(56, 189, 248, 0.22);
            --cyan-neon: #38bdf8;
            --cyan-glow: rgba(56, 189, 248, 0.4);
            --emerald-neon: #10b981;
            --emerald-glow: rgba(16, 185, 129, 0.4);
            --crimson-neon: #f43f5e;
            --crimson-glow: rgba(244, 63, 94, 0.45);
            --amber-neon: #f59e0b;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #e2e8f0;
        }

        /* Top Industrial Header */
        .industrial-hero {
            background: radial-gradient(circle at 10% 20%, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.85) 60%),
                        linear-gradient(135deg, #090e1a 0%, #0d1527 50%, #171630 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 2rem 2.4rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.65), 0 0 25px rgba(56, 189, 248, 0.08);
            position: relative;
            overflow: hidden;
        }

        .industrial-hero::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e, #10b981);
        }

        .industrial-hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.3rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            background: linear-gradient(120deg, #ffffff 0%, #bae6fd 45%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .industrial-hero p {
            color: #94a3b8;
            font-size: 1.02rem;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
            font-weight: 400;
        }

        /* System Telemetry Chips */
        .hud-chip {
            display: inline-flex;
            align-items: center;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #cbd5e1;
            padding: 0.28rem 0.75rem;
            border-radius: 6px;
            font-size: 0.76rem;
            font-family: 'JetBrains Mono', monospace;
            margin-right: 0.5rem;
            margin-top: 0.4rem;
            letter-spacing: 0.03em;
        }

        .hud-chip.active {
            border-color: rgba(56, 189, 248, 0.5);
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.1);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
            display: inline-block;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 12px #10b981; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        /* High-Tech Card */
        .tech-card {
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            transition: all 0.2s ease;
        }

        .tech-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6), 0 0 15px rgba(56, 189, 248, 0.1);
        }

        .card-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Metric Tile */
        .metric-tile {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.1rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .metric-tile .label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 500;
            margin-bottom: 0.3rem;
        }

        .metric-tile .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.65rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .metric-tile .sub {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Decision Verdict Banners */
        .verdict-banner-defect {
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.2) 0%, rgba(159, 18, 57, 0.35) 100%);
            border: 1.5px solid #f43f5e;
            border-radius: 14px;
            padding: 1.4rem 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 0 35px rgba(244, 63, 94, 0.3), inset 0 0 15px rgba(244, 63, 94, 0.1);
            text-align: center;
        }

        .verdict-banner-defect h2 {
            font-family: 'Space Grotesk', sans-serif;
            color: #fda4af;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin: 0;
        }

        .verdict-banner-defect p {
            color: #fecdd3;
            font-size: 0.95rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
            font-family: 'JetBrains Mono', monospace;
        }

        .verdict-banner-normal {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
            border: 1.5px solid #10b981;
            border-radius: 14px;
            padding: 1.4rem 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 0 35px rgba(16, 185, 129, 0.25), inset 0 0 15px rgba(16, 185, 129, 0.1);
            text-align: center;
        }

        .verdict-banner-normal h2 {
            font-family: 'Space Grotesk', sans-serif;
            color: #6ee7b7;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin: 0;
        }

        .verdict-banner-normal p {
            color: #a7f3d0;
            font-size: 0.95rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Streamlit Button Restyling */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 10px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.02em !important;
            padding: 0.65rem 1.5rem !important;
            box-shadow: 0 4px 18px rgba(37, 99, 235, 0.4) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 24px rgba(37, 99, 235, 0.6), 0 0 15px rgba(56, 189, 248, 0.5) !important;
        }

        /* HUD Frame for Images */
        .hud-frame {
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 10px;
            padding: 6px;
            background: rgba(15, 23, 42, 0.6);
            margin-bottom: 0.6rem;
            box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.5);
        }

        /* Progress Bar Accent */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #10b981 0%, #38bdf8 50%, #f43f5e 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Renders the top branding header with high-tech live indicators."""
    st.markdown(
        """
        <div class="industrial-hero">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <h1>SurfaceGuard AI</h1>
                    <p>Automated Optical Inspection (AOI) & Real-Time Pixel-Level Defect Localization</p>
                </div>
                <div style="text-align: right; margin-top: 0.4rem;">
                    <span class="hud-chip active"><span class="pulse-dot"></span>STATION: ACTIVE</span>
                    <span class="hud-chip">LATENCY: &lt;100ms</span>
                </div>
            </div>
            <div style="margin-top: 0.6rem;">
                <span class="hud-chip">🔬 Deep PatchCore Embeddings</span>
                <span class="hud-chip">📐 ResNet-18 Multi-Scale Features</span>
                <span class="hud-chip">🎯 99% Precision Calibrated</span>
                <span class="hud-chip">⚡ Sub-Second Edge CPU</span>
                <span class="hud-chip">🏭 15 Industrial MVTec Classes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_banner(is_anomalous: bool, anomaly_score: float, threshold: float):
    """Renders an eye-catching, high-tech industrial verdict banner with deviation stats."""
    deviation = ((anomaly_score - threshold) / threshold) * 100.0

    if is_anomalous:
        st.markdown(
            f"""
            <div class="verdict-banner-defect">
                <h2>🚨 CRITICAL DEFECT DETECTED // TOLERANCE BREACH</h2>
                <p>ANOMALY SCORE: <strong>{anomaly_score:.4f}</strong> &nbsp;|&nbsp; 
                   OPERATING THRESHOLD: <strong>{threshold:.4f}</strong> &nbsp;|&nbsp; 
                   SURFACE DEVIATION: <strong>+{deviation:.1f}% OVER TOLERANCE</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-banner-normal">
                <h2>✅ SURFACE INTEGRITY VERIFIED // QUALITY PASSED</h2>
                <p>ANOMALY SCORE: <strong>{anomaly_score:.4f}</strong> &nbsp;|&nbsp; 
                   OPERATING THRESHOLD: <strong>{threshold:.4f}</strong> &nbsp;|&nbsp; 
                   TOLERANCE MARGIN: <strong>{abs(deviation):.1f}% WITHIN SPECS</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_tiles(res: Dict[str, Any]):
    """Renders four styled cyber-industrial telemetry metric tiles."""
    score = res["anomaly_score"]
    thresh = res["threshold"]
    latency = res["inference_time_ms"]
    fps = 1000.0 / max(latency, 1.0)
    is_anom = res["is_anomalous"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="label">Discrepancy Score</div>
                <div class="value" style="color: {'#f43f5e' if is_anom else '#38bdf8'};">{score:.4f}</div>
                <div class="sub">Ref Euclidean Distance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="label">Operating Threshold</div>
                <div class="value" style="color: #cbd5e1;">{thresh:.4f}</div>
                <div class="sub">99% Specimen Precision</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        status_color = "#f43f5e" if is_anom else "#10b981"
        status_text = "DEFECT" if is_anom else "VERIFIED"
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="label">Quality Disposition</div>
                <div class="value" style="color: {status_color};">{status_text}</div>
                <div class="sub">{'Route to Reject Bin' if is_anom else 'Pass to Packaging'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="label">Inspection Latency</div>
                <div class="value" style="color: #a78bfa;">{latency:.1f} <span style="font-size:0.9rem;">ms</span></div>
                <div class="sub">{fps:.1f} Frames / Sec</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_specimen_label(path) -> str:
    """Formats raw dataset path into realistic industrial inspection specimen label."""
    parent = path.parent.name
    filename = path.stem

    # Map raw folder defect names to realistic engineering defect terminology
    defect_names = {
        "good": "Standard Baseline Lot (Tolerance Approved)",
        "broken_large": "Severe Structural Fracture",
        "broken_small": "Micro-Fracture / Chip",
        "contamination": "Particulate Contamination",
        "bent_wire": "Terminal Wire Deformation",
        "cable_swap": "Pin Alignment Disorder",
        "combined": "Multiple Structural Flaws",
        "cut_inner_insulation": "Insulation Dielectric Cut",
        "cut_outer_insulation": "Outer Sheath Laceration",
        "missing_cable": "Missing Terminal Wire",
        "missing_wire": "Omitted Core Conductor",
        "poke_insulation": "Puncture Abrasion",
        "crack": "Surface Stress Crack",
        "faulty_imprint": "Imprint / Laser Etch Flaw",
        "poke": "Perforation Puncture",
        "scratch": "Surface Linear Scratch",
        "squeeze": "Compressive Deformation",
        "color": "Chrominance Discoloration",
        "cut": "Shear Laceration",
        "hole": "Void / Penetration Hole",
        "metal_contamination": "Metallic Ingot Inclusions",
        "thread": "Loose Fiber Inclusion",
        "bent": "Geometric Bend Deviation",
        "glue": "Excess Adhesive Residue",
        "thread_side": "Side Filament Entanglement",
        "print": "Silkscreen Print Misalignment",
        "glue": "Adhesive Deposition Fault",
        "rough": "Surface Roughness Non-Uniformity",
    }

    friendly_defect = defect_names.get(parent, parent.replace("_", " ").title())
    is_normal = (parent == "good")

    prefix = "🟢 QC-PASS" if is_normal else "🔴 AUDIT-SAMPLE"
    return f"{prefix} | Lot #{filename} — {friendly_defect}"
