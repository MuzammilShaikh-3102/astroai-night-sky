from __future__ import annotations

from datetime import datetime, time, date
from pathlib import Path

import streamlit as st
from PIL import Image

from src.exif_utils import extract_metadata
from src.vision import load_rgb, prepare_image
from src.catalog import load_catalog
from src.analysis import analyze, constellation_edges
from src.annotate import annotate_image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="AstroAI — Night Sky Identifier", page_icon="🌌", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 1.2rem; }
    .hero { padding: 1.2rem 1.2rem 1.0rem; border-radius: 22px; background: linear-gradient(135deg,#0b1028,#17204b); color: white; }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { margin: .4rem 0 0; opacity: .86; }
    .result-card { padding: 1rem; border: 1px solid #29325e; border-radius: 15px; background:#0d1330; }
    .small { opacity: .75; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🌌 AstroAI</h1><p>AI-assisted night-sky identification from a smartphone photo.</p></div>',
    unsafe_allow_html=True,
)

st.write("Take a picture directly on your phone, or upload a night-sky image. The app combines deep learning, star-pattern geometry, and astronomical catalogs.")

left, right = st.columns([1, 1])
with left:
    camera = st.camera_input("📷 Take a night-sky photo")
with right:
    upload = st.file_uploader("📁 Or upload a photo", type=["jpg", "jpeg", "png", "webp", "heic", "heif"])

source = camera or upload
if source is None:
    st.info("Start with a clear night-sky photo containing several visible stars. A wider field of view is helpful.")
    with st.expander("What this first version can identify"):
        st.write("Bright named stars, likely constellations, and planets when their predicted position is visible in the solved field. Exact performance depends on image quality, field of view, and sky conditions.")
    st.stop()

raw_bytes = source.getvalue()
meta = extract_metadata(raw_bytes)
try:
    image = prepare_image(load_rgb(raw_bytes))
except Exception as exc:
    st.error(f"Could not read this image. For iPhone photos, use the Camera button or export as JPEG/PNG. Details: {exc}")
    st.stop()

st.subheader("1. Observation information")
meta_col1, meta_col2 = st.columns([1,1])
with meta_col1:
    if meta.get("gps"):
        ex_lat, ex_lon = meta["gps"]
        st.success(f"GPS found in photo: {ex_lat:.5f}, {ex_lon:.5f}")
    else:
        ex_lat, ex_lon = 41.88, -87.63
        st.warning("No GPS was found in the uploaded file. Enter the photo location below.")
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=float(ex_lat), format="%.6f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=float(ex_lon), format="%.6f")

with meta_col2:
    default_dt = meta.get("datetime")
    default_date = default_dt.date() if default_dt else date.today()
    default_time = default_dt.time().replace(second=0, microsecond=0) if default_dt else time(22, 0)
    if default_dt:
        st.success(f"Photo timestamp found: {default_dt.strftime('%Y-%m-%d %H:%M:%S')} (timezone still needs confirmation)")
    else:
        st.info("No photo timestamp found. Enter when the image was taken.")
    obs_date = st.date_input("Photo date", value=default_date)
    obs_time = st.time_input("Photo local time", value=default_time)
    tz = st.selectbox("UTC offset at the capture location", options=[x/2 for x in range(-24, 29)], index=24, format_func=lambda x: f"UTC{int(x):+d}" if float(x).is_integer() else f"UTC{x:+g}")

st.subheader("2. Camera / solve settings")
c1, c2 = st.columns([1,1])
with c1:
    hfov_hint = st.slider("Camera horizontal field of view hint (degrees)", 35, 120, 65, 1, help="Use ~60–70° for a typical main phone camera. Ultra-wide lenses are usually much wider.")
with c2:
    star_limit = st.slider("Maximum stars to consider", 10, 60, 40, 5)
use_ml = st.checkbox("Use CNN star verification", value=True, help="The included CNN was trained on synthetic point-source patches. Turn this off if a particular real photo loses too many faint stars.")

analyze_button = st.button("✨ Analyze my sky", type="primary", use_container_width=True)

if analyze_button:
    # The current solver searches FOV around this hint; store it for future refinement.
    dt_local = datetime.combine(obs_date, obs_time)
    with st.spinner("Detecting stars and solving the sky pattern…"):
        catalog, catalog_source = load_catalog(prefer_online=True)
        result = analyze(image, dt_local, float(tz), float(lat), float(lon), catalog=catalog, hfov_hint=hfov_hint, max_detected_stars=star_limit, use_ml=use_ml)
        result["metadata"]["catalog_source"] = catalog_source
        result["metadata"]["fov_hint"] = hfov_hint
    st.session_state["result"] = result
    st.session_state["image"] = image
    st.session_state["catalog"] = catalog
    st.session_state["obs"] = (dt_local, float(tz), float(lat), float(lon))

if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
image = st.session_state["image"]
catalog = st.session_state["catalog"]

st.subheader("3. Analysis result")
if result.get("solve") and result["solve"].success:
    solve = result["solve"]
    a,b,c,d = st.columns(4)
    a.metric("Stars detected", result["metadata"]["stars_detected"])
    b.metric("Pattern inliers", solve.inliers)
    c.metric("Estimated FOV", f"{solve.hfov:.0f}°")
    d.metric("Sky center", f"Az {solve.center_az:.0f}° / Alt {solve.center_alt:.0f}°")
else:
    st.warning(result.get("warning", "The sky could not be solved confidently."))

# Determine a constellation to draw from geometric matches or ML.
constellation = result["constellations"][0][0] if result["constellations"] else None
edges = constellation_edges().get(constellation, []) if constellation else []
matched_names = [s["name"] for s in result["stars"]]
annotated = annotate_image(image, result["detections"], result["matches"], catalog, edges, {p["name"]:(p["x"],p["y"],p["altitude"]) for p in result["planets"]}, matched_names)

img_col, info_col = st.columns([1.25, 0.75])
with img_col:
    st.image(annotated, caption="Detected objects overlaid on the photo", use_container_width=True)
with info_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### ⭐ Stars")
    if result["stars"]:
        for star in result["stars"][:12]:
            st.write(f"**{star['name']}** — {star['confidence']*100:.0f}% · mag {star['magnitude']:.2f}")
    else:
        st.write("No confident star identities yet.")
    st.markdown("### 🪐 Planets")
    if result["planets"]:
        for planet in result["planets"]:
            st.write(f"**{planet['name']}** — {planet['confidence']*100:.0f}% · altitude {planet['altitude']:.1f}°")
    else:
        st.write("No planet confidently matched to a detected point.")
    st.markdown("### ✨ Constellation")
    if constellation:
        st.write(f"**{constellation}**")
        st.caption("CNN star verification: enabled" if result["metadata"].get("ml_star_verifier") else "CNN star verification: not loaded")
    else:
        st.write("Not enough evidence for a constellation label.")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("🔬 Technical details for the ML course"):
    st.write("The project is intentionally hybrid. A compact CNN is trained on synthetic point-source patches to verify likely stars in the photograph. Exact star identities are then solved from the relative pattern of stars against a celestial catalog, and constellation names are inferred from the identified key stars. This avoids treating a single star as if it had a unique visual appearance.")
    if result.get("solve"):
        st.json({
            "success": result["solve"].success,
            "pattern_inliers": result["solve"].inliers,
            "estimated_horizontal_FOV_deg": round(result["solve"].hfov, 2),
            "estimated_center_azimuth_deg": round(result["solve"].center_az, 2),
            "estimated_center_altitude_deg": round(result["solve"].center_alt, 2),
            "catalog": result["metadata"]["catalog_source"],
        })

st.caption("AstroAI prototype — use as a research/demo system, not as a scientific astrometry instrument.")
