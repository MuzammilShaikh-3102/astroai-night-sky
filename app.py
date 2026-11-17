from __future__ import annotations

from datetime import datetime, date, time, timezone
from hashlib import sha256
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
st.markdown("""
<style>
.block-container { max-width: 1100px; padding-top: 1.2rem; }
.hero { padding: 1.2rem 1.2rem 1rem; border-radius: 22px; background: linear-gradient(135deg,#0b1028,#17204b); color: white; }
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin: .4rem 0 0; opacity: .86; }
.result-card { padding: 1rem; border: 1px solid #29325e; border-radius: 15px; background:#0d1330; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌌 AstroAI</h1><p>AI-assisted night-sky identification from a smartphone photo.</p></div>', unsafe_allow_html=True)
st.write("Upload the original photo from your iPhone Camera roll whenever possible. AstroAI reads each image's own metadata; it does not assume that every photo was taken in the same place.")

camera, upload = None, None
left, right = st.columns(2)
with left:
    camera = st.camera_input("📷 Take a night-sky photo")
with right:
    upload = st.file_uploader("📁 Upload the original iPhone photo", type=["jpg", "jpeg", "png", "webp", "heic", "heif"])

source = camera if camera is not None else upload
if source is None:
    st.info("Start with an original photo containing several visible stars. Keep the original file rather than a screenshot or a social-media copy.")
    st.stop()

raw_bytes = source.getvalue()
image_id = sha256(raw_bytes).hexdigest()[:20]
# Prevent stale results and stale widget values from one photo being reused for another.
if st.session_state.get("active_image_id") != image_id:
    for key in ("result", "image", "catalog", "obs", "result_image_id"):
        st.session_state.pop(key, None)
    st.session_state["active_image_id"] = image_id

meta = extract_metadata(raw_bytes)
try:
    image = prepare_image(load_rgb(raw_bytes))
except Exception as exc:
    st.error(f"Could not read this image. Try exporting the original photo as JPEG. Details: {exc}")
    st.stop()

st.subheader("1. Observation information")
if meta.get("camera"):
    st.caption(f"Camera metadata: {meta['camera']}")
if meta.get("metadata_warnings"):
    for warning in meta["metadata_warnings"]:
        st.warning(warning)

meta_col1, meta_col2 = st.columns(2)
with meta_col1:
    st.markdown("**Where was this photo captured?**")
    gps = meta.get("gps")
    if gps:
        st.success(f"GPS read from this photo: {gps[0]:.6f}, {gps[1]:.6f}")
    else:
        st.warning("No GPS in this file. Enter the actual capture location; AstroAI will not insert a default city.")
    lat = st.number_input("Capture latitude", min_value=-90.0, max_value=90.0,
                          value=float(gps[0]) if gps else None, format="%.6f",
                          placeholder="Required when GPS is missing", key=f"lat_{image_id}")
    lon = st.number_input("Capture longitude", min_value=-180.0, max_value=180.0,
                          value=float(gps[1]) if gps else None, format="%.6f",
                          placeholder="Required when GPS is missing", key=f"lon_{image_id}")

with meta_col2:
    gps_utc = meta.get("gps_datetime_utc")
    exif_dt = meta.get("datetime")
    offset_from_exif = meta.get("utc_offset_hours")
    if gps_utc:
        default_date, default_time = gps_utc.date(), gps_utc.time().replace(tzinfo=None, microsecond=0)
        st.success("Capture time read from GPS metadata (already UTC).")
    elif exif_dt:
        default_date, default_time = exif_dt.date(), exif_dt.time().replace(second=0, microsecond=0)
        st.info(f"EXIF local capture time: {exif_dt:%Y-%m-%d %H:%M:%S}")
    else:
        default_date, default_time = date.today(), time(22, 0)
        st.warning("No capture time found. Enter the original local date and time.")
    obs_date = st.date_input("Capture date", value=default_date, key=f"date_{image_id}")
    obs_time = st.time_input("Capture time", value=default_time, key=f"time_{image_id}")

    offsets = [x / 4 for x in range(-56, 57)]
    if gps_utc:
        default_offset = 0.0
        st.caption("GPS timestamp is UTC, so no local-offset conversion is needed.")
    else:
        default_offset = offset_from_exif
        if offset_from_exif is not None:
            st.success(f"UTC offset read from EXIF: {offset_from_exif:+g} hours")
        else:
            st.warning("The photo does not specify its timezone. Select the UTC offset that applied at the capture location on that date.")
    offset_index = offsets.index(default_offset) if default_offset in offsets else None
    tz = st.selectbox(
        "UTC offset at capture location",
        options=offsets,
        index=offset_index,
        placeholder="Select the correct offset",
        format_func=lambda x: f"UTC{x:+g} ({int(abs(x)):02d}:{int(round((abs(x) % 1)*60)):02d})",
        key=f"tz_{image_id}",
        disabled=bool(gps_utc),
    )

st.subheader("2. Camera / solve settings")
fov_meta = meta.get("estimated_hfov_deg")
if fov_meta:
    st.success(f"Estimated horizontal FOV from 35-mm-equivalent focal length: {fov_meta:.1f}°")
    fov_default = int(round(min(120, max(35, fov_meta))))
else:
    st.info("FOV means the horizontal angle of sky covered by the image. It is a search hint, not the camera's direction. If camera lens metadata is unavailable, leave the approximate value unless you know the lens specification.")
    fov_default = 70
c1, c2 = st.columns(2)
with c1:
    hfov_hint = st.slider("Approximate horizontal sky coverage (degrees)", 35, 120, fov_default, 1,
                          help="Use the camera/lens used for this photo. A wrong value can make the current solver miss the correct sky pattern.")
with c2:
    star_limit = st.slider("Maximum star candidates", 10, 60, 40, 5)
use_ml = st.checkbox("Use CNN star verification", value=True,
                     help="This CNN is a candidate filter trained on synthetic patches; disable it if it removes real stars.")

analyze_button = st.button("✨ Analyze my sky", type="primary", use_container_width=True)
if analyze_button:
    if lat is None or lon is None:
        st.error("Capture latitude and longitude are required. Add GPS to the photo or enter the location where it was taken.")
        st.stop()
    if gps_utc is None and tz is None:
        st.error("Please select the UTC offset that applied at the capture location on the capture date.")
        st.stop()
    dt_local = datetime.combine(obs_date, obs_time)
    # GPS date/time is already UTC. Otherwise convert the entered local clock time once.
    effective_offset = 0.0 if gps_utc else float(tz)
    with st.spinner("Detecting stars and solving the sky pattern…"):
        catalog, catalog_source = load_catalog(prefer_online=True)
        result = analyze(image, dt_local, effective_offset, float(lat), float(lon),
                         catalog=catalog, hfov_hint=float(hfov_hint),
                         max_detected_stars=star_limit, use_ml=use_ml)
        result["metadata"]["catalog_source"] = catalog_source
        result["metadata"]["fov_hint"] = float(hfov_hint)
        result["metadata"]["observation_location_source"] = "image EXIF GPS" if gps else "manual entry"
        result["metadata"]["observation_time_source"] = "GPS UTC" if gps_utc else ("EXIF local time" if exif_dt else "manual entry")
        result["metadata"]["utc_offset_hours"] = effective_offset
        result["metadata"]["image_id"] = image_id
    st.session_state["result"] = result
    st.session_state["image"] = image
    st.session_state["catalog"] = catalog
    st.session_state["obs"] = (dt_local, effective_offset, float(lat), float(lon))
    st.session_state["result_image_id"] = image_id

if st.session_state.get("result_image_id") != image_id or "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
image = st.session_state["image"]
catalog = st.session_state["catalog"]
st.subheader("3. Analysis result")
if result.get("solve") and result["solve"].success:
    solve = result["solve"]
    a, b, c, d = st.columns(4)
    a.metric("Stars detected", result["metadata"]["stars_detected"])
    b.metric("Pattern inliers", solve.inliers)
    c.metric("Estimated FOV", f"{solve.hfov:.1f}°")
    d.metric("Sky center", f"Az {solve.center_az:.1f}° / Alt {solve.center_alt:.1f}°")
else:
    st.warning(result.get("warning", "The sky could not be solved confidently."))

constellation = result["constellations"][0][0] if result["constellations"] else None
edges = constellation_edges().get(constellation, []) if constellation else []
matched_names = [s["name"] for s in result["stars"]]
annotated = annotate_image(image, result["detections"], result["matches"], catalog, edges,
                           {p["name"]: (p["x"], p["y"], p["altitude"]) for p in result["planets"]}, matched_names)
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
    else:
        st.write("Not enough evidence for a constellation label.")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("🔬 Technical details"):
    st.json({
        "image_metadata": {
            "gps_found": bool(meta.get("gps")),
            "gps_datetime_utc_found": bool(gps_utc),
            "exif_timezone_offset_found": offset_from_exif is not None,
            "camera": meta.get("camera"),
            "estimated_hfov_deg": meta.get("estimated_hfov_deg"),
        },
        "observation": result["metadata"],
        "solve": ({
            "success": result["solve"].success,
            "pattern_inliers": result["solve"].inliers,
            "estimated_horizontal_FOV_deg": round(result["solve"].hfov, 2),
            "estimated_center_azimuth_deg": round(result["solve"].center_az, 2),
            "estimated_center_altitude_deg": round(result["solve"].center_alt, 2),
            "roll_deg": round(result["solve"].roll, 2),
        } if result.get("solve") else None),
    })
st.caption("AstroAI is a research prototype. It should not claim an identity when the image cannot be solved reliably.")
