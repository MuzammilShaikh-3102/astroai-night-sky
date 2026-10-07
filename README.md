# AstroAI — AI Night-Sky Identifier

AstroAI is a research/demo web application that lets a user open the site on a smartphone, take or upload a night-sky photo, and receive an annotated identification of bright stars, likely planets, and constellations.

## What is in this first release

- 📷 Mobile-friendly Streamlit UI with camera capture and photo upload.
- 🧭 EXIF GPS/time extraction with manual latitude/longitude/time fallback.
- ✨ Computer-vision star-point detection with background subtraction and non-maximum suppression.
- 🛰️ Astronomical sky-position calculation using Astropy.
- ⭐ Star-field camera-pose search and catalog matching.
- 🪐 Planet position checking using Astropy's built-in solar-system ephemeris.
- 🧠 A compact CNN trained on synthetic constellation images as the machine-learning component.
- 🖼️ Annotated image output with star/planet labels and confidence values.
- 📚 HYG v4.1 download-and-cache support, with a 100+ bright-star seed catalog so the application can still start if the download is unavailable.
- 🧪 Reproducible synthetic-data generator and CNN training script.

## Important project design choice

A star usually does not have a unique visual appearance in a normal phone photograph. Exact identification is therefore treated as a star-field/astrometry problem, not a flat image-classification problem. The neural network provides constellation-pattern evidence, while the astronomy layer performs the precise catalog verification.

## Quick start on Windows + Anaconda

```text
conda create -n astroai python=3.11 -y
conda activate astroai
pip install -r requirements.txt
streamlit run app.py
```

The first analysis may download the HYG v4.1 catalog into `data/cache/`. Later runs reuse the local trimmed catalog.

## Test the ML pipeline

From the project root:

```text
python scripts/generate_synthetic.py --per-class 250
python scripts/train_constellation_model.py --epochs 6
```

The trained weights are saved to `models/star_patch_cnn.pt`. The CNN is optional at runtime so a difficult real photo can still be processed with the classical star detector + astronomy solver.

For a quick code-level solver test:

```text
python -m pytest tests
```

## Using the site from your iPhone

For a real phone test, the most reliable path is an HTTPS deployment (for example, Streamlit Community Cloud connected to a GitHub repository). Browsers restrict direct camera access on ordinary insecure web pages, so a plain `http://192.168.x.x:8501` LAN URL may allow uploading a photo but may block one-tap camera capture.

Once deployed over HTTPS:

1. Open the site in Safari on your iPhone.
2. Tap **Take a night-sky photo**.
3. Give camera permission.
4. Take the photo.
5. Confirm the photo location/time. EXIF GPS is used when available; manual values are always accepted.
6. Tap **Analyze my sky**.

For the first real-world test, use the main camera rather than the ultra-wide lens, avoid clouds and bright lights in the frame, and include several visible stars.

## Suggested project experiment

Train on synthetic star fields and evaluate on real smartphone images. Measure:

- star-identification precision/recall,
- constellation top-1/top-3 accuracy,
- pose/field-of-view error,
- failure rate under blur/light pollution/few-star conditions.

The synthetic-to-real gap is an intended research question, not something the project hides.

## Data / astronomy references

HYG v4.1 field definitions and license are documented by the HYG project README. The project uses only a small visible subset at runtime.

Useful official astronomy services for later extensions include ESA Gaia, CDS SIMBAD/VizieR, and NASA/JPL Horizons.
