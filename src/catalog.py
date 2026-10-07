from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# HYG v4.1 is documented in the repository README as the current archived dataset.
HYG_URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
LOCAL_CACHE = CACHE_DIR / "hyg_visible.csv"
SEED_FILE = DATA_DIR / "stars_seed.csv"


def load_seed_catalog() -> pd.DataFrame:
    df = pd.read_csv(SEED_FILE)
    df["source"] = "seed"
    df["display_name"] = df["name"]
    return df[["display_name", "name", "constellation", "ra_h", "dec_deg", "mag", "spectral", "source"]]


def _download_hyg() -> bool:
    import urllib.request

    raw = CACHE_DIR / "hygdata_v41.csv"
    if raw.exists() and raw.stat().st_size > 2_000_000:
        return True
    try:
        req = urllib.request.Request(HYG_URL, headers={"User-Agent": "AstroAI/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response, open(raw, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return raw.exists() and raw.stat().st_size > 2_000_000
    except Exception:
        return False


def _prepare_hyg() -> pd.DataFrame | None:
    raw = CACHE_DIR / "hygdata_v41.csv"
    if not raw.exists():
        return None
    try:
        df = pd.read_csv(raw, usecols=["proper", "bf", "hip", "ra", "dec", "mag", "spect", "con"])
        df = df.rename(columns={"proper": "name", "bf": "bayer_flamsteed", "hip": "hip", "ra": "ra_h", "dec": "dec_deg", "spect": "spectral", "con": "constellation"})
        df = df[pd.to_numeric(df["mag"], errors="coerce").notna()].copy()
        df["mag"] = df["mag"].astype(float)
        df["ra_h"] = pd.to_numeric(df["ra_h"], errors="coerce")
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")
        df = df[df["ra_h"].notna() & df["dec_deg"].notna()]
        # Keep stars likely to appear in a phone image. Named stars are retained even if a
        # little fainter than the visual cutoff so the UI can label familiar objects.
        df = df[df["mag"] <= 5.5]
        df["name"] = df["name"].fillna("").astype(str).str.strip()
        df["display_name"] = df["name"]
        df.loc[df["display_name"] == "", "display_name"] = df.loc[df["display_name"] == "", "bayer_flamsteed"].fillna("")
        df.loc[df["display_name"] == "", "display_name"] = "HIP " + df.loc[df["display_name"] == "", "hip"].fillna("").astype(str)
        df["source"] = "HYG v4.1"
        return df[["display_name", "name", "constellation", "ra_h", "dec_deg", "mag", "spectral", "source"]].drop_duplicates(subset=["display_name", "ra_h", "dec_deg"])
    except Exception:
        return None


def load_catalog(prefer_online: bool = True) -> tuple[pd.DataFrame, str]:
    if prefer_online:
        _download_hyg()
    df = _prepare_hyg()
    if df is not None and len(df) > 100:
        df.to_csv(LOCAL_CACHE, index=False)
        return df.sort_values("mag").reset_index(drop=True), "HYG v4.1 (trimmed)"
    if LOCAL_CACHE.exists():
        try:
            cached = pd.read_csv(LOCAL_CACHE)
            return cached.sort_values("mag").reset_index(drop=True), "HYG v4.1 (cached)"
        except Exception:
            pass
    return load_seed_catalog().sort_values("mag").reset_index(drop=True), "Seed bright-star catalog"


def solver_catalog(df: pd.DataFrame, max_stars: int = 250) -> pd.DataFrame:
    return df.sort_values(["mag", "display_name"]).head(max_stars).copy()
