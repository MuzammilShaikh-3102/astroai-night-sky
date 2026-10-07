from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from PIL import Image


def _rational_to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(value.numerator) / float(value.denominator)
        except Exception:
            return float(value[0]) / float(value[1])


def _dms_to_deg(values: Any) -> float:
    d, m, s = (_rational_to_float(v) for v in values)
    return d + m / 60.0 + s / 3600.0


def _parse_gps(gps: dict[Any, Any]) -> tuple[float, float] | None:
    # EXIF GPS tags: 1=N/S, 2=latitude, 3=E/W, 4=longitude.
    lat = gps.get(2)
    lon = gps.get(4)
    if lat is None or lon is None:
        return None
    lat_deg = _dms_to_deg(lat)
    lon_deg = _dms_to_deg(lon)
    lat_ref = gps.get(1)
    lon_ref = gps.get(3)
    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode(errors="ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode(errors="ignore")
    if str(lat_ref).upper() == "S":
        lat_deg *= -1
    if str(lon_ref).upper() == "W":
        lon_deg *= -1
    return lat_deg, lon_deg


def extract_metadata(file_bytes: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gps": None,
        "datetime": None,
        "raw_datetime": None,
        "camera": None,
        "has_exif": False,
    }
    try:
        image = Image.open(BytesIO(file_bytes))
        exif = image.getexif()
        if not exif:
            return result
        result["has_exif"] = True

        # 34665 is ExifIFDPointer. Pillow exposes the nested EXIF values via get_ifd.
        try:
            exif_ifd = exif.get_ifd(34665)
        except Exception:
            exif_ifd = {}
        try:
            gps_ifd = exif.get_ifd(34853)
        except Exception:
            gps_ifd = {}

        gps = _parse_gps(gps_ifd)
        if gps:
            result["gps"] = gps

        raw_dt = exif_ifd.get(36867) or exif.get(36867)  # DateTimeOriginal
        if isinstance(raw_dt, bytes):
            raw_dt = raw_dt.decode(errors="ignore")
        result["raw_datetime"] = raw_dt
        if raw_dt:
            try:
                result["datetime"] = datetime.strptime(str(raw_dt), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

        make = exif.get(271) or exif_ifd.get(271)
        model = exif.get(272) or exif_ifd.get(272)
        if make or model:
            result["camera"] = " ".join(str(x) for x in [make, model] if x)
    except Exception:
        pass
    return result
