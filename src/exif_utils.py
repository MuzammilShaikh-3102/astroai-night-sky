from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from math import atan, degrees
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


def _text(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip() if value is not None else None


def _dms_to_deg(values: Any) -> float:
    d, m, s = (_rational_to_float(v) for v in values)
    return d + m / 60.0 + s / 3600.0


def _parse_gps(gps: dict[Any, Any]) -> tuple[float, float] | None:
    lat, lon = gps.get(2), gps.get(4)
    if lat is None or lon is None:
        return None
    lat_deg, lon_deg = _dms_to_deg(lat), _dms_to_deg(lon)
    lat_ref, lon_ref = _text(gps.get(1)), _text(gps.get(3))
    if lat_ref and lat_ref.upper() == "S":
        lat_deg *= -1
    if lon_ref and lon_ref.upper() == "W":
        lon_deg *= -1
    if not (-90 <= lat_deg <= 90 and -180 <= lon_deg <= 180):
        return None
    return lat_deg, lon_deg


def _parse_utc_offset(value: Any) -> float | None:
    value = _text(value)
    if not value:
        return None
    try:
        sign = -1 if value.startswith("-") else 1
        hours, minutes = value[1:].split(":")
        h, m = int(hours), int(minutes)
        if h > 14 or m not in (0, 15, 30, 45):
            return None
        return sign * (h + m / 60.0)
    except (ValueError, IndexError):
        return None


def _parse_gps_datetime(gps_ifd: dict[Any, Any]) -> datetime | None:
    """GPSDateStamp + GPSTimeStamp are UTC by EXIF specification."""
    try:
        raw_date = _text(gps_ifd.get(29))
        raw_time = gps_ifd.get(7)
        if not raw_date or raw_time is None:
            return None
        day = datetime.strptime(raw_date, "%Y:%m:%d").date()
        hour, minute, second = (_rational_to_float(v) for v in raw_time)
        stamp = datetime.combine(day, time()) + timedelta(
            hours=hour, minutes=minute, seconds=second
        )
        return stamp.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def extract_metadata(file_bytes: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gps": None,
        "datetime": None,
        "raw_datetime": None,
        "utc_offset_hours": None,
        "gps_datetime_utc": None,
        "camera": None,
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "estimated_hfov_deg": None,
        "image_width": None,
        "image_height": None,
        "has_exif": False,
        "metadata_warnings": [],
    }
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            result["image_width"], result["image_height"] = image.size
            exif = image.getexif()
            if not exif:
                result["metadata_warnings"].append("This image contains no readable EXIF metadata.")
                return result
            result["has_exif"] = True
            try:
                exif_ifd = exif.get_ifd(34665)
            except Exception:
                exif_ifd = {}
            try:
                gps_ifd = exif.get_ifd(34853)
            except Exception:
                gps_ifd = {}

            result["gps"] = _parse_gps(gps_ifd)
            result["gps_datetime_utc"] = _parse_gps_datetime(gps_ifd)

            raw_dt = exif_ifd.get(36867) or exif.get(36867) or exif.get(306)
            result["raw_datetime"] = _text(raw_dt)
            if raw_dt:
                try:
                    result["datetime"] = datetime.strptime(
                        str(result["raw_datetime"]), "%Y:%m:%d %H:%M:%S"
                    )
                except ValueError:
                    result["metadata_warnings"].append("The EXIF capture timestamp could not be parsed.")

            # EXIF tag 36881 is OffsetTimeOriginal, e.g. -06:00 or +05:30.
            offset = exif_ifd.get(36881) or exif.get(36881)
            result["utc_offset_hours"] = _parse_utc_offset(offset)

            make = _text(exif.get(271))
            model = _text(exif.get(272))
            if make or model:
                result["camera"] = " ".join(x for x in (make, model) if x)

            focal = exif_ifd.get(37386) or exif.get(37386)
            focal_35 = exif_ifd.get(41989) or exif.get(41989)
            if focal is not None:
                result["focal_length_mm"] = _rational_to_float(focal)
            if focal_35 is not None:
                result["focal_length_35mm"] = _rational_to_float(focal_35)
                if result["focal_length_35mm"] > 0:
                    # 36 mm is the nominal 35-mm still-camera image width.
                    result["estimated_hfov_deg"] = degrees(
                        2.0 * atan(36.0 / (2.0 * result["focal_length_35mm"]))
                    )

            if result["gps"] is None:
                result["metadata_warnings"].append(
                    "No GPS coordinates were embedded. Enter the location where the photo was captured."
                )
            if result["gps_datetime_utc"] is None and result["datetime"] is None:
                result["metadata_warnings"].append(
                    "No capture timestamp was found. Enter the original capture date and time."
                )
    except Exception as exc:
        result["metadata_warnings"].append(f"Could not read image metadata: {exc}")
    return result
