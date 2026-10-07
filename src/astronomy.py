from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_utc(dt: datetime, offset_hours: float) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    return (dt - timedelta(hours=offset_hours)).replace(tzinfo=timezone.utc)


def star_altaz(catalog: pd.DataFrame, dt_utc: datetime, lat: float, lon: float) -> np.ndarray:
    """Return [azimuth_deg, altitude_deg] for each star using Astropy.

    Astropy is intentionally kept inside this function so the rest of the project can still
    run its image-processing components when astronomy packages are not yet installed.
    """
    from astropy import units as u
    from astropy.utils import iers
    iers.conf.auto_download = False
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    location = EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=0 * u.m)
    obstime = Time(dt_utc)
    coords = SkyCoord(
        ra=np.asarray(catalog["ra_h"], dtype=float) * u.hourangle,
        dec=np.asarray(catalog["dec_deg"], dtype=float) * u.deg,
        frame="icrs",
    )
    altaz = coords.transform_to(AltAz(obstime=obstime, location=location))
    return np.column_stack([altaz.az.deg, altaz.alt.deg])


def planet_altaz(dt_utc: datetime, lat: float, lon: float) -> dict[str, tuple[float, float]]:
    """Return planet azimuth/altitude from Astropy's built-in solar-system ephemeris."""
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time

    location = EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=0 * u.m)
    obstime = Time(dt_utc)
    frame = AltAz(obstime=obstime, location=location)
    names = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    result: dict[str, tuple[float, float]] = {}
    for name in names:
        try:
            body = get_body(name, obstime, location=location).transform_to(frame)
            result[name.title()] = (float(body.az.deg), float(body.alt.deg))
        except Exception:
            continue
    return result


def altaz_to_vector(az_deg: np.ndarray, alt_deg: np.ndarray) -> np.ndarray:
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    x = np.cos(alt) * np.sin(az)  # east
    y = np.cos(alt) * np.cos(az)  # north
    z = np.sin(alt)                # up
    return np.column_stack([x, y, z])
