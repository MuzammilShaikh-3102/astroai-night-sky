from datetime import datetime, timezone

from src.exif_utils import _parse_gps, _parse_gps_datetime, _parse_utc_offset


def test_parse_gps_applies_south_and_west_signs():
    gps = {
        1: "S", 2: (10.0, 30.0, 0.0),
        3: "W", 4: (20.0, 15.0, 0.0),
    }
    lat, lon = _parse_gps(gps)
    assert abs(lat - (-10.5)) < 1e-9
    assert abs(lon - (-20.25)) < 1e-9


def test_parse_gps_rejects_out_of_range_coordinates():
    assert _parse_gps({1: "N", 2: (91, 0, 0), 3: "E", 4: (10, 0, 0)}) is None


def test_parse_exif_utc_offsets():
    assert _parse_utc_offset("-06:00") == -6.0
    assert _parse_utc_offset("+05:30") == 5.5
    assert _parse_utc_offset("+05:45") == 5.75
    assert _parse_utc_offset("not-an-offset") is None


def test_gps_timestamp_is_utc():
    stamp = _parse_gps_datetime({
        29: "2026:11:17",
        7: (2.0, 30.0, 15.0),
    })
    assert stamp == datetime(2026, 11, 17, 2, 30, 15, tzinfo=timezone.utc)
