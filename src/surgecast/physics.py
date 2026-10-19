"""Physical forcing terms for storm surge.

Storm surge is mostly the ocean's response to two things: low atmospheric
pressure lifting the sea surface (inverse barometer) and wind stress piling
water against the coast (wind setup). These are turned into model inputs here
instead of handing the network raw pressure and wind.
"""

from __future__ import annotations

import numpy as np

RHO_WATER = 1025.0  # kg m^-3, sea water
RHO_AIR = 1.225  # kg m^-3
G = 9.81  # m s^-2
P_REF = 1013.25  # hPa
EARTH_RADIUS_KM = 6371.0


def inverse_barometer(pressure_hpa: np.ndarray) -> np.ndarray:
    """Static sea-level response to pressure, in metres (~1 cm per hPa).

    Hydrostatic balance: a drop of dp Pa raises the surface by dp / (rho g).
    """
    return -(pressure_hpa - P_REF) * 100.0 / (RHO_WATER * G)


def drag_coefficient(speed: np.ndarray) -> np.ndarray:
    """Large & Pond (1981) neutral drag coefficient.

    Capped at 33 m/s: the drag coefficient saturates in hurricane-force winds
    (Powell et al., 2003), and the linear fit keeps growing past that.
    """
    u = np.minimum(speed, 33.0)
    return np.where(u < 11.0, 1.2e-3, (0.49 + 0.065 * u) * 1e-3)


def wind_components(speed: np.ndarray, direction_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eastward/northward wind from speed and meteorological direction (blowing FROM)."""
    rad = np.deg2rad(direction_from_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def wind_stress(speed: np.ndarray, direction_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface wind stress (N m^-2) from the quadratic drag law tau = rho_a C_d |U| U."""
    u, v = wind_components(speed, direction_from_deg)
    k = RHO_AIR * drag_coefficient(speed) * speed
    return k * u, k * v


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = p2 - p1
    dlon = np.deg2rad(lon2) - np.deg2rad(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing (radians, clockwise from north) from point 1 towards point 2."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon2) - np.deg2rad(lon1)
    x = np.sin(dlon) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlon)
    return np.arctan2(x, y)
