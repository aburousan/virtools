"""Radiometric and reflectance calibration routines."""
import numpy as np

def dn_to_radiance(dn_cube, itf, exposure_times):
    bands, samples, lines = dn_cube.shape
    assert exposure_times.shape[0] == lines
    radiance_cube = np.full_like(dn_cube, np.nan, dtype=np.float32)
    for l in range(lines):
        frame = dn_cube[:, :, l]
        expo = exposure_times[l]
        if np.isnan(expo) or expo <= 0:
            continue
        radiance_cube[:, :, l] = frame / (itf * expo)
    return radiance_cube

def radiance_to_dn(radiance_cube, itf, exposure_times):
    bands, samples, lines = radiance_cube.shape
    assert exposure_times.shape[0] == lines
    dn_cube = np.full_like(radiance_cube, np.nan, dtype=np.float32)
    for l in range(lines):
        frame = radiance_cube[:, :, l]
        expo = exposure_times[l]
        if np.isnan(expo) or expo <= 0:
            continue
        dn_cube[:, :, l] = frame * itf * expo
    return dn_cube

def reflectance(
    radiance_cube,
    solar_irradiance,
    spacecraft_solar_distance_km) -> np.ndarray:
    K = 149597870.7
    scale_factor = (np.pi * (spacecraft_solar_distance_km ** 2)) / (K ** 2)
    si = solar_irradiance[:, np.newaxis, np.newaxis]
    reflectance = (radiance_cube * scale_factor) / si
    return reflectance

def radiance_from_reflectance(
    reflectance_cube: np.ndarray,
    solar_irradiance: np.ndarray,
    spacecraft_solar_distance_km: float
) -> np.ndarray:
    K = 149597870.7  # Astronomical Unit in km
    scale_factor = (K ** 2) / (np.pi * (spacecraft_solar_distance_km ** 2))
    si = solar_irradiance[:, np.newaxis, np.newaxis]  # Shape: (bands, 1, 1)
    radiance = reflectance_cube * si * scale_factor
    return radiance
