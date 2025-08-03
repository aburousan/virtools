"""Spectral and pixel interpolation tools."""
import numpy as np
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

def fill_nan_with_spline(spectrum, lambda_IR, s=None):
    """
    Fill NaNs in a 1D spectrum using smoothing spline interpolation.

    Parameters:
    - spectrum: 1D numpy array of spectral values
    - lambda_IR: 1D numpy array of wavelengths (same length as spectrum)
    - s: optional smoothing parameter; if None, it is estimated from variance

    Returns:
    - A new spectrum with NaNs filled
    """
    spectrum = spectrum.copy()
    nan_mask = ~np.isfinite(spectrum)
    if not np.any(nan_mask):
        return spectrum
    valid = np.isfinite(spectrum)
    if np.sum(valid) < 4:
        return spectrum
    x_valid = lambda_IR[valid]
    y_valid = spectrum[valid]

    # Estimate smoothing if not provided
    if s is None:
        s = 0.01 * np.nanvar(y_valid) * len(y_valid)

    try:
        spline = UnivariateSpline(x_valid, y_valid, s=s)
    except Exception:
        try:
            spline = InterpolatedUnivariateSpline(x_valid, y_valid)
        except Exception:
            return spectrum  # fallback: return unchanged

    try:
        spectrum[nan_mask] = spline(lambda_IR[nan_mask])
    except Exception:
        pass  # avoid crash, keep NaNs

    return spectrum


def fill_cube_nan_sp(cube, lambda_IR, spara=None):
    """
    Interpolate NaNs in a spectral cube across the band axis using spline interpolation.

    Parameters:
    - cube: 3D numpy array, shape (bands, samples, lines)
    - lambda_IR: 1D array of wavelengths (same length as number of bands)
    - spara: optional smoothing parameter for spline

    Returns:
    - filled cube with NaNs interpolated
    """
    bands, samples, lines = cube.shape
    filled = cube.copy()
    for s in range(samples):
        for l in range(lines):
            filled[:, s, l] = fill_nan_with_spline(filled[:, s, l], lambda_IR, s=spara)
    return filled


def fix_defective_pixels_with_spline(cube, defective_coords, lambda_IR, filter_band_list=None, spara=None):
    """
    Replace defective pixels with NaN and interpolate them using spectral spline interpolation.

    Parameters:
    - cube: 3D numpy array, shape (bands, samples, lines)
    - defective_coords: list of tuples (sample_id, band_id), 1-based indexing
    - lambda_IR: 1D numpy array of wavelengths (length = number of bands)
    - filter_band_list: optional list of 1-based band IDs to exclude from interpolation
    - spara: optional smoothing parameter for spline interpolation

    Returns:
    - corrected cube with defective pixels filled
    """
    corrected = cube.copy()
    bands, samples, lines = cube.shape
    mask = np.zeros((samples, bands), dtype=bool)  # (samples, bands)

    for sample_id, band_id in defective_coords:
        sample_idx = sample_id - 1
        band_idx = band_id - 1
        if filter_band_list is None or band_id not in filter_band_list:
            mask[sample_idx, band_idx] = True

    for sample_idx in range(samples):
        for band_idx in range(bands):
            if mask[sample_idx, band_idx]:
                corrected[band_idx, sample_idx, :] = np.nan

    corrected = fill_cube_nan_sp(corrected, lambda_IR, spara=spara)
    return corrected

def fit_linear_dispersion(wavelengths):
    """
    Fit a linear relation lambda = a*band + b to the given wavelength array.
    Returns (slope, intercept).
    """
    bands = np.arange(1, len(wavelengths) + 1)
    def wv(x, a, b):
        return a * x + b
    para,poc = curve_fit(wv, bands, wavelengths)
    return para,poc

def restore_sat_nan_no(reflectance_cube, flag_mask):
    final_reflectance = reflectance_cube.copy()
    final_reflectance[flag_mask == 1] = -32768
    final_reflectance[(flag_mask == 2) | (flag_mask == 3)] = -32767
    return final_reflectance

def correct_nan_pixels_doc(refl_cube: np.ndarray, lambda_IR: np.ndarray) -> np.ndarray:
    """
    Match the exact logic from IDL Step 1 in the VIR calibration doc:
    only interpolate over NaNs *between* non-continuous segments in right_channels.
    """
    bands, samples, lines = refl_cube.shape
    corrected_cube = refl_cube.copy()
    for s in range(samples):
        for l in range(lines):
            spectrum = corrected_cube[:, s, l]
            nan_mask = ~np.isfinite(spectrum)
            valid_indices = np.where(np.isfinite(spectrum))[0]
            if len(valid_indices) < 21:
                continue  # Not enough data to fit
            for d in range(10, len(valid_indices) - 11):
                current = valid_indices[d]
                next_ = valid_indices[d + 1]

                if next_ != current + 1:
                    fit_range = valid_indices[d - 10 : d + 11]  # 21 points
                    x_fit = lambda_IR[fit_range]
                    y_fit = spectrum[fit_range]
                    nan_range = np.arange(valid_indices[d - 10], valid_indices[d + 1 + 10])
                    nan_range = nan_range[(nan_range > current) & (nan_range < next_)]
                    nan_to_fill = nan_range[nan_mask[nan_range]]
                    if len(nan_to_fill) == 0:
                        continue
                    poly = Polynomial.fit(x_fit, y_fit, deg=2).convert()
                    spectrum[nan_to_fill] = poly(lambda_IR[nan_to_fill])
            corrected_cube[:, s, l] = spectrum
    return corrected_cube