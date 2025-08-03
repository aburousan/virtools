"""Correction for artifacts like odd-even effect, spikes, and stripes."""
import numpy as np
import pywt
from .constants import FILTER_BANDS_VIS, defective_ir_pixels


def restore_sat_nan_no(reflectance_cube, flag_mask):
    final_reflectance = reflectance_cube.copy()
    final_reflectance[flag_mask == 1] = -32768
    final_reflectance[(flag_mask == 2) | (flag_mask == 3)] = -32767
    return final_reflectance


def get_filter_mask(nbands, filter_bands, edge_padding=0):
    """
    Create a boolean mask from a flat list of band indices.

    Parameters:
        nbands: int
        filter_bands: list of int (explicit band indices)
        edge_padding: int, expand around each index

    Returns:
        np.ndarray of shape (nbands,), dtype=bool
    """
    mask = np.zeros(nbands, dtype=bool)
    for b in filter_bands:
        for i in range(max(0, b - edge_padding), min(nbands, b + edge_padding + 1)):
            mask[i] = True
    return mask
def correct_odd_even_weighted_interp(spectrum, filter_band_mask=None):
    """
    Odd-even suppression using weighted neighbor smoothing, while skipping filter regions.
    Follows VIR_CALIBRATION_V3_1 guidelines (Section 4.4.1).
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    corrected = spectrum.copy()
    n = len(spectrum)
    if filter_band_mask is None:
        filter_band_mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if not np.isfinite(spectrum[i]):
            continue
        if filter_band_mask[i]:
            prev = next = None
            for j in range(i - 1, -1, -1):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    prev = (j, spectrum[j])
                    break
            for j in range(i + 1, n):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    next = (j, spectrum[j])
                    break
            if prev and next:
                x0, y0 = prev
                x1, y1 = next
                corrected[i] = y0 + (y1 - y0) * (i - x0) / (x1 - x0)
            continue
        if np.isfinite(spectrum[i - 1]) and np.isfinite(spectrum[i + 1]):
            corrected[i] = (
                0.3 * spectrum[i - 1] + 0.4 * spectrum[i] + 0.3 * spectrum[i + 1])
    return corrected

def correct_cube_odd_even_interp(cube, filter_band_mask=None):
    """
    Apply odd-even correction across a cube using weighted interpolation.

    Parameters:
        cube (np.ndarray): Shape (bands, samples, lines)
        filter_band_mask (np.ndarray): Boolean mask for bands to skip (length = bands)

    Returns:
        np.ndarray: Odd-even corrected cube (same shape)
    """
    bands, samples, lines = cube.shape
    corrected = np.empty_like(cube)
    for s in range(samples):
        for l in range(lines):
            spectrum = cube[:, s, l]
            corrected[:, s, l] = correct_odd_even_weighted_interp(spectrum, filter_band_mask)
    return corrected
def nan_running_avg(x, window=3):
    out = np.full_like(x, np.nan, dtype=float)
    half = window // 2
    for i in range(len(x)):
        i1 = max(0, i - half)
        i2 = min(len(x), i + half + 1)
        window_vals = x[i1:i2]
        valid = np.isfinite(window_vals)
        if valid.sum() > 0:
            out[i] = np.nanmean(window_vals)
    return out
def despike_from_doc(
    spectrum,
    sigma=3.0,
    runavg_window=3,
    polyfit_window=20,
    min_valid_points=5,
    filter_band_mask=None):
    spectrum = np.asarray(spectrum, dtype=np.float64)
    cleaned = spectrum.copy()
    n_bands = len(spectrum)
    if filter_band_mask is None:
        filter_band_mask = np.zeros(n_bands, dtype=bool)
    smoothed = nan_running_avg(spectrum, window=runavg_window)
    ratio = spectrum / (smoothed + 1e-10)
    z = (ratio - np.nanmean(ratio)) / (np.nanstd(ratio) + 1e-10)
    spikes = (np.abs(z) > sigma) & (~filter_band_mask)
    spike_indices = np.where(spikes)[0]
    if len(spike_indices) == 0:
        return cleaned 
    for idx in spike_indices:
        half = polyfit_window // 2
        i1 = max(0, idx - half)
        i2 = min(n_bands, idx + half + 1)
        x_win = np.arange(i1, i2)
        y_win = spectrum[i1:i2]
        mask_win = filter_band_mask[i1:i2]
        valid = np.isfinite(y_win) & (~mask_win)
        if valid.sum() >= min_valid_points:
            try:
                coeffs = np.polyfit(x_win[valid], y_win[valid], deg=2)
                cleaned[idx] = np.polyval(coeffs, idx)
            except Exception:
                pass  # fallback below
        elif valid.sum() >= 2:
            try:
                coeffs = np.polyfit(x_win[valid], y_win[valid], deg=1)
                cleaned[idx] = np.polyval(coeffs, idx)
            except Exception:
                pass
        else:
            prev, next = None, None
            for j in range(idx - 1, -1, -1):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    prev = (j, spectrum[j])
                    break
            for j in range(idx + 1, n_bands):
                if np.isfinite(spectrum[j]) and not filter_band_mask[j]:
                    next = (j, spectrum[j])
                    break
            if prev and next:
                x0, y0 = prev
                x1, y1 = next
                cleaned[idx] = y0 + (y1 - y0) * (idx - x0) / (x1 - x0)
            else:
                cleaned[idx] = np.nan
    return cleaned


def despike_cube_from_doc(
    cube,
    sigma=3.0,
    runavg_window=3,
    polyfit_window=20,
    min_valid_points=5,
    filter_band_mask=None):
    """
    Apply despiking across a cube using official VIR method.

    Parameters:
        cube (np.ndarray): Shape (bands, samples, lines)
        filter_band_mask (np.ndarray): Boolean mask for bands to skip (length = bands)

    Returns:
        np.ndarray: Despiked cube
    """
    bands, samples, lines = cube.shape
    cleaned_cube = np.empty_like(cube)
    for s in range(samples):
        for l in range(lines):
            spectrum = cube[:, s, l]
            cleaned_cube[:, s, l] = despike_from_doc(
                spectrum,
                sigma=sigma,
                runavg_window=runavg_window,
                polyfit_window=polyfit_window,
                min_valid_points=min_valid_points,
                filter_band_mask=filter_band_mask
            )
    return cleaned_cube

def destripe_wavelet_moment(image, wavelet='db4', level=2):
    """
    Destripe a 2D image using wavelet denoising of the row-wise mean.
    Ensures NO NaNs at output, even at boundaries.
    """
    destriped = image.copy()
    row_means = np.nanmean(image, axis=1)
    if np.any(np.isnan(row_means)):
        valid = ~np.isnan(row_means)
        row_means[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), row_means[valid])
    coeffs = pywt.wavedec(row_means, wavelet=wavelet, level=level, mode='periodization')
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(row_means)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    smoothed = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode='periodization')
    smoothed = smoothed[:image.shape[0]]
    for i in range(image.shape[0]):
        row = destriped[i]
        mu, std = np.nanmean(row), np.nanstd(row)
        if std > 1e-6:
            destriped[i] = (row - mu) / std * std + smoothed[i]
        else:
            destriped[i] = np.full_like(row, smoothed[i])
    destriped = np.nan_to_num(destriped, nan=0.0)
    return destriped
def destripe_cube(cube):
    bands, samples, lines = cube.shape
    destriped = np.empty_like(cube)
    for b in range(bands):
        destriped[b] = destripe_wavelet_moment(cube[b])
    destriped = np.nan_to_num(destriped, nan=0.0)
    return destriped