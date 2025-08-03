import numpy as np

def compute_spectral_errorbars_mad(cube, mask_nan=True, axis=(1, 2)):
    """
    Compute 1-sigma spectral error bars using the Median Absolute Deviation (MAD)
    over spatial pixels for each spectral band.

    Parameters:
    - cube: ndarray, shape (bands, samples, lines) or (bands, height, width)
    - mask_nan: bool, whether to ignore NaNs
    - axis: tuple of int, the spatial axes to reduce over

    Returns:
    - mad: ndarray, shape (bands,), 1-sigma error estimate per band
    """
    if mask_nan:
        median = np.nanmedian(cube, axis=axis, keepdims=True)
        mad = np.nanmean(np.abs(cube - median), axis=axis)
    else:
        median = np.median(cube, axis=axis, keepdims=True)
        mad = np.mean(np.abs(cube - median), axis=axis)
    return mad

def conversion_DN_E(val, gain):
    """Convert DN to electrons using provided gain (e-/DN)."""
    return val * gain

def nansqrt(arr):
    """Safe square root: ignores NaNs and returns NaN where input is NaN."""
    return np.where(np.isnan(arr), np.nan, np.sqrt(arr))

def compute_readout_noise_bias_cube(bias_cube):
    """
    Compute readout noise from a bias cube with shape (bands, samples, frames).
    
    Parameters:
        bias_cube (ndarray): shape (bands, samples, N_frames)

    Returns:
        ron_map (2D ndarray): per-pixel readout noise (bands x samples)
        ron_mean (float): mean readout noise across all pixels
    """
    if bias_cube.shape[2] < 2:
        raise ValueError("Need at least 2 bias frames to compute readout noise")
    diff = bias_cube[:, :, 1:] - bias_cube[:, :, :-1]
    std_diff = np.nanstd(diff, axis=2)
    ron_map = std_diff / np.sqrt(2)
    ron_mean = np.nanmean(ron_map)
    return ron_map, ron_mean

def compute_snr(raw_cube, calibrated_cube, bias_cube, gain):
    """
    Compute SNR using Poisson noise model + readout noise.
    Parameters:
        raw_cube (ndarray): raw DN values (bands, samples, lines)
        calibrated_cube (ndarray): bias/dark-corrected DN values (same shape)
        bias_cube (ndarray): (bands, samples, N_bias_frames)
        gain (float): conversion factor from DN to electrons (e⁻/DN)
    
    Returns:
        snr (ndarray): same shape as input cubes
    """
    ron_map, ron_mean = compute_readout_noise_bias_cube(bias_cube)
    signal_e = conversion_DN_E(calibrated_cube, gain)
    readout_e = conversion_DN_E(ron_mean, gain)
    noise_e = nansqrt(signal_e + readout_e**2)
    return signal_e / noise_e
