"""Dark Current model and predict routines."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

planck_constant = 6.62607015e-34
speed_of_light = 2.99792458e8
boltzmann_constant = 1.380649e-23

def predict_dark_current_doc(raw_cube, closed_indexes, apply_smoothing=False):
    """
    Predict the dark-current signal for each pixel (band,sample,line).

    Parameters:
        raw_cube (ndarray): Raw data cube of shape (bands, samples, lines).
        closed_indexes (list of int): Line indices where dark frames (shutter closed) occur.
        temps (ndarray): Array of length `lines` with detector temperature (K) at each line.
        exp_times (ndarray): Array of length `lines` with exposure time (sec) at each line.
        model_type (str): 'arrhenius', 'poly2', or 'dawn' (default 'arrhenius').
        apply_smoothing (bool): If True, apply Gaussian smoothing across samples on each dark frame.

    Returns:
        predicted_dark_cube (ndarray): Dark-current cube of shape (bands, samples, lines).
    """
    bands, samples, lines = raw_cube.shape
    closed = sorted(closed_indexes)
    if len(closed) == 0:
        raise ValueError("No closed (dark) frames provided.")
    dark_frames = np.array(raw_cube[..., closed], dtype=float)
    if apply_smoothing:
        for i in range(dark_frames.shape[2]):
            for b in range(bands):
                dark_frames[b, :, i] = gaussian_filter1d(dark_frames[b, :, i], sigma=1)
    if len(closed) == 1:
        dark_ref = dark_frames[:, :, 0]
        predicted = np.tile(dark_ref[:, :, np.newaxis], (1, 1, lines))
        return predicted
    predicted_dark = np.zeros((bands, samples, lines), dtype=float)
    line_idx = np.arange(lines)
    known_idx = np.array(closed)
    for b in range(bands):
        for s in range(samples):
            known_vals = dark_frames[b, s, :]
            predicted_dark[b, s, :] = np.interp(line_idx, known_idx, known_vals)
    return predicted_dark


def fit_dark_thermal_model(cube, exposure_times, ir_temperatures, spec_temperatures,
                           closed_indices, wavelengths, itf):
    """
    Fit a per-pixel dark current + thermal background model for a VIR IR cube.

    Parameters
    ----------
    cube : np.ndarray, shape (bands, samples, lines)
        Cube with dark + bias frames. First 5 lines are bias frames.
    exposure_times : np.ndarray, shape (lines,)
        Integration times for each line.
    ir_temperatures : np.ndarray, shape (lines,)
        Detector IR temperature for each line.
    spec_temperatures : np.ndarray, shape (lines,)
        Spectrometer (optics) temperature for Planck term.
    closed_indices : list[int]
        Line indices of the dark frames.
    wavelengths : np.ndarray, shape (bands,)
        Wavelength centers in microns.
    itf : np.ndarray, shape (bands, samples)
        Instrument transfer function.

    Returns
    -------
    a_map, b_map, c_map : np.ndarray, shape (bands, samples)
        Fitted parameter maps for exp and thermal model.
    """
    bands, samples, lines = cube.shape
    closed_indices = np.asarray(closed_indices)
    n_bias = 5
    n_dark = len(closed_indices)

    T_dark = np.asarray(ir_temperatures)[closed_indices]
    T_spec = np.asarray(spec_temperatures)[closed_indices]
    t_int = np.asarray(exposure_times)[closed_indices]

    a_map = np.full((bands, samples), np.nan)
    b_map = np.full((bands, samples), np.nan)
    c_map = np.full((bands, samples), np.nan)

    def planck_radiance(wavelength_um, temperature):
        wavelength_m = wavelength_um * 1e-6
        numerator = 2 * planck_constant * speed_of_light**2 / wavelength_m**5
        exponent = planck_constant * speed_of_light / (wavelength_m * boltzmann_constant * temperature)
        return numerator / (np.exp(exponent) - 1)

    def model_residuals(params, T_dark, thermal_term, observed_dn):
        a, b, c = params
        model = np.exp(a / T_dark + b) + c * thermal_term
        return model - observed_dn

    bias_cube = cube[:, :, :n_bias]

    successful_fits = 0

    for b in range(bands):
        λ = wavelengths[b]
        try:
            planck_vals = planck_radiance(λ, T_spec)
        except FloatingPointError:
            continue

        for s in range(samples):
            itf_val = itf[b, s]
            thermal_term = planck_vals * t_int * itf_val

            dark_vals = np.zeros(n_dark)
            for i in range(n_dark):
                bias_line = i % n_bias
                dark_vals[i] = cube[b, s, closed_indices[i]] - bias_cube[b, s, bias_line]

            if not np.all(np.isfinite(dark_vals)) or np.any(thermal_term <= 0):
                continue

            try:
                result = least_squares(
                    model_residuals,
                    x0=[-1000, 0, 1],
                    bounds=([-1e4, -100, 0], [0, 100, 1e4]),
                    args=(T_dark, thermal_term, dark_vals),
                    loss='soft_l1',
                    max_nfev=5000
                )
                a_fit, b_fit, c_fit = result.x
                if result.success and a_fit < 0 and c_fit >= 0:
                    a_map[b, s] = a_fit
                    b_map[b, s] = b_fit
                    c_map[b, s] = c_fit
                    successful_fits += 1
            except Exception:
                continue
    print(f"Total successful fits: {successful_fits} / {bands * samples}")
    return a_map, b_map, c_map

def predict_dark_cube_from_model(a_map, b_map, c_map, exposure_times, ir_temperatures, spectral_temperatures, wavelengths, itf):
    """
    Predict dark current cube from model parameters (a, b, c), using separate temperatures
    for the dark current and thermal background components.

    Parameters:
        a_map, b_map, c_map      : (bands, samples) arrays from model fit
        exposure_times           : (lines,) array of integration times (seconds)
        ir_temperatures          : (lines,) array of IR detector temperatures (K) [used in exp(a/T + b)]
        spectral_temperatures    : (lines,) array of spectrometer temperatures (K) [used in Planck]
        wavelengths              : (bands,) center wavelengths (microns)
        itf                      : (bands, samples) Instrument Transfer Function (DN/e⁻)

    Returns:
        dark_cube_predicted : (bands, samples, lines) array with predicted dark current values
    """
    bands, samples = a_map.shape
    lines = len(exposure_times)
    dark_cube = np.full((bands, samples, lines), np.nan)
    T_dark = np.asarray(ir_temperatures)
    T_spec = np.asarray(spectral_temperatures)
    t_line = np.asarray(exposure_times)
    for b in range(bands):
        wavelength_m = wavelengths[b] * 1e-6
        factor1 = 2 * planck_constant * speed_of_light**2 / wavelength_m**5
        exponent = planck_constant * speed_of_light / (wavelength_m * boltzmann_constant * T_spec)
        B_lambda_Tspec = factor1 / (np.exp(exponent) - 1)
        for s in range(samples):
            a = a_map[b, s]
            b_ = b_map[b, s]
            c_ = c_map[b, s]
            itf_val = itf[b, s]
            if not np.isfinite(a) or not np.isfinite(c_):
                continue
            exp_term = np.exp(a / T_dark + b_)
            thermal_term = c_ * B_lambda_Tspec * t_line * itf_val
            dark_cube[b, s, :] = exp_term + thermal_term
    return dark_cube

def remove_closed_shutter_frames(raw_cube, closed_indexes):
    """
    Efficiently remove closed-shutter frames from predicted dark cube.

    Parameters:
        predicted_dark_cube (ndarray): (bands, samples, lines)
        closed_indexes (list of int): indexes of closed shutter frames

    Returns:
        dark_cube_open (ndarray): dark current for open frames only
        open_indexes (ndarray): indexes of open shutter frames
    """
    bands, samples, lines = raw_cube.shape
    mask = np.ones(lines, dtype=bool)
    mask[closed_indexes] = False
    dark_cube_open = raw_cube[:, :, mask]
    open_indexes = np.flatnonzero(mask)
    return dark_cube_open, open_indexes

def remove_indices_from_list(data_list, remove_indexes):
    """
    Remove elements from data_list at positions specified in remove_indexes.

    Parameters:
        data_list (list or ndarray): 1D list of data
        remove_indexes (list or ndarray): indexes to remove

    Returns:
        filtered_list (ndarray): data_list with elements at remove_indexes removed
    """
    data_array = np.asarray(data_list)
    mask = np.ones(data_array.shape[0], dtype=bool)
    mask[remove_indexes] = False
    filtered_list = data_array[mask]
    return filtered_list