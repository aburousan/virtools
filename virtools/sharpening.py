"""Image sharpening using star-based PSF."""

import numpy as np
from skimage.restoration import richardson_lucy
from scipy.optimize import curve_fit
from .constants import FILTER_BANDS_VIS, defective_ir_pixels


def sharpen_science_cube_with_star_psf(science_cube, star_cube, band_idx=None,
                                       star_position=None, patch_size=15, iterations=20):
    """
    Automatically estimate PSF from a star_cube and use it to sharpen a science_cube.
    
    Parameters:
    - science_cube: numpy array of shape (band, sample, line)
    - star_cube:    numpy array of same shape (band, sample, line)
    - band_idx:     band index for PSF extraction (if None, uses band with max signal)
    - star_position: (sample, line) of the star center. If None, auto-detects brightest.
    - patch_size:   size of square patch to crop around the star
    - iterations:   number of iterations for Richardson-Lucy
    
    Returns:
    - sharpened_cube: numpy array of shape (band, sample, line)
    - star_patch:     2D numpy array of shape (patch_size, patch_size) cropped around the star
    - band_idx:       The band index used for PSF estimation
    """
    def two_d_gaussian(xy, x0, y0, sigma_x, sigma_y, amplitude, offset):
        x, y = xy
        g = amplitude * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) +
                                 ((y - y0)**2) / (2 * sigma_y**2))) + offset
        return g.ravel()
    def fit_psf(star_patch):
        star_patch = np.nan_to_num(star_patch)
        y, x = np.indices(star_patch.shape)
        max_idx = np.unravel_index(np.argmax(star_patch), star_patch.shape)
        x0_guess, y0_guess = max_idx[1], max_idx[0]
        
        initial_guess = (
            x0_guess, y0_guess,
            2, 2,
            np.max(star_patch),
            np.min(star_patch))
        bounds = (
            [0, 0, 0.5, 0.5, 0, -np.inf],
            [star_patch.shape[1], star_patch.shape[0], 5, 5, np.inf, np.inf])
        try:
            popt, _ = curve_fit(two_d_gaussian, (x, y), star_patch.ravel(),
                                p0=initial_guess, bounds=bounds, maxfev=5000)
        except RuntimeError:
            raise RuntimeError("PSF fit failed. Try a cleaner star patch or better guess.")
        return popt
    def create_psf_kernel(size, sigma_x, sigma_y):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-((xx**2) / (2. * sigma_x**2) + (yy**2) / (2. * sigma_y**2)))
        return kernel / np.sum(kernel)
    if band_idx is None:
        summed = np.sum(star_cube, axis=(1,2))  # Sum over spatial axes
        band_idx = np.argmax(summed)
    star_image = star_cube[band_idx]  # shape: (sample, line)
    if star_position is None:
        star_position = np.unravel_index(np.nanargmax(star_image), star_image.shape)
    sample_c, line_c = star_position
    half = patch_size // 2
    star_patch = star_image[
        max(sample_c - half, 0):sample_c + half + 1,
        max(line_c - half, 0):line_c + half + 1]
    if star_patch.shape[0] < patch_size or star_patch.shape[1] < patch_size:
        raise ValueError("Patch size too large or star too close to the edge.")
    x0, y0, sigma_x, sigma_y, amp, offset = fit_psf(star_patch)
    psf_kernel = create_psf_kernel(size=patch_size, sigma_x=sigma_x, sigma_y=sigma_y)
    def fill_nan(image, fill_value=0.0):
        return np.nan_to_num(image, nan=fill_value, posinf=fill_value, neginf=fill_value)
    sharpened_cube = np.empty_like(science_cube)
    for b in range(science_cube.shape[0]):
        image = fill_nan(science_cube[b])
        sharpened_cube[b] = richardson_lucy(image, psf_kernel, num_iter=iterations)
    return sharpened_cube, star_patch, band_idx