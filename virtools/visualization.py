"""Visualization utilities for bands and spectra."""
from matplotlib.colors import Normalize, to_rgba
import numpy as np
import matplotlib.pyplot as plt
from .constants import FILTER_BANDS_VIS, defective_ir_pixels

def show_band_image(cube, band_index, cmap_val="gray", b_to_wv_um=None, stretch=False,stretch_amount= [10, 96],
                    drop_sample_ranges=None, drop_line_ranges=None, title="", zoom_region=None, cbar_label="Reflectance",save=False):
    """
    Show a single band image from the cube with optional zoom and sample/line drop.

    Parameters:
        cube : np.ndarray
            Data cube of shape (bands, samples, lines)
        band_index : int
            Index of the band to display (1-based)
        cmap_val : str
            Colormap used for plotting
        b_to_wv_um : function
            Function that maps band index to wavelength in microns
        stretch : bool
            Whether to stretch contrast between 2nd and 98th percentile (but keep colorbar in real units)
        drop_sample_ranges : list of tuple(int, int)
            List of (start_sample, end_sample) to drop from the image (sample axis)
        drop_line_ranges : list of tuple(int, int)
            List of (start_line, end_line) to drop from the image (line axis)
        title : str
            Custom title string
        zoom_region : tuple(int, int, int, int)
            (sample_start, sample_end, line_start, line_end) to crop image region
        save : bool
            Whether to save the image to a PDF file
    """
    img = cube[band_index - 1].copy()
    if drop_sample_ranges:
        sample_mask = np.ones(img.shape[0], dtype=bool)
        for start, end in drop_sample_ranges:
            sample_mask[start:end] = False
        img = img[sample_mask, :]
    if drop_line_ranges:
        line_mask = np.ones(img.shape[1], dtype=bool)
        for start, end in drop_line_ranges:
            line_mask[start:end] = False
        img = img[:, line_mask]
    if zoom_region:
        s0, s1, l0, l1 = zoom_region
        img = img[s0:s1, l0:l1]
    if stretch:
        vmin, vmax = np.nanpercentile(img, stretch_amount)
    else:
        vmin, vmax = np.nanmin(img), np.nanmax(img)
    plt.imshow(img.T, cmap=cmap_val, origin='lower', aspect='auto',
               norm=Normalize(vmin=vmin, vmax=vmax))
    if b_to_wv_um is not None:
        band_um = b_to_wv_um(band_index)
        plt.title(f"Band {band_index} - {band_um:.4f} µm {title}")
    else:
        plt.title(f"Band {band_index} {title}")
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.xlabel("Sample")
    plt.ylabel("Line")
    if save:
        fname = title if title else f"band_{band_index}"
        plt.savefig(f"{fname}.pdf", bbox_inches='tight')
    plt.show()

def plot_superimposed_bands(
    cube,
    bands,
    colors,
    stretch=True,
    alpha=0.6,
    zoom_region=None,
    drop_sample_ranges=None,
    drop_line_ranges=None,
    title="",
    save=False):
    """
    Superimpose selected bands from a cube using specified colors.

    Parameters:
        cube : np.ndarray
            Data cube of shape (bands, samples, lines)
        bands : list of int
            Band indices to visualize (1-based)
        colors : list of str
            Matplotlib color names (e.g., "red", "green", "blue", etc.)
        stretch : bool
            Whether to apply contrast stretching (2-98 percentile)
        alpha : float
            Transparency for overlays
        zoom_region : tuple(int, int, int, int)
            (sample_start, sample_end, line_start, line_end)
        drop_sample_ranges : list of tuple(int, int)
            List of sample ranges to mask (columns)
        drop_line_ranges : list of tuple(int, int)
            List of line ranges to mask (rows)
        title : str
            Plot title
        save : bool
            Save to PDF
    """
    assert len(bands) == len(colors), "bands and colors must be same length"
    bands = [b - 1 for b in bands]
    ref_img = cube[bands[0]].copy()
    if drop_sample_ranges:
        sample_mask = np.ones(ref_img.shape[0], dtype=bool)
        for start, end in drop_sample_ranges:
            sample_mask[start:end] = False
        ref_img = ref_img[sample_mask, :]
    else:
        sample_mask = np.ones(ref_img.shape[0], dtype=bool)
    if drop_line_ranges:
        line_mask = np.ones(ref_img.shape[1], dtype=bool)
        for start, end in drop_line_ranges:
            line_mask[start:end] = False
        ref_img = ref_img[:, line_mask]
    else:
        line_mask = np.ones(ref_img.shape[1], dtype=bool)
    if zoom_region:
        s0, s1, l0, l1 = zoom_region
        ref_img = ref_img[s0:s1, l0:l1]
        sample_mask[s0:s1] = sample_mask[s0:s1]  # reduce to zoom
        line_mask[l0:l1] = line_mask[l0:l1]
    shape = ref_img.shape
    composite = np.zeros((shape[0], shape[1], 4))
    for band, color in zip(bands, colors):
        img = cube[band].copy()
        img = img[sample_mask, :]
        img = img[:, line_mask]
        if zoom_region:
            s0, s1, l0, l1 = zoom_region
            img = img[s0:s1, l0:l1]
        if stretch:
            p2, p98 = np.nanpercentile(img, [2, 98])
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
        else:
            min_val, max_val = np.nanmin(img), np.nanmax(img)
            img = np.clip((img - min_val) / (max_val - min_val + 1e-10), 0, 1)
        rgba = to_rgba(color, alpha=alpha)
        for c in range(3):
            composite[..., c] += img * rgba[c]
        composite[..., 3] = 1.0
    composite[..., :3] = np.clip(composite[..., :3], 0, 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(composite.transpose(1, 0, 2), origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Line")
    if save:
        fname = title if title else "superimposed_bands"
        plt.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.show()
