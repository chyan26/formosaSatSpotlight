"""Reusable helpers for the NCU Spotlight PAN1/PAN2 alignment pipeline.

This module is meant to keep ncu-spots.ipynb clean: it holds the image
pre-processing, alignment, stacking, source detection, PSF fitting and
plotting utilities used by both the hybrid (SEP + phase) and the
correlation-only workflows.

Optional dependencies
---------------------
``convert_tiff_to_fits()`` requires ``tifffile``. The rest of the module does
not depend on it, so ``tifffile`` is imported lazily inside that function.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import sep
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

# ---------------------------------------------------------------------------
# Workspace / path helpers
# ---------------------------------------------------------------------------

# Workspace root used when the notebook is run from a different CWD.
# This mirrors the original notebook's hard-coded path for backward
# compatibility. It is intentionally overridable: callers can pass explicit
# ``src_root`` / ``dst_root`` paths, or set the ``NCU_SPOTLIGHT_ROOT``
# environment variable before importing this module.
WORKSPACE_ROOT = Path(
    os.environ.get("NCU_SPOTLIGHT_ROOT", "/Users/chyan/Desktop/NCU Spotlight")
)


def first_existing(candidates: list[Path]) -> Path:
    """Return the first path in ``candidates`` that exists.

    If none of the candidates exist, the first candidate is returned as the
    fallback (the caller can then create it or raise an error as needed).
    """
    return next((p for p in candidates if p.exists()), candidates[0])


def get_roots() -> tuple[Path, Path]:
    """Resolve the FITS input folder and the stacked-output folder.

    Tries several plausible working-directory layouts so the same code works
    whether the notebook is launched from the workspace root or from the
    ``NCU Spotlight`` project folder.
    """
    fits_root = first_existing(
        [
            WORKSPACE_ROOT / "fits_outputs",
            WORKSPACE_ROOT / "NCU Spotlight" / "fits_outputs",
            Path.cwd() / "fits_outputs",
            Path.cwd() / "NCU Spotlight" / "fits_outputs",
        ]
    )
    stack_root = first_existing(
        [
            WORKSPACE_ROOT / "fits_stacked",
            WORKSPACE_ROOT / "NCU Spotlight" / "fits_stacked",
            Path.cwd() / "fits_stacked",
            Path.cwd() / "NCU Spotlight" / "fits_stacked",
        ]
    )
    stack_root.mkdir(parents=True, exist_ok=True)
    return fits_root, stack_root


# ---------------------------------------------------------------------------
# Image cropping / display helpers
# ---------------------------------------------------------------------------


def center_crop(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Crop a 2D array to ``(ny, nx)`` around its center."""
    y0 = max(0, (arr.shape[0] - ny) // 2)
    x0 = max(0, (arr.shape[1] - nx) // 2)
    return arr[y0 : y0 + ny, x0 : x0 + nx]


def crop_pair_to_common_center(
    data1: np.ndarray, data2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop both frames of a pair down to their common (smallest) shape."""
    ny = min(data1.shape[0], data2.shape[0])
    nx = min(data1.shape[1], data2.shape[1])
    return center_crop(data1, ny, nx), center_crop(data2, ny, nx)


def robust_limits(arr: np.ndarray) -> tuple[float, float]:
    """Return ``(vmin, vmax)`` display limits using 5th/99.5th percentiles."""
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) < 10:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(arr[finite], [5, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    return float(vmin), float(vmax)


# Alias kept for backward compatibility with the original notebook names.
image_limits = robust_limits


# ---------------------------------------------------------------------------
# Source detection and catalog matching
# ---------------------------------------------------------------------------


def detect_sources(
    image: np.ndarray, thresh: float = 4.0, minarea: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Run SEP source extraction and return ``(y, x)`` coordinates and flux."""
    finite = np.isfinite(image)
    if np.count_nonzero(finite) < 100:
        return np.empty((0, 2)), np.empty((0,))

    fill = float(np.nanmedian(image[finite]))
    img = np.where(finite, image, fill).astype(np.float32)
    bkg = sep.Background(img)
    obj = sep.extract(img - bkg, thresh=thresh, err=bkg.globalrms, minarea=minarea)

    if len(obj) == 0:
        return np.empty((0, 2)), np.empty((0,))

    coords = np.vstack([obj["y"], obj["x"]]).T.astype(np.float64)
    flux = np.asarray(obj["flux"], dtype=np.float64)
    return coords, flux


def match_stats(
    coords1: np.ndarray, coords2: np.ndarray, shift_yx: tuple[float, float], radius: float = 1.0
) -> tuple[int, float]:
    """Count one-to-one star matches and return ``(n_matches, rms_distance)``."""
    if len(coords1) == 0 or len(coords2) == 0:
        return 0, np.inf

    tree = cKDTree(coords2)
    query = np.column_stack(
        [coords1[:, 0] - shift_yx[0], coords1[:, 1] - shift_yx[1]]
    )
    dist, nn = tree.query(query, distance_upper_bound=radius)
    good = np.isfinite(dist) & (dist < radius)
    if np.count_nonzero(good) == 0:
        return 0, np.inf

    dist_good = dist[good]
    nn_good = nn[good]
    order = np.argsort(dist_good)
    used: set[int] = set()
    keep: list[float] = []
    for k in order:
        j = int(nn_good[k])
        if j in used:
            continue
        used.add(j)
        keep.append(float(dist_good[k]))

    if not keep:
        return 0, np.inf
    keep = np.asarray(keep, dtype=np.float64)
    return len(keep), float(np.sqrt(np.mean(keep**2)))


# ---------------------------------------------------------------------------
# Shift estimation (catalog / phase / correlation)
# ---------------------------------------------------------------------------

FALLBACK_SHIFT = (-0.5, -0.5)
N_BRIGHT = 30
MATCH_RADIUS = 1.0
MIN_MATCHES_OK = 2
MAX_RMS_OK = 1.0

EPS = 1e-6          # small offset to avoid division by zero
CORR_EPS = 1e-12    # small offset for normalized correlation denominator
N_REF_STARS = 12    # number of brightest stars used as sharpness references
GRID_HALF_RANGE = 0.6   # +/- pixel range for local shift refinement
GRID_STEP = 0.1         # pixel step for local shift refinement
SCORE_MATCH_BONUS = 0.01
SCORE_RMS_PENALTY = 0.02
SCORE_BAD_RMS = 10.0
CATALOG_BIN_FACTOR = 4.0  # granularity factor for 2D histogram mode estimation


def sharpness_score(stacked: np.ndarray, ref_coords: np.ndarray) -> float:
    """Return a sharpness/peakedness score around reference star positions."""
    finite = np.isfinite(stacked)
    if np.count_nonzero(finite) < 50:
        return -np.inf

    if len(ref_coords) == 0:
        med = np.nanmedian(stacked[finite])
        std = np.nanstd(stacked[finite])
        if std <= 0:
            return -np.inf
        return float((np.nanmax(stacked[finite]) - med) / std)

    vals = []
    for y, x in ref_coords:
        yi = int(round(y))
        xi = int(round(x))
        if yi < 3 or xi < 3 or yi >= stacked.shape[0] - 3 or xi >= stacked.shape[1] - 3:
            continue
        patch = stacked[yi - 2 : yi + 3, xi - 2 : xi + 3]
        ok = np.isfinite(patch)
        if np.count_nonzero(ok) < 20:
            continue
        p = patch[ok]
        b = np.median(p)
        peak = np.max(p) - b
        flux = np.sum(np.clip(p - b, 0.0, None)) + EPS
        vals.append(peak / flux)

    return float(np.median(vals)) if vals else -np.inf


def estimate_shift_catalog(coords1: np.ndarray, flux1: np.ndarray, coords2: np.ndarray, flux2: np.ndarray) -> tuple[float, float]:
    """Estimate global shift from the mode of bright-star pairwise offsets.

    Uses the ``N_BRIGHT`` brightest stars from each catalog. If either catalog
    is empty, returns ``FALLBACK_SHIFT``.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return FALLBACK_SHIFT

    c1 = coords1[np.argsort(flux1)[-N_BRIGHT:]]
    c2 = coords2[np.argsort(flux2)[-N_BRIGHT:]]

    dy = c1[:, None, 0] - c2[None, :, 0]
    dx = c1[:, None, 1] - c2[None, :, 1]
    deltas = np.column_stack([dy.ravel(), dx.ravel()])

    quantized = np.round(deltas * CATALOG_BIN_FACTOR) / CATALOG_BIN_FACTOR
    uniq, cnt = np.unique(quantized, axis=0, return_counts=True)
    return (float(uniq[np.argmax(cnt), 0]), float(uniq[np.argmax(cnt), 1]))


def estimate_shift_phase(data1: np.ndarray, data2: np.ndarray) -> tuple[float, float]:
    """Estimate global shift using sub-pixel phase cross-correlation."""
    finite1 = np.isfinite(data1)
    finite2 = np.isfinite(data2)
    if np.count_nonzero(finite1) < 50 or np.count_nonzero(finite2) < 50:
        return FALLBACK_SHIFT

    fill1 = float(np.nanmedian(data1[finite1]))
    fill2 = float(np.nanmedian(data2[finite2]))
    img1 = np.where(finite1, data1, fill1).astype(np.float32)
    img2 = np.where(finite2, data2, fill2).astype(np.float32)
    img1 -= np.median(img1)
    img2 -= np.median(img2)

    s, _, _ = phase_cross_correlation(img1, img2, upsample_factor=20)
    return (float(s[0]), float(s[1]))


def refine_shift(
    data1: np.ndarray,
    data2: np.ndarray,
    coords1: np.ndarray,
    coords2: np.ndarray,
    flux1: np.ndarray,
    base_shift: tuple[float, float],
    fallback_shift: tuple[float, float] = FALLBACK_SHIFT,
) -> dict[str, Any]:
    """Refine a candidate shift on a local grid and return the best candidate.

    Parameters
    ----------
    data1, data2
        The two images to be aligned and stacked.
    coords1, coords2
        Detected source coordinates (y, x) for each image.
    flux1
        Flux values for the sources in ``data1``; used to pick reference stars.
    base_shift
        Starting (dy, dx) shift estimate around which the local grid is built.
    fallback_shift
        (dy, dx) shift returned when no valid grid candidate can be evaluated
        (e.g. both images contain almost no valid pixels).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``shift``: tuple[float, float] -- selected (dy, dx) shift
        - ``n_match``: int -- one-to-one matched stars at ``shift``
        - ``rms``: float -- matching RMS at ``shift``
        - ``stacked``: np.ndarray -- stacked image using ``shift``
        - ``score``: float -- sharpness-based score used to pick ``shift``
    """
    if len(coords1) > 0:
        ref_coords = coords1[np.argsort(flux1)[-N_REF_STARS:]]
    else:
        ref_coords = np.empty((0, 2))

    dy0, dx0 = base_shift
    dy_grid = np.arange(dy0 - GRID_HALF_RANGE, dy0 + GRID_HALF_RANGE + 1e-9, GRID_STEP)
    dx_grid = np.arange(dx0 - GRID_HALF_RANGE, dx0 + GRID_HALF_RANGE + 1e-9, GRID_STEP)

    best = None
    for dy in dy_grid:
        for dx in dx_grid:
            s = (float(dy), float(dx))
            n_match, rms = match_stats(coords1, coords2, s, radius=MATCH_RADIUS)
            st = stack_pair(data1, data2, s)
            sharp = sharpness_score(st, ref_coords)
            score = sharp + SCORE_MATCH_BONUS * n_match - SCORE_RMS_PENALTY * (
                rms if np.isfinite(rms) else SCORE_BAD_RMS
            )
            if (best is None) or (score > best["score"]):
                best = {"shift": s, "n_match": n_match, "rms": rms, "stacked": st, "score": score}

    if best is None:
        best = {
            "shift": fallback_shift,
            "n_match": 0,
            "rms": np.inf,
            "stacked": stack_pair(data1, data2, fallback_shift),
            "score": -np.inf,
        }
    return best


# ---------------------------------------------------------------------------
# Correlation-only alignment helpers
# ---------------------------------------------------------------------------

UPSAMPLE_FACTOR = 20
FINE_HALF_RANGE = 0.6
FINE_STEP = 0.1


def preprocess_for_corr(image: np.ndarray) -> np.ndarray | None:
    """Fill non-finite pixels and standardize an image for cross-correlation."""
    finite = np.isfinite(image)
    if np.count_nonzero(finite) < 50:
        return None

    fill = float(np.nanmedian(image[finite]))
    img = np.where(finite, image, fill).astype(np.float32)
    img -= np.median(img)
    std = np.std(img)
    if std > 0:
        img /= std
    return img


def corr_score(
    ref_img: np.ndarray, mov_img: np.ndarray, shift_yx: tuple[float, float]
) -> float:
    """Normalized cross-correlation after shifting ``mov_img`` by ``shift_yx``."""
    moved = ndi_shift(mov_img, shift=shift_yx, order=3, mode="constant", cval=np.nan)
    valid = np.isfinite(ref_img) & np.isfinite(moved)
    if np.count_nonzero(valid) < 100:
        return -np.inf

    a = ref_img[valid].astype(np.float64)
    b = moved[valid].astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + CORR_EPS
    return float(np.dot(a, b) / denom)


def estimate_shift_correlation(
    data1: np.ndarray, data2: np.ndarray
) -> tuple[tuple[float, float], float, float]:
    """Coarse phase-correlation + fine grid search maximizing normalized correlation."""
    ref_img = preprocess_for_corr(data1)
    mov_img = preprocess_for_corr(data2)
    if ref_img is None or mov_img is None:
        return FALLBACK_SHIFT, -np.inf, np.nan

    coarse, phase_err, _ = phase_cross_correlation(
        ref_img, mov_img, upsample_factor=UPSAMPLE_FACTOR
    )
    dy0, dx0 = float(coarse[0]), float(coarse[1])

    dy_grid = np.arange(dy0 - FINE_HALF_RANGE, dy0 + FINE_HALF_RANGE + 1e-9, FINE_STEP)
    dx_grid = np.arange(dx0 - FINE_HALF_RANGE, dx0 + FINE_HALF_RANGE + 1e-9, FINE_STEP)

    best_shift = (dy0, dx0)
    best_corr = -np.inf
    for dy in dy_grid:
        for dx in dx_grid:
            s = (float(dy), float(dx))
            c = corr_score(ref_img, mov_img, s)
            if c > best_corr:
                best_corr = c
                best_shift = s

    return best_shift, best_corr, float(phase_err)


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

STACK_WEIGHT = 0.5  # equal weighting when averaging the two aligned frames


def stack_pair(
    data1: np.ndarray, data2: np.ndarray, shift_yx: tuple[float, float]
) -> np.ndarray:
    """Shift ``data2`` by ``shift_yx`` and average it with ``data1``."""
    moved = ndi_shift(data2, shift=shift_yx, order=3, mode="constant", cval=np.nan)
    valid = (
        ndi_shift(
            np.ones_like(data2, dtype=np.float32),
            shift=shift_yx,
            order=0,
            mode="constant",
            cval=0.0,
        )
        > 0.5
    )
    return np.where(valid, STACK_WEIGHT * (data1 + moved), np.nan)


# ---------------------------------------------------------------------------
# PSF fitting
# ---------------------------------------------------------------------------

SEP_THRESH_SIGMA = 3.5
SEP_MINAREA = 5
BRIGHT_FLUX_QUANTILE = 0.7
MAX_LABELS = 20
OUTLIER_IQR_K = 1.5
MIN_POINTS_FOR_IQR = 8
# Conversion factor from Gaussian sigma to FWHM: 2 * sqrt(2 * ln(2))
FWHM_SIGMA = 2.354820045
PSF_INITIAL_SIGMA = 1.5
PSF_INITIAL_THETA = 0.0
PSF_MIN_AMP = 1e-3


def gaussian2d_model(
    xy: tuple[np.ndarray, np.ndarray],
    amp: float,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """Flattened 2D elliptical Gaussian used by ``curve_fit``."""
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    a = (ct**2) / (2.0 * sx**2) + (st**2) / (2.0 * sy**2)
    b = -np.sin(2.0 * theta) / (4.0 * sx**2) + np.sin(2.0 * theta) / (4.0 * sy**2)
    c = (st**2) / (2.0 * sx**2) + (ct**2) / (2.0 * sy**2)
    z = (
        amp
        * np.exp(
            -(a * (x - x0) ** 2 + 2.0 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
        )
        + offset
    )
    return z.ravel()


def fit_psf_on_patch(
    image: np.ndarray, x: float, y: float, half_size: int = 6
) -> dict[str, Any] | None:
    """Fit a 2D Gaussian to a patch centered on ``(x, y)`` and return fit info.

    Returns
    -------
    dict[str, Any] | None
        If the fit succeeds, returns a dictionary with keys:
        - ``x``, ``y``: float -- fitted star center in full-image coordinates
        - ``fwhm``: float -- circularized full-width at half-maximum (pixels)
        - ``patch``: np.ndarray -- cutout used for fitting
        - ``model``: np.ndarray -- best-fit Gaussian rendered on the cutout
        - ``x0_local``, ``y0_local``: float -- fitted center in cutout coordinates
        Returns ``None`` if the patch is too small or the fit fails.
    """
    ny, nx = image.shape
    xi, yi = int(round(x)), int(round(y))
    x1, x2 = max(0, xi - half_size), min(nx, xi + half_size + 1)
    y1, y2 = max(0, yi - half_size), min(ny, yi + half_size + 1)

    patch = image[y1:y2, x1:x2]
    if patch.size < 25:
        return None

    valid = np.isfinite(patch)
    if np.count_nonzero(valid) < 25:
        return None

    z = patch.astype(np.float64)
    yy, xx = np.indices(z.shape, dtype=np.float64)

    z_med = np.nanmedian(z)
    z_peak = np.nanmax(z)
    amp0 = max(z_peak - z_med, PSF_MIN_AMP)
    p0 = [
        amp0,
        float(x - x1),
        float(y - y1),
        PSF_INITIAL_SIGMA,
        PSF_INITIAL_SIGMA,
        PSF_INITIAL_THETA,
        z_med,
    ]
    lower = [0.0, 0.0, 0.0, 0.3, 0.3, -np.pi / 2.0, -np.inf]
    upper = [np.inf, z.shape[1] - 1.0, z.shape[0] - 1.0, 8.0, 8.0, np.pi / 2.0, np.inf]

    try:
        popt, _ = curve_fit(
            gaussian2d_model,
            (xx[valid], yy[valid]),
            z[valid].ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000,
        )
    except Exception:
        return None

    _, x0, y0, sx, sy, _, _ = popt
    sx = float(abs(sx))
    sy = float(abs(sy))
    sigma = np.sqrt(0.5 * (sx**2 + sy**2))
    fwhm = FWHM_SIGMA * sigma
    model = gaussian2d_model((xx, yy), *popt).reshape(z.shape)

    return {
        "x": x1 + float(x0),
        "y": y1 + float(y0),
        "fwhm": float(fwhm),
        "patch": z,
        "model": model,
        "x0_local": float(x0),
        "y0_local": float(y0),
    }


def measure_psf_for_bright_stars(
    stacked: np.ndarray,
    thresh: float = SEP_THRESH_SIGMA,
    minarea: int = SEP_MINAREA,
    bright_flux_quantile: float = BRIGHT_FLUX_QUANTILE,
    max_labels: int = MAX_LABELS,
) -> list[dict]:
    """Detect stars with SEP and fit a Gaussian PSF to the bright ones."""
    finite = np.isfinite(stacked)
    if np.count_nonzero(finite) < 100:
        return []

    fill = float(np.nanmedian(stacked[finite]))
    img = np.where(finite, stacked, fill).astype(np.float32)
    bkg = sep.Background(img)
    sub = img - bkg

    objects = sep.extract(sub, thresh=thresh, err=bkg.globalrms, minarea=minarea)
    if len(objects) == 0:
        return []

    flux = np.asarray(objects["flux"], dtype=np.float64)
    flux_cut = np.quantile(flux, bright_flux_quantile)

    bright = [o for o in objects if float(o["flux"]) >= float(flux_cut)]
    bright = sorted(bright, key=lambda o: float(o["flux"]), reverse=True)[:max_labels]

    psf_list: list[dict] = []
    for o in bright:
        res = fit_psf_on_patch(sub, float(o["x"]), float(o["y"]), half_size=6)
        if res is not None and np.isfinite(res["fwhm"]):
            res["flux"] = float(o["flux"])
            psf_list.append(res)

    return psf_list


# ---------------------------------------------------------------------------
# Outlier filtering / statistics
# ---------------------------------------------------------------------------


def filter_outliers_iqr(
    values: list[float] | np.ndarray,
    k: float = OUTLIER_IQR_K,
    min_points: int = MIN_POINTS_FOR_IQR,
) -> tuple[np.ndarray, int, float, float]:
    """Remove outliers using the IQR rule.

    Returns ``(cleaned_array, n_removed, low_bound, high_bound)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, 0, np.nan, np.nan

    if arr.size < min_points:
        return arr, 0, np.nan, np.nan

    q1, q3 = np.percentile(arr, [25.0, 75.0])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return arr, 0, np.nan, np.nan

    low = q1 - k * iqr
    high = q3 + k * iqr
    keep = (arr >= low) & (arr <= high)
    cleaned = arr[keep]

    removed = int(arr.size - cleaned.size)
    if cleaned.size == 0:
        return arr, 0, low, high

    return cleaned, removed, float(low), float(high)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_fit_profiles(
    fig: "matplotlib.figure.Figure",
    spec: "matplotlib.gridspec.SubplotSpec",
    psf_fit: dict | None,
) -> None:
    """Draw X/Y intensity profiles (data vs. fit) for a PSF fit."""
    ax_x = fig.add_subplot(spec[0, 0])
    ax_y = fig.add_subplot(spec[1, 0])

    if psf_fit is None:
        for ax, label in ((ax_x, "X profile"), (ax_y, "Y profile")):
            ax.text(0.2, 0.5, "No valid bright-star fit", transform=ax.transAxes)
            ax.set_title(label)
            ax.set_axis_off()
        return

    patch = psf_fit["patch"]
    model = psf_fit["model"]
    row_idx = int(np.clip(round(psf_fit["y0_local"]), 0, patch.shape[0] - 1))
    col_idx = int(np.clip(round(psf_fit["x0_local"]), 0, patch.shape[1] - 1))

    x_coords = np.arange(patch.shape[1], dtype=np.float64)
    y_coords = np.arange(patch.shape[0], dtype=np.float64)

    ax_x.plot(x_coords, patch[row_idx, :], "o", color="black", markersize=3, label="Data")
    ax_x.plot(
        x_coords, model[row_idx, :], "-", color="crimson", linewidth=1.8, label="Fit"
    )
    ax_x.set_title(f"X profile @ y={row_idx}")
    ax_x.set_xlabel("Patch X")
    ax_x.set_ylabel("Intensity")
    ax_x.grid(alpha=0.2)
    ax_x.legend(fontsize=8)

    ax_y.plot(y_coords, patch[:, col_idx], "o", color="black", markersize=3, label="Data")
    ax_y.plot(
        y_coords, model[:, col_idx], "-", color="royalblue", linewidth=1.8, label="Fit"
    )
    ax_y.set_title(f"Y profile @ x={col_idx}")
    ax_y.set_xlabel("Patch Y")
    ax_y.set_ylabel("Intensity")
    ax_y.grid(alpha=0.2)
    ax_y.legend(fontsize=8)


# ---------------------------------------------------------------------------
# TIFF -> FITS conversion
# ---------------------------------------------------------------------------


def convert_tiff_to_fits(
    src_root: Path | str = "./NCU Spotlight/raw_data",
    dst_root: Path | str | None = None,
) -> tuple[int, int]:
    """Batch-convert TIFF images under ``src_root`` to FITS format.

    The destination mirrors the source layout. Returns ``(converted, failed)``.
    """
    src_root = Path(src_root)
    if dst_root is None:
        dst_root = src_root / "fits_outputs"
    else:
        dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    all_tif_files = [
        p
        for p in src_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".tif", ".tiff")
        and "fits_outputs" not in p.relative_to(src_root).parts
        and not any(part.startswith(".") for part in p.relative_to(src_root).parts)
    ]

    converted = 0
    failed = 0
    for tif_path in sorted(all_tif_files):
        out_path = dst_root / tif_path.relative_to(src_root).parent / f"{tif_path.stem}.fits"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import tifffile

            image_data = tifffile.imread(str(tif_path))
            fits.PrimaryHDU(image_data).writeto(str(out_path), overwrite=True)
            print(f"CONVERTED: {tif_path.relative_to(src_root)} -> {out_path.relative_to(dst_root)}")
            converted += 1
        except Exception as exc:
            print(f"FAILED: {tif_path.relative_to(src_root)} ({exc})")
            failed += 1

    print(f"Summary: {converted} converted, {failed} failed")
    return converted, failed
