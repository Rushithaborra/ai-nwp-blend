"""Verification metrics. All take/return xarray objects; dim is reduced (e.g. "latitude","longitude" or "time")."""
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


def rmse(pred, obs, dim=None):
    return np.sqrt(((pred - obs) ** 2).mean(dim))


def bias(pred, obs, dim=None):
    return (pred - obs).mean(dim)


def ets(pred, obs, threshold, dim=None):
    """Equitable Threat Score at a rainfall (or other) threshold."""
    hit = ((pred >= threshold) & (obs >= threshold)).sum(dim)
    miss = ((pred < threshold) & (obs >= threshold)).sum(dim)
    false_alarm = ((pred >= threshold) & (obs < threshold)).sum(dim)
    n = (pred >= -np.inf).sum(dim)  # point count, works with dim=None too
    hit_random = (hit + miss) * (hit + false_alarm) / n
    return (hit - hit_random) / (hit + miss + false_alarm - hit_random)


def fss(pred, obs, threshold, window):
    """Fractions Skill Score over a square neighborhood of `window` grid cells.
    pred/obs: 2-D (latitude, longitude) arrays for a single field/time/lead."""
    p = uniform_filter((pred.values >= threshold).astype(float), size=window)
    o = uniform_filter((obs.values >= threshold).astype(float), size=window)
    mse = np.mean((p - o) ** 2)
    mse_ref = np.mean(p ** 2) + np.mean(o ** 2)
    return 1 - mse / mse_ref if mse_ref > 0 else np.nan


def crps_ensemble(members, obs, dim="model"):
    """CRPS for a finite ensemble (Gneiting & Raftery closed form), reduced over `dim`."""
    n = members.sizes[dim]
    term1 = np.abs(members - obs).mean(dim)
    diff = members - members.rename({dim: dim + "_2"})
    term2 = np.abs(diff).sum([dim, dim + "_2"]) / (2 * n * n)
    return term1 - term2
