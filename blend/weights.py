"""Tiers 1-2 of the blend: equal-weight mean, then inverse-error weighting.
Tiers 3-4 (regime-conditioned, meta-learner) need the multi-month history
to fit on - see regimes.py once run_all.py has enough dates downloaded."""


def equal_weight(forecasts, dim="model"):
    return forecasts.mean(dim)


def inverse_error_weight(forecasts, error, dim="model", power=2):
    """error: per-model skill (e.g. rolling-window RMSE), same `dim` as forecasts.
    Returns (blended, weights); weights sum to 1 over `dim`."""
    inv = 1.0 / (error ** power)
    weights = inv / inv.sum(dim)
    return (forecasts * weights).sum(dim), weights
