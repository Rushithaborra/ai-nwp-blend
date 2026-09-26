"""Tier 3: forecast-derived regime labels (never from observed/verified fields -
see README - a model can't know the true regime at lead time 3-7 days)."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REGIME_VARS = ("u850", "v850", "z500")


def fit_regimes(regime_ds, n_clusters=4, vars=REGIME_VARS):
    """regime_ds needs a leading sample dim (e.g. "date") plus "step_h",
    "latitude", "longitude" - one row per (date, step_h) forecast field."""
    stacked = regime_ds[list(vars)].to_array("var").stack(sample=("date", "step_h"))
    X = stacked.stack(feature=("var", "latitude", "longitude")).transpose(
        "sample", "feature").values
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(scaler.transform(X))
    return km, scaler


def predict_regime(km, scaler, regime_ds, vars=REGIME_VARS):
    """Label a single forecast's regime fields (same sample dims as fit_regimes)."""
    stacked = regime_ds[list(vars)].to_array("var").stack(sample=("date", "step_h"))
    X = stacked.stack(feature=("var", "latitude", "longitude")).transpose(
        "sample", "feature").values
    return km.predict(scaler.transform(X))
