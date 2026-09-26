"""Load and align forecast/truth sources onto a common (model, step_h, lat, lon) grid."""
import numpy as np
import xarray as xr

MODELS = ["gfs", "graphcast", "ifs", "aifs-single"]


def _standardize(ds):
    """Rename to latitude/longitude/step_h (hours) regardless of source."""
    ds = ds.rename({c: "latitude" for c in ds.coords if c in ("lat",)})
    ds = ds.rename({c: "longitude" for c in ds.coords if c in ("lon",)})
    if "step" in ds.dims and "step_h" not in ds.dims:
        hours = (ds["step"].values / np.timedelta64(1, "h")).astype(int)
        ds = ds.assign_coords(step=hours).rename(step="step_h")
    keep = ("latitude", "longitude", "step_h", "time")
    return ds.drop_vars([c for c in ds.coords if c not in keep])


def load_forecasts(date, models=MODELS, data_dir="data"):
    """One Dataset with a 'model' dim, aligned on the shared 6-hourly steps."""
    per_model = {m: _standardize(xr.open_dataset(f"{data_dir}/{m}_{date}.nc"))
                 for m in models}
    common_steps = sorted(set.intersection(
        *(set(ds["step_h"].values) for ds in per_model.values())))
    aligned = [ds.sel(step_h=common_steps) for ds in per_model.values()]
    return xr.concat(aligned, dim=xr.DataArray(models, dims="model", name="model"))


def load_regime(date, data_dir="data"):
    return _standardize(xr.open_dataset(f"{data_dir}/regime_{date}.nc"))


def load_imd(var, year, data_dir="data"):
    return _standardize(xr.open_dataset(f"{data_dir}/imd_{var}_{year}_{year}.nc"))


def to_imd_grid(ds, imd_ds):
    """Reindex a forecast Dataset onto IMD's (smaller-extent, same-lattice) grid."""
    return ds.sel(latitude=imd_ds["latitude"], longitude=imd_ds["longitude"],
                  method="nearest")
