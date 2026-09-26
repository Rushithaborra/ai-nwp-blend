"""
SIH 26081 - data downloader for the hybrid AI-NWP blending prototype.

Sources:
  GFS, GraphCastGFS  -> NOAA AWS buckets via Herbie (no account)
  ECMWF IFS, AIFS    -> AWS mirror via ecmwf-opendata (no account)
  ERA5               -> Copernicus CDS via cdsapi (free account + licence)
  IMD gridded        -> IMD Pune via imdlib (no account)

Install (conda recommended because of eccodes):
  conda create -n sih -c conda-forge python=3.11 xarray netcdf4 cfgrib eccodes \
        herbie-data ecmwf-opendata cdsapi pandas
  pip install imdlib

UNTESTED against live servers - run each source for ONE date first.
"""
import os
import glob
import pandas as pd
import xarray as xr

# ---------------- fixed scope (do not change mid-project) ----------------
LAT_N, LAT_S, LON_W, LON_E = 40, 5, 65, 100
OUT = "data"
STEPS_6H = list(range(0, 121, 6))        # 0..120 h
STEPS_3H = list(range(0, 121, 3))        # for models that offer 3-hourly


def crop(ds):
    """Crop to India box; handles 0-360 vs -180-180 and lat order."""
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    if float(ds[lon].min()) < 0 and LON_W >= 0:
        pass  # -180..180 grid, India box already positive
    lat_desc = ds[lat][0] > ds[lat][-1]
    lat_slice = slice(LAT_N, LAT_S) if lat_desc else slice(LAT_S, LAT_N)
    return ds.sel({lat: lat_slice, lon: slice(LON_W, LON_E)})


def save(ds, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    ds.to_netcdf(path)
    print("saved", path)


# ---------------- GFS and GraphCastGFS (Herbie) ----------------
HERBIE_SEARCH = {
    "t2m":  ":TMP:2 m above ground",
    "u10":  ":UGRD:10 m above ground",
    "v10":  ":VGRD:10 m above ground",
    # running total since init; verify with H.inventory() that it exists
    "tp":   r":APCP:surface:0-[0-9]+ hour acc",
}

# Forecast-derived regime features (GFS only - shared GDAS init with
# GraphCastGFS, used as the single regime reference for all 4 models so
# weighting stays consistent across the IFS/AIFS branch too).
REGIME_SEARCH = {
    "u850": ":UGRD:850 mb",
    "v850": ":VGRD:850 mb",
    "z500": ":HGT:500 mb",
}


def get_noaa(date, model="gfs", steps=STEPS_6H, search=HERBIE_SEARCH,
             out_name=None, skip_step0=("tp",)):
    from herbie import Herbie
    pieces = []
    for fxx in steps:
        H = Herbie(f"{date} 00:00", model=model, product="pgrb2.0p25", fxx=fxx)
        for var, pattern in search.items():
            if var in skip_step0 and fxx == 0:
                continue
            try:
                ds = H.xarray(pattern, remove_grib=True)
                if isinstance(ds, list):
                    ds = ds[0]
                ds = crop(ds)
                name = list(ds.data_vars)[0]
                da = ds[name].rename(var).expand_dims(step_h=[fxx])
                pieces.append(da.drop_vars(
                    [c for c in da.coords if c not in
                     ("latitude", "longitude", "step_h")]))
            except Exception as e:
                print(f"[{model} {date} f{fxx:03d} {var}] missing: {e}")
    if pieces:
        save(xr.merge(pieces), f"{out_name or model}_{date}.nc")


def get_regime(date, steps=STEPS_6H):
    """850mb u/v wind + 500mb height from GFS, for forecast-derived
    regime clustering (available at forecast time, unlike observed
    regimes - see README)."""
    get_noaa(date, "gfs", steps, search=REGIME_SEARCH, out_name="regime",
             skip_step0=())


# ---------------- ECMWF IFS / AIFS (AWS mirror) ----------------
def get_ecmwf(date, model="ifs", steps=STEPS_6H):
    from ecmwf.opendata import Client
    import cfgrib
    client = Client(source="aws")
    tmp = f"tmp_{model}_{date}.grib2"
    client.retrieve(date=date.replace("-", ""), time=0, step=steps,
                    stream="oper", type="fc", model=model,
                    param=["2t", "10u", "10v", "tp"], target=tmp)
    # mixed level types -> several hypercubes
    parts = [crop(d) for d in cfgrib.open_datasets(tmp)]
    ds = xr.merge(parts, compat="override")
    if "tp" in ds:
        # IFS reports tp in metres (needs x1000 -> mm); AIFS reports
        # kg/m^2, numerically already mm of water. Convert only the former.
        if ds["tp"].attrs.get("GRIB_units") == "m":
            ds["tp"] = ds["tp"] * 1000.0
        ds["tp"].attrs["units"] = "mm"
    save(ds, f"{model}_{date}.nc")
    for f in glob.glob(tmp + "*"):
        os.remove(f)                           # delete raw + .idx


# ---------------- ERA5 (CDS) ----------------
def get_era5_month(year, month):
    """Needs ~/.cdsapirc with the CDS url and your key."""
    import cdsapi
    c = cdsapi.Client()
    days = [f"{d:02d}" for d in range(1, 32)]
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "10m_u_component_of_wind",
                     "10m_v_component_of_wind"],
        "year": str(year), "month": f"{month:02d}", "day": days,
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
        "area": [LAT_N, LON_W, LAT_S, LON_E],   # N, W, S, E
        "data_format": "netcdf", "download_format": "unarchived",
    }, os.path.join(OUT, f"era5_{year}{month:02d}.nc"))


# ---------------- IMD gridded rainfall / temperature ----------------
def get_imd(var="rain", start_yr=2025, end_yr=2026):
    import imdlib as imd
    d = imd.get_data(var, start_yr, end_yr, fn_format="yearwise",
                     file_dir=os.path.join(OUT, "imd_raw"))
    save(d.get_xarray(), f"imd_{var}_{start_yr}_{end_yr}.nc")


# ---------------- rainfall-day helper ----------------
def daily_rain(tp_cum, boundary_steps):
    """
    tp_cum: DataArray of running-total rain with coord 'step_h' (hours).
    boundary_steps: e.g. [3, 27, 51, 75, 99] for IMD 03 UTC days from a
    00z run, or [6, 30, 54, 78, 102] for 6-hourly models (06 UTC days -
    document the 3 h mismatch).
    """
    out = []
    for a, b in zip(boundary_steps[:-1], boundary_steps[1:]):
        out.append((tp_cum.sel(step_h=b) - tp_cum.sel(step_h=a))
                   .clip(min=0).expand_dims(lead_day=[b // 24 + 1]))
    return xr.concat(out, dim="lead_day")


if __name__ == "__main__":
    # Smoke test: ONE date per source before scaling up.
    test_date = "2025-07-15"
    get_noaa(test_date, "gfs", STEPS_3H)
    get_noaa(test_date, "graphcast", STEPS_6H)
    get_ecmwf(test_date, "ifs", STEPS_3H)
    get_ecmwf(test_date, "aifs-single", STEPS_6H)
    get_regime(test_date)
    get_imd("rain", 2025, 2025)
    # get_era5_month(2025, 7)   # enable after setting up ~/.cdsapirc

    # Full run (after smoke test passes), 00z runs only:
    # for d in pd.date_range("2025-03-01", "2026-09-20").strftime("%Y-%m-%d"):
    #     for m in ["gfs", "graphcast"]:
    #         get_noaa(d, m)
    #     for m in ["ifs", "aifs-single"]:
    #         get_ecmwf(d, m)
