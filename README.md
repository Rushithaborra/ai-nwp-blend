# AI-NWP Blend (SIH 26081)

Hybrid AI + numerical-weather-prediction (NWP) rainfall/weather blending prototype for the India region (lat 40-5N, lon 65-100E).

## Why

Physics-based NWP models and newer AI weather models each have different strengths. This project pulls forecasts from both kinds of model plus two ground-truth sources, and blends them into a single forecast that should beat either family alone over India.

## Data sources

| Source | Type | Role |
|---|---|---|
| GFS (NOAA) | Physics-based | Baseline global forecast |
| GraphCastGFS | AI (on GFS init conditions) | AI counterpart to GFS |
| ECMWF IFS | Physics-based | Second, stronger physics baseline |
| ECMWF AIFS | AI (on IFS init conditions) | AI counterpart to IFS |
| ERA5 (CDS) | Reanalysis | Historical "ground truth" for training/validation |
| IMD gridded | Observation | India-specific ground truth for scoring |
| Regime (GFS 850mb wind + 500mb height) | Forecast-derived | Features for regime-conditioned blend weights (never from observed/verified fields - see Blending below) |

## Setup

```bash
conda create -n sih -c conda-forge python=3.11 xarray netcdf4 cfgrib eccodes \
      herbie-data ecmwf-opendata cdsapi pandas -y
conda activate sih
pip install imdlib
python -c "import cfgrib, herbie, ecmwf.opendata, cdsapi, imdlib; print('all good')"
```

ERA5 needs a free Copernicus CDS account + a `~/.cdsapirc` file (not committed — see `.gitignore`):

```
url: https://cds.climate.copernicus.eu/api
key: <your-token>
```

## Usage

```bash
# smoke test: one date per source
python download_data.py

# bulk download per source, skips files that already exist, logs failures to logs/failed.txt
python run_all.py --source noaa   --start 2025-03-01 --end 2026-09-20
python run_all.py --source ecmwf  --start 2025-03-01 --end 2026-09-20
python run_all.py --source era5   --start 2025-03-01 --end 2026-09-20
python run_all.py --source imd    --start 2025-03-01 --end 2026-09-20
python run_all.py --source regime --start 2025-03-01 --end 2026-09-20
```

IMD is downloaded one year at a time: the current (incomplete) year has fewer
records than a finished year, which breaks `imdlib`'s fixed-record binary
parser if you ask for a mixed range in one call. The in-progress year will
keep failing (logged, not fatal) until IMD publishes the finished grid, or
until you swap in a real-time product (e.g. IMERG) for recent months.

**Hackathon prototype window:** with a 3-day submission deadline, the full
18-month backfill above (5-8+ days at observed download speed, mostly
ECMWF throttling) isn't viable. Use a focused window instead - enough days
to compute rolling error weights and find one real case-study day, without
blocking the deadline:

```bash
python run_all.py --source noaa   --start 2025-07-01 --end 2025-07-31
python run_all.py --source ecmwf  --start 2025-07-01 --end 2025-07-31
python run_all.py --source regime --start 2025-07-01 --end 2025-07-31
```

Raw data (`data/`) and logs (`logs/`) are gitignored — they're large (10-15 GB for the full run) and regenerable from the scripts above, so they aren't pushed to the repo. Run the smoke test to get a few sample `.nc` files locally before writing blending code against them.

## Blending

`blend/` has the tiered blend: `io.py` loads and aligns the 4 forecast
models plus IMD onto a common grid/step axis (IFS/AIFS use `step`
timedeltas, GFS/GraphCast/regime use integer `step_h` - `io.py` unifies
this); `weights.py` has tier 1 (equal-weight) and tier 2 (inverse-error);
`regimes.py` has tier 3 (k-means on forecast-derived 850mb wind + 500mb
height - fit only once enough dates exist, never on observed regimes);
`metrics.py` has RMSE, bias, ETS, and FSS. See
`notebooks/01_smoke_test_blend.py` for a working end-to-end example on the
smoke-test date (run notebook scripts as `PYTHONPATH=. python
notebooks/<script>.py` from the repo root, so `import blend` resolves).
`notebooks/02_find_case_study_days.py` picks candidate extreme-rainfall
days from IMD data alone - **2025-07-26** stood out (209 grid cells over
IMD's 64.5mm heavy-rain threshold, most of any July 2025 day). Tier 4 (meta-learner) and CRPS-based extreme calibration
come after the bulk download gives enough history to train on.

## Fixed scope

Do not change the India bounding box (`LAT_N, LAT_S, LON_W, LON_E` in [download_data.py](download_data.py)) or the step lists mid-project without discussing with the team — downstream blending code assumes this grid.
