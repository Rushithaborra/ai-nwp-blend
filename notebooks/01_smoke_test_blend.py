import xarray as xr
from download_data import daily_rain
from blend.io import load_forecasts, load_imd, to_imd_grid
from blend.weights import equal_weight
from blend.metrics import rmse, ets

date = "2025-07-15"
fc = load_forecasts(date)  # common steps across all 4 models -> 6-hourly only
imd = load_imd("rain", 2025)

fc_imd_grid = to_imd_grid(fc, imd)
blend = equal_weight(fc_imd_grid)

# GraphCast/AIFS are 6-hourly, so lead day 1 uses the 06-30h (06-06 UTC) window,
# not IMD's native 03 UTC boundary - see README's rainfall-day note (3h mismatch).
obs_day1 = imd["rain"].sel(time=date)
blend_day1 = daily_rain(blend["tp"], [6, 30]).isel(lead_day=0)

print("blend vs IMD  rmse:", float(rmse(blend_day1, obs_day1)))
print("blend vs IMD  ets>2mm:", float(ets(blend_day1, obs_day1, 2.0)))

for m in fc_imd_grid["model"].values:
    day1 = daily_rain(fc_imd_grid["tp"].sel(model=m), [6, 30]).isel(lead_day=0)
    print(f"{m:12s} rmse:", float(rmse(day1, obs_day1)),
          " ets>2mm:", float(ets(day1, obs_day1, 2.0)))
