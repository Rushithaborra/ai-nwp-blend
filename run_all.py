"""
Bulk runner for download_data.py. Skips files that already exist, so you
can stop and restart it any time. Failures go to logs/failed.txt.

Examples:
  python run_all.py --source noaa   --start 2025-03-01 --end 2026-09-20
  python run_all.py --source ecmwf  --start 2025-03-01 --end 2026-09-20
  python run_all.py --source era5   --start 2025-03-01 --end 2026-09-20
  python run_all.py --source imd    --start 2025-03-01 --end 2026-09-20
"""
import argparse
import os
import traceback
import pandas as pd
import download_data as dd

os.makedirs("logs", exist_ok=True)


def log_fail(tag):
    with open("logs/failed.txt", "a") as f:
        f.write(f"{tag}\n{traceback.format_exc()}\n")
    print("FAILED", tag)


def exists(name):
    return os.path.exists(os.path.join(dd.OUT, name))


def run(source, start, end):
    days = pd.date_range(start, end).strftime("%Y-%m-%d")

    if source == "noaa":
        for d in days:
            for model, steps in [("gfs", dd.STEPS_3H), ("graphcast", dd.STEPS_6H)]:
                if exists(f"{model}_{d}.nc"):
                    continue
                try:
                    dd.get_noaa(d, model, steps)
                except Exception:
                    log_fail(f"{model} {d}")

    elif source == "ecmwf":
        for d in days:
            for model, steps in [("ifs", dd.STEPS_3H), ("aifs-single", dd.STEPS_6H)]:
                if exists(f"{model}_{d}.nc"):
                    continue
                try:
                    dd.get_ecmwf(d, model, steps)
                except Exception:
                    log_fail(f"{model} {d}")

    elif source == "era5":
        months = pd.period_range(start, end, freq="M")
        for p in months:
            if exists(f"era5_{p.year}{p.month:02d}.nc"):
                continue
            try:
                dd.get_era5_month(p.year, p.month)
            except Exception:
                log_fail(f"era5 {p}")

    elif source == "imd":
        y0, y1 = int(start[:4]), int(end[:4])
        for var in ["rain", "tmax", "tmin"]:
            if exists(f"imd_{var}_{y0}_{y1}.nc"):
                continue
            try:
                dd.get_imd(var, y0, y1)
            except Exception:
                log_fail(f"imd {var}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["noaa", "ecmwf", "era5", "imd"])
    ap.add_argument("--start", default="2025-03-01")
    ap.add_argument("--end", default="2026-09-20")
    a = ap.parse_args()
    run(a.source, a.start, a.end)
