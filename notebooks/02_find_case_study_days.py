"""Find candidate extreme-rainfall case-study days in July 2025 from IMD data
(no forecast data needed - just the observed truth we already have)."""
from blend.io import load_imd

imd = load_imd("rain", 2025)
july = imd["rain"].sel(time=slice("2025-07-01", "2025-07-31"))

# IMD's own heavy-rainfall threshold: 64.5 mm/day, area-averaged over the max
area_max = july.max(["latitude", "longitude"])
top5 = area_max.to_series().sort_values(ascending=False).head(5)
print("Top 5 days by max grid-cell rainfall (mm) over India, July 2025:")
print(top5)

heavy_area = (july >= 64.5).sum(["latitude", "longitude"])
top5_area = heavy_area.to_series().sort_values(ascending=False).head(5)
print("\nTop 5 days by number of grid cells exceeding IMD heavy-rain threshold (64.5mm):")
print(top5_area)
