"""Make the figures used in presentation.md from the focos_analise parquet files.

Run from the repo root after downloading the data into data/raw/focos_analise/:

    uv run python presentation/make_figures.py

The script also prints the audit numbers quoted on the slides. It only describes the
data; no models are trained here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "focos_analise"
FIGURES = ROOT / "presentation" / "figures"
YEARS = range(2018, 2026)
COLUMNS = [
    "foco_id", "latitude", "longitude", "geometria", "data_hora_gmt", "satelite", "bioma",
    "classe_mb", "nome_classe_mb", "altitude_m", "temp_c", "rh", "wind_speed", "tp_mm",
    "ffwi", "numero_dias_sem_chuva", "precipitacao", "risco_fogo", "vegetacao", "frp",
]
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#8C8C8C", "check": "#55A868"}
SPLIT_LABELS = {
    "train": "Train 2018-2022", "val": "Validation 2023", "test": "Test 2024",
    "check": "Later check 2025",
}
WEATHER_LABELS = {
    "rh": "Relative humidity",
    "ffwi": "Fire weather index (FFWI)",
    "tp_mm": "Precipitation",
    "numero_dias_sem_chuva": "Days without rain",
    "temp_c": "Temperature",
    "wind_speed": "Wind speed",
    "altitude_m": "Altitude",
}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
               "Dec"]

FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def year_split(year: int) -> str:
    """Name the split used by the research notebook for one calendar year."""
    if year <= 2022:
        return "train"
    if year == 2023:
        return "val"
    if year == 2024:
        return "test"
    return "check"


def draw_box(ax, x: float, y: float, width: float, height: float, text: str,
             color: str) -> None:
    """Draw one rounded pipeline box with centered text."""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="#555555", linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=12)


# Load every year of the enriched hotspot table with only the columns we use.
yearly_frames = [pd.read_parquet(RAW / f"focos_enriquecido_{year}.parquet", columns=COLUMNS)
                 for year in YEARS]
hotspots = pd.concat(yearly_frames, ignore_index=True)
hotspots["year"] = hotspots["data_hora_gmt"].dt.year
hotspots["month"] = hotspots["data_hora_gmt"].dt.month
hotspots["split"] = hotspots["year"].map(year_split)
print(f"Rows: {len(hotspots):,}")
print("Rows per year:")
print(hotspots["year"].value_counts().sort_index().to_string())

# Audit: missing targets, duplicate IDs, and satellite source.
missing_frp = hotspots["frp"].isna()
print(f"\nMissing FRP: {missing_frp.sum():,} rows, years "
      f"{sorted(hotspots.loc[missing_frp, 'year'].unique())}")
print(f"Duplicate foco_id: {hotspots['foco_id'].duplicated().sum()}")
print(f"Satellites: {hotspots['satelite'].value_counts().to_dict()}")

# Audit: geometria is written as POINT (lat lon); compare it with the coordinate columns.
point_text = hotspots["geometria"].str.removeprefix("POINT (").str.removesuffix(")")
point_parts = point_text.str.split(" ", expand=True).astype(float)
latitude_gap = (point_parts[0] - hotspots["latitude"]).abs()
longitude_gap = (point_parts[1] - hotspots["longitude"]).abs()
mismatched_point = (latitude_gap > 1e-6) | (longitude_gap > 1e-6)
print(f"geometria first value matches latitude in "
      f"{(latitude_gap <= 1e-6).mean():.1%} of rows")
print(f"geometria differs from the coordinate columns in {mismatched_point.sum():,} rows; "
      f"max latitude gap {latitude_gap.max():.5f} deg; "
      f"same rows as missing FRP: {(mismatched_point == missing_frp).all()}")

# Audit: other missing values and implausible dry spells.
for column in ["altitude_m", "precipitacao", "numero_dias_sem_chuva", "risco_fogo",
               "vegetacao"]:
    print(f"Missing {column}: {hotspots[column].isna().sum():,}")
dry_days = hotspots["numero_dias_sem_chuva"]
print(f"Days without rain above 365: {(dry_days > 365).sum():,}; max {dry_days.max():.0f}")

# Audit: calendar days without any detection.
detection_days = hotspots["data_hora_gmt"].dt.normalize().unique()
all_days = pd.date_range("2018-01-01", "2025-12-31", freq="D")
empty_days = all_days.difference(detection_days)
print(f"\nDays with no detections: {len(empty_days)} of {len(all_days)}")
print(f"Empty days in April 2022: {(empty_days.strftime('%Y-%m') == '2022-04').sum()}")

# Audit: rows grouped under the unknown land-cover class.
unknown_cover = hotspots["nome_classe_mb"].eq("Desconhecido")
print(f"Desconhecido land cover: {unknown_cover.sum():,} rows, codes "
      f"{sorted(hotspots.loc[unknown_cover, 'classe_mb'].unique())}")

# Audit: hotspots that share their day and weather values with another hotspot.
weather_key = [hotspots["data_hora_gmt"].dt.date, "temp_c", "rh", "wind_speed", "tp_mm"]
distinct_weather = hotspots.groupby(weather_key, dropna=False).ngroups
print(f"Distinct day-and-weather combinations: {distinct_weather:,}")

# Figure: map of every hotspot, binned into hexagons.
fig, ax = plt.subplots(figsize=(6.2, 6.2))
hexes = ax.hexbin(hotspots["longitude"], hotspots["latitude"], gridsize=110, mincnt=1,
                  cmap="inferno_r", norm=LogNorm())
fig.colorbar(hexes, ax=ax, shrink=0.75, label="Hotspots per cell")
ax.set(title="Hotspots, 2018-2025", xlabel="Longitude", ylabel="Latitude")
ax.set_aspect("equal")
fig.savefig(FIGURES / "hotspot_map.png")
plt.close(fig)

# Figure: how the dataset author joined the four sources (our diagram of the card).
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set(xlim=(0, 12), ylim=(0, 4.6))
ax.axis("off")
sources = [
    ("INPE hotspots\nMODIS on Aqua, ~1 km\nlocation, time, FRP", 3.55),
    ("ERA5 reanalysis\ndaily, 0.25° (~28 km)\nweather + FFWI", 2.45),
    ("MapBiomas Collection 7\n2021 map, 30 m\nland cover", 1.35),
    ("NASA SRTM\n~90 m\naltitude", 0.25),
]
for text, y in sources:
    draw_box(ax, 0.2, y, 3.2, 0.85, text, "#E8EEF7")
    ax.annotate("", xy=(4.6, 2.3), xytext=(3.45, y + 0.42),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2})
draw_box(ax, 4.65, 1.75, 3.0, 1.1, "Match each hotspot\nby location\n(and by day for ERA5)",
         "#FFF4E0")
ax.annotate("", xy=(8.4, 2.3), xytext=(7.7, 2.3),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2})
draw_box(ax, 8.45, 1.6, 3.3, 1.4, "focos_analise\n1,430,756 rows, 33 columns\none row per hotspot",
         "#E9F3EA")
fig.savefig(FIGURES / "pipeline.png")
plt.close(fig)

# Figure: detections per month, colored by the year split used for modeling.
monthly = hotspots.groupby(hotspots["data_hora_gmt"].dt.to_period("M")).size()
month_starts = monthly.index.to_timestamp()
month_colors = [SPLIT_COLORS[year_split(month.year)] for month in month_starts]
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.bar(month_starts, monthly.values / 1000, width=25, color=month_colors)
legend_handles = [Patch(color=color, label=SPLIT_LABELS[split])
                  for split, color in SPLIT_COLORS.items()]
ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=4)
ax.set(ylabel="Hotspots (thousands)", title="Hotspots per month")
ax.set_ylim(0, monthly.max() / 1000 * 1.2)
fig.savefig(FIGURES / "detections_per_month.png")
plt.close(fig)
split_counts = hotspots["split"].value_counts()
print(f"\nSplit counts: {split_counts.to_dict()}")
print(f"Busiest month: {monthly.idxmax()} with {monthly.max():,} hotspots")

# Figure: where and when the hotspots happen (biome counts and month-of-year share).
biome_counts = hotspots["bioma"].value_counts().sort_values()
month_share = hotspots["month"].value_counts(normalize=True).sort_index() * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
axes[0].barh(biome_counts.index, biome_counts.values / 1000, color="#4C72B0")
for position, count in enumerate(biome_counts.values):
    axes[0].text(count / 1000, position, f" {count:,}", va="center", fontsize=11)
axes[0].set(title="Hotspots by biome", xlabel="Hotspots (thousands)")
axes[0].set_xlim(0, biome_counts.max() / 1000 * 1.35)
dry_season = [month in (8, 9, 10) for month in month_share.index]
axes[1].bar(MONTH_NAMES, month_share.values,
            color=["#C44E52" if dry else "#8C8C8C" for dry in dry_season])
axes[1].set(title="Share of hotspots by month", ylabel="% of hotspots")
fig.tight_layout()
fig.savefig(FIGURES / "biome_counts.png")
plt.close(fig)
amazon_cerrado = biome_counts[["Amazônia", "Cerrado"]].sum() / len(hotspots)
print(f"\nBiome counts: {biome_counts.sort_values(ascending=False).to_dict()}")
print(f"Amazônia + Cerrado share: {amazon_cerrado:.1%}")
print(f"August-October share: {month_share.loc[8:10].sum():.1f}%")

# FRP summary on rows that have a target.
with_frp = hotspots.loc[~missing_frp]
frp = with_frp["frp"]
frp_quantiles = frp.quantile([0.5, 0.9, 0.99, 0.999])
top_one_percent = frp.sort_values(ascending=False).head(len(frp) // 100)
print(f"\nFRP min {frp.min():.1f}, median {frp.median():.1f}, mean {frp.mean():.2f}, "
      f"max {frp.max():.1f} MW")
print(f"FRP quantiles: {frp_quantiles.round(1).to_dict()}")
print(f"Rows above 1,000 MW: {(frp > 1000).sum():,}")
print(f"Top 1% of rows hold {top_one_percent.sum() / frp.sum():.1%} of total FRP")

# Figure: FRP distribution on a log scale, overall and by biome.
biome_order = with_frp.groupby("bioma")["frp"].median().sort_values().index.tolist()
biome_values = [with_frp.loc[with_frp["bioma"].eq(biome), "frp"] for biome in biome_order]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
log_bins = np.logspace(np.log10(frp.min()), np.log10(frp.max()), 60)
axes[0].hist(frp, bins=log_bins, color="#4C72B0")
axes[0].axvline(frp.median(), color="#C44E52", linestyle="--",
                label=f"Median {frp.median():.1f} MW")
axes[0].set(xscale="log", title="FRP per hotspot", xlabel="FRP (MW, log scale)",
            ylabel="Hotspots")
axes[0].legend(frameon=False)
axes[1].boxplot(biome_values, orientation="horizontal", showfliers=False, widths=0.6,
                medianprops={"color": "#C44E52"})
axes[1].set_yticks(range(1, len(biome_order) + 1), biome_order)
axes[1].set(xscale="log", title="FRP by biome (whiskers 1.5 IQR, no outliers)",
            xlabel="FRP (MW, log scale)")
fig.tight_layout()
fig.savefig(FIGURES / "frp_distribution.png")
plt.close(fig)
print(f"Median FRP by biome: {with_frp.groupby('bioma')['frp'].median().round(1).to_dict()}")

# Figure: rank correlation of each weather and terrain column with FRP, and FRP by humidity.
correlations = (with_frp[list(WEATHER_LABELS)].corrwith(frp, method="spearman")
                .sort_values())
humidity_bins = pd.qcut(with_frp["rh"], 10)
humidity_medians = with_frp.groupby(humidity_bins, observed=True)["frp"].median()
humidity_centers = with_frp.groupby(humidity_bins, observed=True)["rh"].median()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
bar_colors = ["#C44E52" if value < 0 else "#4C72B0" for value in correlations.values]
axes[0].barh([WEATHER_LABELS[name] for name in correlations.index], correlations.values,
             color=bar_colors)
axes[0].axvline(0, color="#555555", linewidth=1)
for position, value in enumerate(correlations.values):
    alignment = "right" if value < 0 else "left"
    axes[0].text(value, position, f" {value:.2f} ", va="center", ha=alignment, fontsize=11)
axes[0].set(title="Spearman correlation with FRP", xlim=(-0.4, 0.3))
axes[1].plot(humidity_centers.values, humidity_medians.values, marker="o", color="#4C72B0")
axes[1].set(title="Median FRP by humidity decile", xlabel="Relative humidity (%)",
            ylabel="Median FRP (MW)")
axes[1].set_ylim(0, humidity_medians.max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURES / "weather_vs_frp.png")
plt.close(fig)
print(f"\nSpearman with FRP: {correlations.round(3).to_dict()}")
print(f"Median FRP, driest vs. wettest humidity decile: {humidity_medians.iloc[0]:.1f} vs. "
      f"{humidity_medians.iloc[-1]:.1f} MW")
