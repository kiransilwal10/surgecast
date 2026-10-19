"""Figures for the README: network map, skill curves, storm case studies, animation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # maps fall back to plain lat/lon axes
    ccrs = None

EXTENT = (-98.5, -66.0, 24.0, 45.5)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COLORS = {
    "Tide table": "#8c959f", "Persistence": "#bf8700", "Ridge (per gauge)": "#9a6700",
    "GRU (no graph)": "#0969da", "GRU + storm node": "#54aeff",
    "GNN (no storm node)": "#8250df", "SurgeCast": "#cf222e",
}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def map_axes(fig, rect=111, extent=EXTENT):
    if ccrs is None:
        ax = fig.add_subplot(rect)
        ax.set_xlim(extent[:2]), ax.set_ylim(extent[2:])
        ax.set_aspect(1 / np.cos(np.deg2rad(35)))
        return ax, {}
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f4f6", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#ffffff")
    ax.coastlines("50m", linewidth=0.5, color="#8c959f")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="#d0d7de")
    ax.grid(False)
    return ax, {"transform": ccrs.PlateCarree()}


def network_map(lat, lon, src, dst, path: Path, tracks=None):
    fig = plt.figure(figsize=(7.5, 6))
    ax, tf = map_axes(fig)
    for s, d in zip(src, dst):
        if s < d:
            ax.plot([lon[s], lon[d]], [lat[s], lat[d]], color="#8250df", lw=0.6, alpha=0.5, **tf)
    if tracks:
        for name, tlat, tlon in tracks:
            ax.plot(tlon, tlat, color="#cf222e", lw=1.0, alpha=0.6, **tf)
            # label where the storm comes closest to the network (about landfall)
            d = (tlat[:, None] - lat) ** 2 + ((tlon[:, None] - lon) * np.cos(np.deg2rad(30))) ** 2
            i = int(np.argmin(d.min(1)))
            ax.text(tlon[i] + 0.3, tlat[i] - 0.6, name, color="#cf222e", fontsize=8,
                    clip_on=True, **tf)
    ax.scatter(lon, lat, s=14, color=INK, zorder=3, **tf)
    ax.set_title(f"{len(lat)} NOAA tide gauges, {len(src) // 2} graph edges", loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


def skill_curves(results: dict, path: Path, models: list[str]):
    leads = results["leads"]
    titles = {"all": "All test hours (2022–2025)", "hurricane": "Hurricane within 500 km",
              "extreme": "Top 1% surge anomalies"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, subset in zip(axes, titles):
        table = results["rmse_cm"][subset]
        for name in models:
            ys = [table[name]["by_lead"][str(l)] if str(l) in table[name]["by_lead"]
                  else table[name]["by_lead"][l] for l in leads]
            ax.plot(leads, ys, marker="o", ms=3.5, lw=1.8 if name == "SurgeCast" else 1.2,
                    color=COLORS.get(name, INK), label=name)
        ax.set_title(titles[subset], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("forecast lead (h)")
        ax.set_xticks(leads)
    axes[0].set_ylabel("RMSE of water level (cm)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def case_studies(cases: list[dict], path: Path, lead: int):
    """Each case: title, times, observed, and a dict of forecasts valid at ``times``."""
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 2.4 * len(cases)), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        ax.plot(case["times"], case["observed"], color=INK, lw=2.2, label="observed")
        for name, series in case["forecasts"].items():
            main = name == "SurgeCast"
            ax.plot(case["times"], series, color=COLORS.get(name, MUTED), lw=2.0 if main else 1.0,
                    ls="-" if main else "--", alpha=1.0 if main else 0.9, label=name)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(case["title"], loc="left", fontsize=10, color=INK)
        ax.set_ylabel("above tide (m)")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0, 0].legend(ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.suptitle(f"Forecasts issued {lead} h before the time shown", x=0.01, ha="left",
                 fontsize=9, color=MUTED, y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def storm_animation(times, lat, lon, observed, forecast, storm_lat, storm_lon, title,
                    path: Path, lead: int, extent, vmax: float = 2.0):
    """Side-by-side gauge maps: observed surge vs. SurgeCast forecast issued ``lead`` h earlier."""
    fig = plt.figure(figsize=(10.5, 4.3))
    axes, tfs = [], []
    for k, rect in enumerate((121, 122)):
        ax, tf = map_axes(fig, rect, extent)
        axes.append(ax), tfs.append(tf)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.86, bottom=0.08, wspace=0.04)
    axes[0].set_title("Observed surge", loc="left", color=INK)
    axes[1].set_title(f"SurgeCast, {lead} h ahead", loc="left", color=INK)
    scat = [ax.scatter(lon, lat, c=np.zeros_like(lat), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       s=60, edgecolor=INK, linewidth=0.4, zorder=3, **tf)
            for ax, tf in zip(axes, tfs)]
    track = [ax.plot([], [], color="#cf222e", lw=1.2, **tf)[0] for ax, tf in zip(axes, tfs)]
    eye = [ax.plot([], [], marker="o", ms=9, mfc="none", mec="#cf222e", mew=2, **tf)[0]
           for ax, tf in zip(axes, tfs)]
    stamp = fig.text(0.01, 0.02, "", color=MUTED)
    fig.colorbar(scat[1], cax=fig.add_axes([0.915, 0.18, 0.012, 0.6]), label="surge above normal (m)")
    fig.suptitle(title, x=0.01, ha="left", color=INK, y=0.98)

    def draw(i):
        for k, values in enumerate((observed[i], forecast[i])):
            scat[k].set_array(np.ma.masked_invalid(values))
            ok = np.isfinite(storm_lat[: i + 1])
            track[k].set_data(storm_lon[: i + 1][ok], storm_lat[: i + 1][ok])
            eye[k].set_data([storm_lon[i]] if ok[-1] else [], [storm_lat[i]] if ok[-1] else [])
        stamp.set_text(times[i].strftime("%Y-%m-%d %H:%M UTC"))
        return scat + track + eye + [stamp]

    anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def influence_map(lat, lon, influence, target, title, path: Path, extent):
    fig = plt.figure(figsize=(6.5, 5.2))
    ax, tf = map_axes(fig, 111, extent)
    w = influence / influence.max()
    order = np.argsort(w)
    sc = ax.scatter(lon[order], lat[order], c=w[order], s=20 + 180 * w[order], cmap="magma_r",
                    vmin=0, vmax=1, edgecolor=INK, linewidth=0.3, zorder=3, **tf)
    ax.scatter([lon[target]], [lat[target]], marker="*", s=260, color="#0969da",
               edgecolor="white", zorder=4, **tf)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="relative influence on forecast")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    fig.savefig(path)
    plt.close(fig)
