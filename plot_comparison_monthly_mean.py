import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FuncFormatter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
ARCHIVE_PATH = Path("/Users/jamesbriant/Documents/Projects/CAM-hybrid/archives")
ARCHIVE_HYBRID_RUN = (
    "F2000climo_30days_lowres_user_nl_cam-Debi_rest10days_hybrid-202511181713"
)
ARCHIVE_STANDARD_RUN = (
    "F2000climo_30days_lowres_user_nl_cam-Debi_rest10days-202509262219"
)
CESM_FILE = "monthly_mean_from_h1.nc"
ERA5_PATH = Path("/Volumes/T7/CESM-GP/ERA5")
ERA5_SFC_FILE = "2D.grib"
ERA5_PL_FILES = ["T_Q_RELHUM_3D.grib", "U_V_OMEGA_3D.grib", "Z3_3D.grib"]
OUTPUT_DIR = "comparison_figures"

# Variable Mappings
VAR_CONFIG = {
    "TS": {
        "obs_var": "skt",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "K",
        "long_name": "Skin Temperature",
        "cmap_mean": "inferno",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "TREFHT": {
        "obs_var": "t2m",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "K",
        "long_name": "2m Temperature",
        "cmap_mean": "inferno",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "PSL": {
        "obs_var": "msl",
        "scale_cesm": 0.01,
        "scale_obs": 0.01,
        "units": "hPa",
        "long_name": "Sea Level Pressure",
        "cmap_mean": "viridis",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "PS": {
        "obs_var": "sp",
        "scale_cesm": 0.01,
        "scale_obs": 0.01,
        "units": "hPa",
        "long_name": "Surface Pressure",
        "cmap_mean": "viridis",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "SST": {
        "obs_var": "sst",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "K",
        "long_name": "Sea Surface Temperature",
        "cmap_mean": "inferno",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "CAPE": {
        "obs_var": "cape",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "J/kg",
        "long_name": "Convective Available Potential Energy",
        "cmap_mean": "viridis",
        "extend_mean": "max",
        "cmap_bias": "RdBu_r",
    },
    "PHIS": {
        "obs_var": "z",
        "scale_cesm": 1 / 9.80665,
        "scale_obs": 1 / 9.80665,
        "units": "m",
        "long_name": "Surface Geopotential Height",
        "cmap_mean": "viridis",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "PRECT": {
        "obs_var": "tp",
        "scale_cesm": 8.64e7,
        "scale_obs": 24000,
        "units": "mm/day",
        "long_name": "Total Precipitation",
        "cmap_mean": "viridis",
        "extend_mean": "max",
        "cmap_bias": "RdBu",
    },
    "T": {
        "obs_var": "t",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "K",
        "long_name": "Temperature",
        "cmap_mean": "inferno",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "Q": {
        "obs_var": "q",
        "scale_cesm": 1000,
        "scale_obs": 1000,
        "units": "g/kg",
        "long_name": "Specific Humidity",
        "cmap_mean": "GnBu",
        "extend_mean": "max",
        "cmap_bias": "RdBu_r",
    },
    "RELHUM": {
        "obs_var": "r",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "%",
        "long_name": "Relative Humidity",
        "cmap_mean": "GnBu",
        "extend_mean": "max",
        "cmap_bias": "RdBu_r",
    },
    "U": {
        "obs_var": "u",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "m/s",
        "long_name": "Zonal Wind",
        "cmap_mean": "RdBu_r",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "V": {
        "obs_var": "v",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "m/s",
        "long_name": "Meridional Wind",
        "cmap_mean": "RdBu_r",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "Z3": {
        "obs_var": "z",
        "scale_cesm": 1,
        "scale_obs": 1 / 9.80665,
        "units": "m",
        "long_name": "Geopotential Height",
        "cmap_mean": "viridis",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
    "OMEGA": {
        "obs_var": "w",
        "scale_cesm": 1,
        "scale_obs": 1,
        "units": "Pa/s",
        "long_name": "Vertical Velocity",
        "cmap_mean": "RdBu_r",
        "extend_mean": "both",
        "cmap_bias": "RdBu_r",
    },
}

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================


def _safe_mean(ds: xr.Dataset) -> xr.Dataset:
    """Computes mean over time/valid_time/step dimensions if they exist."""
    dims = list(ds.dims)
    dims_to_mean = []

    # Identify time dimensions
    if "time" in dims:
        dims_to_mean.append("time")
    if "valid_time" in dims:
        dims_to_mean.append("valid_time")
    if "step" in dims:
        dims_to_mean.append("step")

    if dims_to_mean:
        print(f"      Averaging over dimensions: {dims_to_mean}...")
        return ds.mean(dim=dims_to_mean, keep_attrs=True)
    return ds


def _process_surface(raw_path: Path, processed_path: Path) -> None:
    """
    Process 2D Surface GRIB file.
    Handles the split between 'tp' (Accumulated) and other variables (Instantaneous).
    """
    print(
        f"    [Processing 2D] Splitting TP and Main variables from {raw_path.name}..."
    )

    datasets_to_merge = []

    # 1. Process Precipitation (TP)
    try:
        # Load TP (paramId 228)
        # See https://codes.ecmwf.int/grib/param-db/
        ds_tp = xr.open_dataset(
            raw_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"paramId": 228}},
            chunks={
                "time": 10
            },  # Safe to chunk here as we know 2D structure usually has time
        )

        # Normalize Accumulation to Rate
        if "step" in ds_tp.dims:
            print("      Normalizing TP accumulation to rate...")
            step_hours = ds_tp["step"].astype("float64") / 3.6e12
            ds_tp = ds_tp / step_hours

        datasets_to_merge.append(_safe_mean(ds_tp))

    except Exception as e:
        print(f"      Note: Could not load TP from {raw_path.name} ({e})")

    # 2. Process Main Variables
    # Use paramId to avoid shortName ambiguity (e.g. 10u vs u10)
    # 165: 10u, 166: 10v, 167: 2t, 151: msl, 34: sst, 134: sp, 235: skt, 59: cape, 129: z
    # See https://codes.ecmwf.int/grib/param-db/
    target_ids = [165, 166, 167, 151, 34, 134, 235, 59, 129]
    try:
        ds_main = xr.open_dataset(
            raw_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"paramId": target_ids}},
            chunks={"time": 10},
        )
        print(f"      Loaded main vars: {list(ds_main.data_vars)}")
        datasets_to_merge.append(_safe_mean(ds_main))
    except Exception as e:
        print(f"      Note: Could not load main vars from {raw_path.name} ({e})")

    if not datasets_to_merge:
        raise ValueError(f"Failed to load any data from {raw_path}")

    # Merge and Save
    ds_final = xr.merge(datasets_to_merge)
    print(f"    Saving processed file to {processed_path.name}...")
    ds_final.to_netcdf(processed_path)
    ds_final.close()


def _process_3d(raw_path: Path, processed_path: Path) -> None:
    """
    Process 3D GRIB file.
    Standard load, assumes no 'tp' conflict, handles valid_time vs time.
    """
    print(f"    [Processing 3D] Standard load and average for {raw_path.name}...")

    # Load without chunks first to avoid 'time' dimension errors if 'valid_time' is used
    ds = xr.open_dataset(raw_path, engine="cfgrib", chunks="auto")

    # Compute Mean
    ds_avg = _safe_mean(ds)

    # Save
    print(f"    Saving processed file to {processed_path.name}...")
    ds_avg.to_netcdf(processed_path)
    ds_avg.close()
    ds.close()


def get_file(raw_path: Path, file_type: str = "2d") -> xr.Dataset:
    """
    Orchestrator: Checks for processed file, launches processing if needed, returns dataset.
    file_type: '2d' (Surface) or '3d' (Pressure Levels)
    """
    processed_path = raw_path.parent / f"{raw_path.stem}_processed.nc"

    if not processed_path.exists():
        print(f"  Processing raw file: {raw_path.name}")
        if file_type == "2d":
            _process_surface(raw_path, processed_path)
        else:
            _process_3d(raw_path, processed_path)
    else:
        print(f"  Found pre-processed file: {processed_path.name}")

    # Load the clean NetCDF
    # chunks='auto' is safe because NetCDF structure is known and consistent
    return xr.open_dataset(processed_path, chunks="auto")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def get_bottom_level_pairs(
    ds_cesm: xr.Dataset,
    ds_era5: xr.Dataset,
    n_levels: int = 6,
) -> List[Dict[str, Any]]:
    print(f"  Matching bottom {n_levels} vertical levels (unique ERA5 targets)...")

    # Try to calculate pressure from hybrid coeffs
    if "P0" in ds_cesm and "hyam" in ds_cesm and "hybm" in ds_cesm:
        p0 = ds_cesm["P0"].values
        ps_ref = 101325.0
        cesm_levs_hpa = (ds_cesm["hyam"] * p0 + ds_cesm["hybm"] * ps_ref) / 100.0
    elif "lev" in ds_cesm.coords:
        # Fallback: use 'lev' coordinate directly if it looks like hPa
        print(
            "    Using 'lev' coordinate as approximate pressure (P0/hyam/hybm missing)."
        )
        cesm_levs_hpa = ds_cesm["lev"].values
    else:
        raise ValueError(
            "Cannot determine CESM vertical levels (missing P0/hyam/hybm and lev)."
        )

    # Check ERA5 Vertical Coord name
    if "isobaricInhPa" in ds_era5.coords:
        era5_levs = ds_era5["isobaricInhPa"].values
        era5_coord = "isobaricInhPa"
    elif "level" in ds_era5.coords:
        era5_levs = ds_era5["level"].values
        era5_coord = "level"
    else:
        # Fallback if dimensions were dropped or renamed
        possible_levs = [c for c in ds_era5.coords if "isobaric" in c or "level" in c]
        if possible_levs:
            era5_coord = possible_levs[0]
            era5_levs = ds_era5[era5_coord].values
        else:
            raise ValueError(
                f"ERA5 vertical coordinate not found. Available: {list(ds_era5.coords)}"
            )

    # Handle both DataArray and numpy array for cesm_levs_hpa
    cesm_vals = (
        cesm_levs_hpa.values if hasattr(cesm_levs_hpa, "values") else cesm_levs_hpa
    )

    # 1. Identify available ERA5 levels and sort them (descending pressure = bottom up)
    era5_levs_sorted = np.sort(era5_levs)[::-1]

    # 2. Select the bottom N ERA5 levels
    target_era5_levs = era5_levs_sorted[:n_levels]

    selected = []
    for era5_val in target_era5_levs:
        era5_val = float(era5_val)

        # Find closest CESM level
        cesm_diffs = np.abs(cesm_vals - era5_val)
        cesm_idx = np.argmin(cesm_diffs)
        cesm_val = float(cesm_vals[cesm_idx])

        diff = abs(cesm_val - era5_val)

        print(
            f"    ERA5: {era5_val} hPa -> Closest CESM: {cesm_val:.2f} hPa (Idx: {cesm_idx}, Diff: {diff:.2f})"
        )

        selected.append(
            {
                "cesm_idx": cesm_idx,
                "cesm_val": cesm_val,
                "era5_val": era5_val,
                "era5_coord": era5_coord,
                "diff": diff,
            }
        )

    return selected


def _ensure_latlon_names(
    obj: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    """Ensure coordinates are named `lat` and `lon` on a Dataset/DataArray."""
    rename_map = {}
    if "latitude" in obj.coords and "lat" not in obj.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in obj.coords and "lon" not in obj.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        try:
            obj = obj.rename(rename_map)
        except Exception:
            # If obj is a DataArray, use .to_dataset first
            try:
                obj = obj.to_dataset(name="tmp").rename(rename_map)["tmp"]
            except Exception:
                pass
    return obj


def _to_0_360(obj: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
    """Convert longitudes to 0..360 range and sort if necessary."""
    if "lon" not in obj.coords:
        return obj
    lon = obj["lon"]
    try:
        lon_min = float(lon.min())
    except Exception:
        return obj
    if lon_min < 0:
        newlon = (lon + 360) % 360
        obj = obj.assign_coords(lon=newlon)
        # ensure monotonic lon ordering for interpolation
        obj = obj.sortby("lon")
    return obj


def compute_area_weights(da: xr.DataArray) -> xr.DataArray:
    """Compute cosine latitude weights aligned with `da`'s lat coordinate."""
    if "lat" not in da.coords:
        raise ValueError("DataArray/Dataset has no 'lat' coordinate for weighting")
    lat = da["lat"]
    weights = np.cos(np.deg2rad(lat))
    # Expand weights to 2D (lat, lon) when used with broadcasting
    return xr.DataArray(weights, coords={"lat": lat}, dims=("lat",))


def global_rmse(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """Compute area-weighted global RMSE between two DataArrays (lat,lon).

    Both inputs must share `lat` and `lon` coords and be on the same grid.
    """
    diff = sim - obs
    sq = diff**2
    weights = compute_area_weights(sim)
    # Sum over lat/lon with broadcasting (weights has dim lat)
    # normalize by sum of weights over lat dimension multiplied by lon length
    # Create 2D weights by broadcasting
    if "lon" in sim.coords:
        # Build a 2D weight array by repeating the 1D lat weights across longitudes.
        # Use an outer product so shapes become (nlat, nlon).
        nlon = int(sim["lon"].size)
        w2_np = np.outer(weights.values, np.ones(nlon))
        w2 = xr.DataArray(
            w2_np, coords={"lat": sim["lat"], "lon": sim["lon"]}, dims=("lat", "lon")
        )
    else:
        w2 = weights
    # xarray broadcasting: align weights dims
    try:
        weighted = (sq * w2).sum(dim=("lat", "lon")) / w2.sum(dim=("lat", "lon"))
    except Exception:
        # Fallback: normalize by sum of weights times number of longitudes
        weighted = (sq * weights).sum(dim=("lat")) / weights.sum(dim=("lat",))
        weighted = weighted.mean(dim="lon")
    return float(np.sqrt(weighted.values))


def plot_variable_comparison(
    da_std: xr.DataArray,
    da_hyb: xr.DataArray,
    da_obs: xr.DataArray,
    var_name: str,
    level_name: str,
    output_path: Union[str, Path],
    units: str,
    rmse_std: Optional[float] = None,
    rmse_hyb: Optional[float] = None,
    title_suffix: str = "",
) -> None:
    """Render variable comparison with a layout matching plot_std_precip.py."""
    print(f"  Plotting {var_name} comparison ({level_name}) -> {output_path}...")

    # Use A4 portrait dimensions (approx 8.3 x 11.7 inches)
    fig = plt.figure(figsize=(8.3, 11.7))

    # Get Long Name
    long_name = VAR_CONFIG[var_name].get("long_name", var_name)

    # Add main title with detailed level info
    if title_suffix:
        fig.suptitle(f"{long_name} Comparison\n{title_suffix}", fontsize=12, y=0.98)
    else:
        fig.suptitle(f"{long_name} Comparison", fontsize=12, y=0.98)

    proj = ccrs.PlateCarree()

    # GridSpec: 3 rows, 20 columns to allow fine control over width/centering
    # Increase height ratio for bottom row so the map can expand to fill the width
    gs = fig.add_gridspec(3, 20, height_ratios=[1, 1, 1.5])

    # Common extent
    lon_min = float(da_std["lon"].min())
    lon_max = float(da_std["lon"].max())
    lat_min = float(da_std["lat"].min())
    lat_max = float(da_std["lat"].max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Ticks
    lon_ticks = [-180, -90, 0, 90, 180]
    lat_ticks = [-90, -60, -30, 0, 30, 60, 90]

    def _lon_label(x, pos=None):
        try:
            xv = float(x)
        except Exception:
            return ""
        if xv > 180:
            xv = xv - 360.0
        return f"{int(round(xv))}"

    def _lat_label(y, pos=None):
        try:
            yv = float(y)
        except Exception:
            return ""
        return f"{int(round(yv))}"

    def plot_panel(ax, data, title, cmap, vmin, vmax, extend, unit, aspect=30):
        im = data.plot(
            ax=ax,
            transform=proj,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            add_colorbar=False,
        )
        ax.coastlines()
        ax.set_title(title, fontsize=10)
        try:
            ax.set_extent(extent, crs=proj)
        except Exception:
            pass

        ax.set_xticks(lon_ticks, crs=proj)
        ax.set_yticks(lat_ticks, crs=proj)
        ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
        ax.yaxis.set_major_formatter(FuncFormatter(_lat_label))
        ax.tick_params(labelsize=8)

        ax.set_xlabel("Longitude (°)", fontsize=9)
        ax.set_ylabel("Latitude (°)", fontsize=9)

        # Colorbar attached to axis with sufficient padding to clear x-axis label
        cb = plt.colorbar(
            im, ax=ax, orientation="horizontal", aspect=aspect, pad=0.15, extend=extend
        )
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        return im

    # Ranges
    vmin = min(
        np.nanquantile(da_std, 0.01),
        np.nanquantile(da_hyb, 0.01),
        np.nanquantile(da_obs, 0.01),
    )
    vmax = max(
        np.nanquantile(da_std, 0.99),
        np.nanquantile(da_hyb, 0.99),
        np.nanquantile(da_obs, 0.99),
    )

    # Config
    cmap_mean = VAR_CONFIG[var_name].get("cmap_mean", "viridis")
    extend_mean = VAR_CONFIG[var_name].get("extend_mean", "both")
    cmap_bias = VAR_CONFIG[var_name].get("cmap_bias", "RdBu_r")

    # Row 1: Means (Left: cols 0-9, Right: cols 11-20 -> Gap of 2 cols)
    ax1 = fig.add_subplot(gs[0, 0:9], projection=proj)
    plot_panel(
        ax1, da_std, f"Standard ({var_name})", cmap_mean, vmin, vmax, extend_mean, units
    )

    ax2 = fig.add_subplot(gs[0, 11:20], projection=proj)
    plot_panel(
        ax2, da_hyb, f"Hybrid ({var_name})", cmap_mean, vmin, vmax, extend_mean, units
    )

    # Row 2: Biases
    bias_std = da_std - da_obs
    bias_hyb = da_hyb - da_obs
    diff_max = max(
        abs(np.nanmin(bias_std)),
        abs(np.nanmax(bias_std)),
        abs(np.nanmin(bias_hyb)),
        abs(np.nanmax(bias_hyb)),
    )

    title_std = "Standard - ERA5"
    if rmse_std is not None:
        title_std += f"\nRMSE: {rmse_std:.2f} {units}"

    title_hyb = "Hybrid - ERA5"
    if rmse_hyb is not None:
        title_hyb += f"\nRMSE: {rmse_hyb:.2f} {units}"

    ax3 = fig.add_subplot(gs[1, 0:9], projection=proj)
    plot_panel(
        ax3,
        bias_std,
        title_std,
        cmap_bias,
        -diff_max,
        diff_max,
        "both",
        f"Bias [{units}]",
    )

    ax4 = fig.add_subplot(gs[1, 11:20], projection=proj)
    plot_panel(
        ax4,
        bias_hyb,
        title_hyb,
        cmap_bias,
        -diff_max,
        diff_max,
        "both",
        f"Bias [{units}]",
    )

    # Row 3: Improvement
    abs_std = np.abs(bias_std)
    abs_hyb = np.abs(bias_hyb)
    improve = abs_std - abs_hyb
    imax = max(abs(np.nanmin(improve)), abs(np.nanmax(improve)))

    # Bottom row: Centered 80% width (cols 2-18 out of 20)
    ax5 = fig.add_subplot(gs[2, 2:18], projection=proj)
    plot_panel(
        ax5,
        improve,
        "Improvement (|Std-Obs| - |Hyb-Obs|)",
        "BrBG",
        -imax,
        imax,
        "both",
        f"Improvement [{units}]",
        aspect=40,
    )

    # Adjust spacing: reduce hspace to close vertical gaps
    plt.subplots_adjust(
        wspace=0.5, hspace=0.12, top=0.95, bottom=0.05, left=0.05, right=0.95
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_cesm_avg(path: Union[str, Path]) -> xr.Dataset:
    """Simple helper to load CESM NetCDF and average time."""
    print(f"  Loading CESM: {Path(path).name}...")
    ds = xr.open_dataset(path, chunks={"time": 10})
    return _safe_mean(ds)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(">>> Loading Data...")

    # 1. Load ERA5 Surface
    ds_sfc = get_file(ERA5_PATH / ERA5_SFC_FILE, file_type="2d")

    # 2. Load ERA5 3D Files
    pl_datasets = []
    for f in ERA5_PL_FILES:
        ds = get_file(ERA5_PATH / f, file_type="3d")
        pl_datasets.append(ds)
    ds_pl = xr.merge(pl_datasets)

    # 3. Load CESM runs
    print("\n>>> Loading CESM Data...")
    std_path = ARCHIVE_PATH / ARCHIVE_STANDARD_RUN / "atm" / "hist" / CESM_FILE
    hyb_path = ARCHIVE_PATH / ARCHIVE_HYBRID_RUN / "atm" / "hist" / CESM_FILE

    if not std_path.exists():
        print(f"  Standard CESM file not found: {std_path}")
        return
    if not hyb_path.exists():
        print(f"  Hybrid CESM file not found: {hyb_path}")
        return

    ds_cesm_std = load_cesm_avg(std_path)
    ds_cesm_hyb = load_cesm_avg(hyb_path)

    print("  CESM Standard vars:", list(ds_cesm_std.data_vars))

    # 4. Loop over variables
    print("\n>>> Starting Variable Loop...")

    missing_vars = []

    for var_name, config in VAR_CONFIG.items():
        print(f"\n--- Processing {var_name} ---")

        if var_name not in ds_cesm_std or var_name not in ds_cesm_hyb:
            print(f"  {var_name} not in CESM output. Skipping.")
            missing_vars.append(f"{var_name} (Missing in CESM)")
            continue

        # Get CESM DataArrays
        da_std_raw = ds_cesm_std[var_name]
        da_hyb_raw = ds_cesm_hyb[var_name]

        # Identify if 3D or 2D
        is_3d = "lev" in da_std_raw.dims or "ilev" in da_std_raw.dims

        if is_3d:
            ds_obs_source = ds_pl
        else:
            ds_obs_source = ds_sfc

        obs_var = config["obs_var"]
        da_obs_raw = ds_obs_source[obs_var]
        da_obs_raw = _ensure_latlon_names(da_obs_raw)
        da_obs_raw = _to_0_360(da_obs_raw)

        # Define levels to plot
        levels_to_plot = []  # List of (cesm_slice, hyb_slice, obs_slice, level_label, title_suffix)

        if not is_3d:
            # 2D Surface Variable
            da_std = da_std_raw * config["scale_cesm"]
            da_hyb = da_hyb_raw * config["scale_cesm"]
            da_obs = da_obs_raw * config["scale_obs"]

            # Regrid Obs to CESM
            da_obs_regrid = da_obs.interp(
                lon=da_std.lon, lat=da_std.lat, method="linear"
            )

            levels_to_plot.append((da_std, da_hyb, da_obs_regrid, "Surface", ""))

        else:
            # 3D Variable - Find matching levels
            # We need to match CESM levels to ERA5 levels
            try:
                # Use ds_cesm_std for vertical grid info
                # Plot bottom 6 levels
                pairs = get_bottom_level_pairs(ds_cesm_std, ds_obs_source, n_levels=10)
            except Exception as e:
                print(f"  Could not match levels for {var_name}: {e}")
                missing_vars.append(f"{var_name} (Level matching failed: {e})")
                continue

            for p in pairs:
                cesm_idx = p["cesm_idx"]
                cesm_val = p["cesm_val"]
                era5_val = p["era5_val"]
                era5_coord = p["era5_coord"]

                # Slice CESM
                # Assuming 'lev' dimension index
                if "lev" in da_std_raw.dims:
                    da_std = da_std_raw.isel(lev=cesm_idx) * config["scale_cesm"]
                    da_hyb = da_hyb_raw.isel(lev=cesm_idx) * config["scale_cesm"]
                elif "ilev" in da_std_raw.dims:
                    da_std = da_std_raw.isel(ilev=cesm_idx) * config["scale_cesm"]
                    da_hyb = da_hyb_raw.isel(ilev=cesm_idx) * config["scale_cesm"]
                else:
                    print(f"  Unknown vertical dim for {var_name}. Skipping.")
                    continue

                # Slice ERA5
                da_obs = (
                    da_obs_raw.sel({era5_coord: era5_val}, method="nearest")
                    * config["scale_obs"]
                )

                # Regrid Obs
                da_obs_regrid = da_obs.interp(
                    lon=da_std.lon, lat=da_std.lat, method="linear"
                )

                label = f"{int(era5_val)}hPa"
                title_suffix = f"ERA5: {int(era5_val)} hPa | CESM: {cesm_val:.1f} hPa"
                levels_to_plot.append(
                    (da_std, da_hyb, da_obs_regrid, label, title_suffix)
                )

        # Plot each level
        for da_s, da_h, da_o, lev_label, t_suffix in levels_to_plot:
            # Compute RMSE
            try:
                rmse_std = global_rmse(da_s, da_o)
                rmse_hyb = global_rmse(da_h, da_o)
            except Exception as e:
                print(f"  Error computing RMSE for {var_name} {lev_label}: {e}")
                rmse_std = None
                rmse_hyb = None

            outname = f"{OUTPUT_DIR}/{var_name}_{lev_label}_standard_vs_hybrid.png"
            plot_variable_comparison(
                da_s,
                da_h,
                da_o,
                var_name,
                lev_label,
                outname,
                config["units"],
                rmse_std,
                rmse_hyb,
                title_suffix=t_suffix,
            )

    print("\n========================================")
    print("SUMMARY OF MISSING / SKIPPED VARIABLES")
    print("========================================")
    if missing_vars:
        for v in missing_vars:
            print(f" - {v}")
    else:
        print(" All variables processed successfully.")
    print("========================================")


if __name__ == "__main__":
    main()
