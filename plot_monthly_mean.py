import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.util import add_cyclic_point


def _plot_latlon_map(
    data: xr.DataArray,
    title: str,
    outpath: Path,
    cmap: str = "viridis",
    levels: int | list | None = 20,
    extend: str = "both",
):
    """Create and save a lat/lon map for `data` using Cartopy.

    This function will attempt to add a cyclic point on the longitude
    dimension to avoid the dateline seam, then plot with an
    appropriate PlateCarree transform.
    """
    fig, ax = plt.subplots(
        figsize=(12, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Try to add a cyclic point along the longitude axis to avoid seam
    data_to_plot = data
    try:
        lon_idx = list(data.dims).index("lon")
        vals, lons = add_cyclic_point(
            data.values, coord=data["lon"].values, axis=lon_idx
        )
        coords = {
            dim: (data[dim].values if dim != "lon" else lons) for dim in data.dims
        }
        data_to_plot = xr.DataArray(vals, coords=coords, dims=data.dims)
        data_to_plot.attrs = data.attrs
    except Exception:
        # If something unexpected happens (unusual dims, missing lon), plot original
        data_to_plot = data

    # Extract metadata
    long_name = data.attrs.get("long_name", data.name)
    units = data.attrs.get("units", "")

    # Use PlateCarree transform so Cartopy knows coordinates are regular lon/lat
    data_to_plot.plot.contourf(
        ax=ax,
        levels=levels,
        cmap=cmap,
        extend=extend,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
            "label": units,
            "orientation": "horizontal",
            "pad": 0.08,
            "shrink": 0.8,
        },
    )
    ax.set_title(f"{title}\n{long_name}", fontsize=14)
    ax.coastlines(linewidth=1.2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def _plot_zonal_mean(
    zonal: xr.DataArray,
    title: str,
    outpath: Path,
    cmap: str = "viridis",
    levels: int | list | None = 20,
    extend: str = "both",
):
    """Plot and save a zonal-mean (latitude vs level) figure.

    Expects `zonal` to be a DataArray averaged over longitude (dims include 'lev' and 'lat').
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract metadata
    long_name = zonal.attrs.get("long_name", zonal.name)
    units = zonal.attrs.get("units", "")

    zonal.plot.contourf(
        ax=ax,
        y="lev",
        levels=levels,
        cmap=cmap,
        extend=extend,
        cbar_kwargs={"label": units},
    )
    ax.invert_yaxis()
    ax.set_title(f"{title}\n{long_name}", fontsize=14)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Level / Pressure")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_variables(ds, output_dir: Path):
    """
    Plots the variables in the dataset and saves them to disk.
    - For 3D vars (time, lev, lat, lon):
        - Plots Zonal Mean (Lat vs Lev).
        - Plots 2D Map at surface (bottom) level (Lat vs Lon).
    - For 2D vars (time, lat, lon): Plots Map (Lat vs Lon).
    """
    output_dir.mkdir(exist_ok=True)

    print("Generating plots...")

    # We iterate over the data variables in the dataset
    for var_name in ds.data_vars:
        var_data = ds[var_name]

        # Skip if it's not spatial data (e.g. time bounds)
        if "lat" not in var_data.dims:
            continue

        # Select the first time step (January mean in this specific 8-day case)
        # .squeeze() removes dimensions of size 1 (like time after selection if kept)
        data_slice = var_data.isel(time=0)

        # Convert Kelvin to Celsius for specific variables
        if var_name in ["TS", "SST", "TREFHT"]:
            # Preserve attributes after arithmetic
            attrs = data_slice.attrs
            data_slice = data_slice - 273.15
            data_slice.attrs = attrs
            data_slice.attrs["units"] = "C"

        # Determine colorbar extension (arrows)
        if var_name in ["PRECT", "Q", "RELHUM", "CAPE", "PS", "PSL", "PHIS"]:
            extend = "max"
        else:
            extend = "both"

        # Choose plotting path depending on variable dimensionality
        if "lev" in data_slice.dims:
            # 3D atmosphere variable: make a zonal mean and save a zonal plot
            zonal_mean = data_slice.mean(dim="lon", keep_attrs=True)
            outpath = output_dir / f"{var_name}_monthly_avg.png"
            _plot_zonal_mean(
                zonal_mean,
                f"{var_name} - Zonal Mean (January)",
                outpath,
                extend=extend,
            )

        elif "lat" in data_slice.dims and "lon" in data_slice.dims:
            # 2D surface variable: make a lat/lon map and save
            outpath = output_dir / f"{var_name}_monthly_avg.png"
            _plot_latlon_map(
                data_slice,
                f"{var_name} - Surface Map (January)",
                outpath,
                extend=extend,
            )

        else:
            print(f"Skipping {var_name}: dimensions {data_slice.dims} not supported.")
            continue

        # If the variable has a vertical coordinate, also save a surface-level map
        if "lev" in data_slice.dims:
            lev_max = data_slice.lev.max().item()
            surface_data = data_slice.sel(lev=lev_max)  # Select lowest level
            surface_data.attrs = data_slice.attrs  # Ensure attrs are preserved
            outpath2 = output_dir / f"{var_name}_surface_map_monthly_avg.png"
            _plot_latlon_map(
                surface_data,
                f"{var_name} - Surface Map at Lowest Level (January)",
                outpath2,
                extend=extend,
            )


def main(run_name: Path, archive_path: Path):
    # --- Configuration ---
    root = archive_path / run_name / "atm"
    hist_dir = root / "hist"
    fig_dir = root / "figures"

    # Output filename based on run name
    output_file = hist_dir / "monthly_mean_from_h1.nc"

    # Find all h1 history files in the hist directory and merge them
    h1_files = sorted(hist_dir.glob("*.cam.h1.*.nc"))
    if not h1_files:
        if output_file.exists():
            print(
                f"No h1 files found. Using existing processed file: {output_file.name}"
            )
            ds_monthly = xr.open_dataset(output_file)
            plot_variables(ds_monthly, output_dir=fig_dir)
            print("Done successfully.")
            return
        else:
            print(f"Error: No 'h1' history files found in {hist_dir}")
            return

    # List of "Important" variables to average.
    # Selected based on standard atmospheric state and flux analysis:
    # Surface:
    #   TS      : Surface Temperature
    #   PS      : Surface Pressure
    #   PSL     : Sea Level Pressure
    #   PRECT   : Total Precipitation Rate
    #   SST     : Sea Surface Temperature
    #   TREFHT  : Reference Height Temperature (2m Temp)
    #   CAPE    : Convective Available Potential Energy
    #   PHIS    : Surface Geopotential Height
    # 3D Atmosphere:
    #   T       : Temperature
    #   Q       : Specific Humidity
    #   U       : Zonal Wind
    #   V       : Meridional Wind
    #   Z3      : Geopotential Height
    #   OMEGA   : Vertical Velocity
    #   RELHUM  : Relative Humidity
    variables_to_process = [
        "TS",
        "PS",
        "PSL",
        "PRECT",
        "SST",
        "TREFHT",
        "CAPE",
        "PHIS",
        "T",
        "Q",
        "U",
        "V",
        "Z3",
        "OMEGA",
        "RELHUM",
    ]

    # --- Processing ---
    print(f"Opening dataset(s): {len(h1_files)} files found")
    for p in h1_files:
        print(f"  - {p.name}")

    try:
        if len(h1_files) == 1:
            ds = xr.open_dataset(h1_files[0], chunks={"time": 10})
        else:
            # Open multiple history files as a single dataset concatenated by coordinates
            ds = xr.open_mfdataset(
                [str(p) for p in h1_files], combine="by_coords", chunks={"time": 10}
            )
    except Exception as e:
        print(f"Error opening dataset(s): {e}")
        return

    # Filter dataset to only the variables we care about (plus coordinates)
    available_vars = set(ds.data_vars.keys())
    selected_vars = [v for v in variables_to_process if v in available_vars]

    missing_vars = set(variables_to_process) - available_vars
    if missing_vars:
        print(
            f"Warning: The following requested variables were not found in the file: {missing_vars}"
        )

    print(f"Processing variables: {selected_vars}")
    ds_subset = ds[selected_vars]

    print("Calculating monthly averages...")
    # Resample to '1MS' (1 Month Start frequency).
    ds_monthly = ds_subset.resample(time="1MS").mean(dim="time", keep_attrs=True)

    # Update attributes to reflect the processing
    ds_monthly.attrs["history"] = (
        f"Created by calculating monthly means from {run_name.name} (merged h1 files)"
    )
    ds_monthly.attrs["description"] = "Monthly time-averaged climatology"

    # --- Saving Data ---
    print(f"Saving result to: {output_file}")
    ds_monthly.to_netcdf(output_file)

    # --- Plotting ---
    # Pass the computed monthly averages to the plotting function
    plot_variables(ds_monthly, output_dir=fig_dir)

    print("Done successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and plot monthly mean climatology from CAM output."
    )
    parser.add_argument(
        "run_name", type=Path, help="Path to the archived run root directory."
    )
    parser.add_argument(
        "--archive_path",
        type=Path,
        default=Path("/data/ucakjcb/archives"),
        help="Base path to archives",
    )
    args = parser.parse_args()
    main(
        run_name=args.run_name,
        archive_path=args.archive_path,
    )
