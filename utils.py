import xarray as xr
import pystac_client
import fsspec
import planetary_computer
import datetime as dt
import rioxarray
import numpy as np
import pyproj
from typing import Union

from consts import GOES_MAX, GOES_MIN, ERA5_MINS, ERA5_MAXS


def normalize_era5(ds):
    ds["2m_temperature"] = (ds["2m_temperature"] - ERA5_MINS.sel(variable="2m_temperature")) / (
        ERA5_MAXS.sel(variable="2m_temperature") - ERA5_MINS.sel(variable="2m_temperature")
    )
    ds["10m_u_component_of_wind"] = (
        ds["10m_u_component_of_wind"] - ERA5_MINS.sel(variable="10m_u_component_of_wind")
    ) / (
        ERA5_MAXS.sel(variable="10m_u_component_of_wind")
        - ERA5_MINS.sel(variable="10m_u_component_of_wind")
    )
    ds["10m_v_component_of_wind"] = (
        ds["10m_v_component_of_wind"] - ERA5_MINS.sel(variable="10m_v_component_of_wind")
    ) / (
        ERA5_MAXS.sel(variable="10m_v_component_of_wind")
        - ERA5_MINS.sel(variable="10m_v_component_of_wind")
    )
    return ds


def fourier_features(
    lat: float, lon: float, time_of_year: float, time_of_day: float, num_harmonics: int = 5
) -> np.ndarray:
    """
    Calculate the fourier features for a given lat/lon and time of year and time of day
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        time_of_year: Time of year, in fraction of the days in year
        time_of_day: Time of day, in fraction of the day since the start of the day
        num_harmonics: Number of harmonics to use

    Returns:
        Array of fourier features
    """
    # Calculate the fourier features
    features = []
    for i in range(1, num_harmonics + 1):
        features.append(np.sin(i * 2 * np.pi * time_of_year))
        features.append(np.cos(i * 2 * np.pi * time_of_year))
        features.append(np.sin(i * 2 * np.pi * time_of_day))
        features.append(np.cos(i * 2 * np.pi * time_of_day))
    # Add the lat/lon sin and cos
    features.append(np.sin(lat))
    features.append(np.cos(lat))
    features.append(np.sin(lon))
    features.append(np.cos(lon))
    return np.array(features)


def get_goes_full_image(
    start_datetime: dt.datetime,
    end_datetime: dt.datetime,
    image_type: str = "CONUS",
    max_images: int = 4,
) -> xr.Dataset:
    """
    Get the GOES images for the given time period

    Args:
        start_datetime: datetime to start getting images from
        end_datetime: End datetime to get images from
        image_type: Which GOES image type to get, either 'CONUS' or 'FULL DISK'
        max_images: Max number of images to get

    Returns:
        Xarray Dataset containing the GOES images
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["goes-cmi"],
        datetime=[start_datetime, end_datetime],
        limit=max_images,
        query={"goes:image-type": {"eq": image_type}},
    )
    timesteps = []
    for i, item in enumerate(search.items()):
        if i >= max_images:
            break
        bands = []
        for idx in range(1, 7):
            bands.append(f"C{idx:02d}_2km")
        common_names = [
            item.assets[band].extra_fields["eo:bands"][0]["common_name"]
            for band in bands
            if "common_name" in item.assets[band].extra_fields["eo:bands"][0]
        ]
        ds = xr.concat(
            [rioxarray.open_rasterio(item.assets[band].href) for band in bands], dim="band"
        ).assign_coords(band=common_names)
        # Add created date as a coordinate
        ds = ds.assign_coords(
            {"time": dt.datetime.strptime(ds.attrs["date_created"], "%Y-%m-%dT%H:%M:%S.%fZ")}
        )
        timesteps.append(ds)
    if len(timesteps) != max_images:  # Only want to return if all images are available
        return None
    ds = xr.concat(timesteps, dim="time")
    ds = ds.sortby("time").transpose("time", "band", "x", "y")
    # Add lat/lon coordinates
    ds = calc_latlon(ds)
    # Add CRS to main attributes
    ds.attrs["crs"] = ds.rio.crs
    return ds


def get_goes_image_cutout(
    goes: xr.DataArray,
    lat_lon: list[float],
    image_size: int = 400,
    add_fourier_features: bool = True,
    normalize: bool = False,
) -> Union[np.ndarray, None]:
    """
    Get a cutout of the GOES image, centered on the given lat/lon

    Args:
        goes: Xarray Dataset containing the GOES images
        lat_lon: Center lat/lon of the cutout
        image_size: Image size in pixels
        normalize: Whether to normalize the data to be between 0 and 1
        add_fourier_features: Whether to add fourier features to the cutout

    Returns:
        Numpy array of the cutout, with multiple time steps if available
    """
    # Get center lat/lon pixel
    x_index, y_index = get_nearest_x_y_index_from_latlon(goes, lat_lon[0], lat_lon[1])
    # Get the bounding box of the cutout
    lat1 = y_index - image_size // 2
    lat2 = y_index + image_size // 2
    lon1 = x_index - image_size // 2
    lon2 = x_index + image_size // 2
    # Get the cutout
    cutout = goes.isel(x=slice(lat1, lat2), y=slice(lon1, lon2))
    # If the majority of the pixels are -1, then the cutout is outside the field of view, so return None
    if (cutout == -1).sum() > 0.5 * cutout.size:
        return None
    # If any of the dimensions are 0, then the cutout is outside the field of view, so return None
    if 0 in cutout.shape:
        return None
    # Normalize if wanted
    if normalize:
        cutout = (cutout - GOES_MIN) / (GOES_MAX - GOES_MIN)

    cutout = cutout.transpose("band", "time", "x", "y")
    # Add fourier features if needed
    if add_fourier_features:
        # Get the time of year and time of day of the most recent image in the GOES data
        time_of_year = cutout.time.dt.dayofyear[-1] / 366
        time_of_day = cutout.time.dt.hour[-1] / 24 + cutout.time.dt.minute[-1] / 60
        # Calculate the fourier features
        features = fourier_features(lat_lon[0], lat_lon[1], time_of_year, time_of_day)
        features = np.expand_dims(features, axis=[1, 2, 3])
        # Convert the goes data to numpy array
        cutout = cutout.data
        # Add the fourier features to the numpy array
        # Tile the fourier features to match the x and y dimensions of the cutout
        # Expand the dimensions to match the number of time steps
        features = np.tile(features, (1, cutout.shape[1], cutout.shape[2], cutout.shape[3]))
        cutout = np.concatenate([cutout, features], axis=0)
    else:
        # Convert the goes data to numpy array
        cutout = cutout.data
    return cutout


def setup_imerg() -> xr.Dataset:
    """
    Setup the IMERG dataset from Planetary Computer

    Returns:
        Xarray Dataset containing the IMERG data
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    asset = catalog.get_collection("gpm-imerg-hhr").assets["zarr-abfs"]
    fs = fsspec.get_mapper(asset.href, **asset.extra_fields["xarray:storage_options"])
    ds = xr.open_zarr(fs, **asset.extra_fields["xarray:open_kwargs"])
    datetimeindex = ds.indexes["time"].to_datetimeindex()
    ds["time"] = datetimeindex
    ds = ds.sel(time=slice(dt.datetime(2017, 2, 28), None))
    # Transpose to (time, lat, lon)
    ds = ds.transpose("time", "lat", "lon", ...)
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    # The datetime is in the Julian Calendar, so need to convert to a normal DatetimeIndex
    ds = ds[["precipitationCal"]].chunk({"time": 1})
    return ds


def setup_era5_reanalysis() -> xr.Dataset:
    """
    Setup the ERA5 reanalysis dataset from WeatherBench2

    Returns:
        Xarray Dataset containing the ERA5 reanalysis data

    """
    data = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        chunks={"time": 1},
        consolidated=True,
    )
    data = data.drop_vars("level")
    # Select the variables we want, 2m_temperature, and 10m_wind_speed
    data = data[["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]]
    # GOES images are only available from 2017-02-28 onwards, so we need to filter out the IMERG data before that
    data = data.sel(time=slice(dt.datetime(2017, 2, 28), None))
    return data


def setup_era5_forecast(forecast_timesteps: int = 2) -> xr.Dataset:
    """
    Setup the ERA5 forecast dataset from WeatherBench2

    Args:
        forecast_timesteps: Number of forecast timesteps to include

    Returns:
        Xarray Dataset containing the ERA5 forecast data
    """
    data = xr.open_zarr(
        "gs://weatherbench2/datasets/era5-forecasts/2020-1440x721.zarr", chunks={"time": 1}
    )
    data = data[["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]]
    data = data.isel(prediction_timedelta=slice(0, forecast_timesteps))
    return data


def get_imerg_images(
    imerg_dataarray: xr.Dataset,
    lat_lon: list[float],
    end_datetime: dt.datetime,
    max_images: int = 4,
    image_size: int = 400,
) -> np.ndarray:
    """
    Get the last 4 images from the IMERG data, in a 16x16 crop centered on the given lat/lon
    Args:
        imerg_dataarray: Dataarray containing the IMERG data
        lat_lon: Lat/Lon center of the crop
        end_datetime: End datetime to select data
        max_images: Maximum number of images to return
        image_size: Image size in pixels

    Returns:
        Numpy containing the IMERG data
    """
    ds: xr.DataArray = imerg_dataarray.sel(time=slice(None, end_datetime))
    # Get the pixel closest to the lat/lon
    lat_index, lon_index = get_nearest_latlon_index(ds, lat_lon[0], lat_lon[1])
    # Get the bounding box of the cutout
    ds = ds.isel(
        lat=slice(lat_index - image_size // 2, lat_index + image_size // 2),
        lon=slice(lon_index - image_size // 2, lon_index + image_size // 2),
    )
    # Get the last 4 images, as they are in reverse chronological order
    ds = ds.isel(time=slice(-max_images, None))
    # Convert to numpy array
    ds: np.ndarray = ds["precipitationCal"].transpose("time", "lat", "lon").data
    return ds


def get_nearest_latlon_index(ds: xr.Dataset, lat: float, lon: float) -> tuple[float, float]:
    """Get the nearest index into lat/lon to the given lat/lon"""
    lat_index = np.abs(ds.latitude.data - lat).argmin()
    lon_index = np.abs(ds.longitude.data - lon).argmin()
    return lat_index, lon_index


def get_era5_reanalysis(
    era5_reanalysis: xr.Dataset,
    lat_lon: list[float,],
    start_datetime: dt.datetime,
    end_datetime: dt.datetime,
    normalize: bool = False,
    time_resolution_hours: int = 1,
    max_frames: int = 1,
    image_size: int = 16,
) -> np.ndarray:
    """
    Get a 16x16 crop of the reanalysis, the closest to the given lat/lon

    Args:
        era5_reanalysis: Xarray Dataset containing the ERA5 reanalysis data
        lat_lon: Lat/lon of the center of the crop
        start_datetime: Start datetime to select data
        end_datetime: End datetime to select data
        normalize: Whether to normalize the data to be between 0 and 1
        time_resolution_hours: Time resolution in hours to select data
        max_frames: Maximum number of frames to return
        image_size: Size of the image to return

    Returns:
        Numpy array containing the reanalysis data in the 16x16 crop
    """
    ds: xr.Dataset = era5_reanalysis.sel(time=slice(start_datetime, end_datetime))
    # Add 180 to the longitude, as the ERA5 data is from 0 to 360, and the GOES data is from -180 to 180
    lat, lon = lat_lon
    lon = lon + 180
    lat_index, lon_index = get_nearest_latlon_index(ds, lat, lon)
    ds = ds.isel(
        latitude=slice(lat_index - image_size // 2, lat_index + image_size // 2),
        longitude=slice(lon_index - image_size // 2, lon_index + image_size // 2),
    )
    # Subselect the time resolution
    ds = ds.sel(time=ds.time.dt.hour.isin(range(0, 24, time_resolution_hours)))
    # Select the last N frames if there are more than N frames
    ds = ds.isel(time=slice(-max_frames, None))
    if normalize:
        ds = normalize_era5(ds)
    # Convert to numpy array and return
    ds: np.ndarray = (
        ds.to_array(dim="variable").transpose("variable", "time", "latitude", "longitude").data
    )

    return ds


def convert_x_y_to_lat_lon(crs: str, lon: list[float], lat: list[float]) -> tuple[float, float]:
    """Convert the given x/y coordinates to lat/lon in the given CRS"""
    transformer = pyproj.Transformer.from_crs(crs, "epsg:4326")
    xs, ys = transformer.transform(lon, lat)
    return xs, ys


def convert_lat_lon_to_x_y(crs: str, x: list[float], y: list[float]) -> tuple[float, float]:
    """Convert the given lat/lon to x/y coordinates in the given CRS"""
    transformer = pyproj.Transformer.from_crs("epsg:4326", crs)
    lons, lats = transformer.transform(x, y)
    return lons, lats


def get_nearest_x_y_index_from_latlon(ds: xr.Dataset, lat: float, lon: float) -> tuple[int, int]:
    """Get the nearest x and y index into the dataset from the given lat/lon"""
    x, y = convert_lat_lon_to_x_y(ds.rio.crs, y=[lat], x=[lon])
    # Now get the index of the closests x and y to the returned x and y
    x_index = np.abs(ds.x.data - x).argmin()
    y_index = np.abs(ds.y.data - y).argmin()
    return x_index, y_index


def calc_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the latitude and longitude coordinates for the given dataset

    Args:
        ds: Xarray Dataset to calculate the lat/lon coordinates for, with x and y coordinates

    Returns:
        Xarray Dataset with the latitude and longitude coordinates added
    """
    XX, YY = np.meshgrid(ds.x.data, ds.y.data)
    lons, lats = convert_x_y_to_lat_lon(ds.rio.crs, XX, YY)
    # Check if lons and lons_trans are close in value
    # Set inf to NaN values
    lons[lons == np.inf] = np.nan
    lats[lats == np.inf] = np.nan

    ds = ds.assign_coords({"latitude": (["y", "x"], lats), "longitude": (["y", "x"], lons)})
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["units"] = "degrees_east"
    return ds
