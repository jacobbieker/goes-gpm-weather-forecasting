import xarray as xr
import numpy as np


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in d.keys()],
        coords={"variable": list(d.keys())},
    ).astype(np.float32)


# Compute for 5000 random timesteps for 2017-2019
IMERG_MIN = 0.0
IMERG_MAX = 120.0
IMERG_MEAN = 0.11074574291706085
IMERG_STD = 0.8968361020088196

# Computed for 2000 random timesteps from 2017-2019
ERA5_MINS = {
    "2m_temperature": 193.7,
    "10m_u_component_of_wind": -36.66,
    "10m_v_component_of_wind": -35.32,
}
ERA5_MAXS = {
    "2m_temperature": 325.1,
    "10m_u_component_of_wind": 32.97,
    "10m_v_component_of_wind": 36.04,
}
ERA5_MEANS = {
    "2m_temperature": 279.1,
    "10m_u_component_of_wind": -0.04658,
    "10m_v_component_of_wind": 0.2025,
}
ERA5_STD = {
    "2m_temperature": 21.15,
    "10m_u_component_of_wind": 5.605,
    "10m_v_component_of_wind": 4.792,
}

ERA5_MINS = _to_data_array(ERA5_MINS)
ERA5_MAXS = _to_data_array(ERA5_MAXS)
ERA5_MEANS = _to_data_array(ERA5_MEANS)
ERA5_STD = _to_data_array(ERA5_STD)

GOES_MAX = 16384.0
GOES_MIN = 0.0
