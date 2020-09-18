"""
Provides helper and utility routines.
"""

# License: MIT

from __future__ import division

from .computation import (calculate_truncated_svd,
                          downsample_data, get_named_regions,
                          get_region_lon_bounds,
                          select_latlon_box,
                          select_named_region, standardized_anomalies)
from .defaults import get_lat_field, get_lon_field, get_time_field
from .eofs import eofs, reofs
from .validation import (check_array_shape, check_base_period,
                         check_data_array, check_unit_axis_sums,
                         datetime_to_string, detect_frequency, get_valid_variables,
                         is_dask_array, is_data_array, is_dataset,
                         is_integer, is_scalar, is_xarray_object)
