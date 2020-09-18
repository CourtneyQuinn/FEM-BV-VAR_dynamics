"""
Provides helper routines for calculations.
"""

# License: MIT

from __future__ import absolute_import, division

import dask.array as da
import numpy as np
import pandas as pd
import scipy.linalg as sl
import xarray as xr

from scipy.sparse.linalg import svds

from .defaults import get_lat_field, get_lon_field, get_time_field
from .validation import check_base_period, is_dask_array


def _fix_svd_phases(u, vh):
    """Impose fixed phase convention on left- and right-singular vectors.

    Given a set of left- and right-singular vectors as the columns of u
    and rows of vh, respectively, imposes the phase convention that for
    each left-singular vector, the element with largest absolute value
    is real and positive.

    Parameters
    ----------
    u : array, shape (M, K)
        Unitary array containing the left-singular vectors as columns.

    vh : array, shape (K, N)
        Unitary array containing the right-singular vectors as rows.

    Returns
    -------
    u_fixed : array, shape (M, K)
        Unitary array containing the left-singular vectors as columns,
        conforming to the chosen phase convention.

    vh_fixed : array, shape (K, N)
        Unitary array containing the right-singular vectors as rows,
        conforming to the chosen phase convention.
    """

    n_cols = u.shape[1]
    max_elem_rows = np.argmax(np.abs(u), axis=0)

    if np.any(np.iscomplexobj(u)):
        phases = np.exp(-1j * np.angle(u[max_elem_rows, range(n_cols)]))
    else:
        phases = np.sign(u[max_elem_rows, range(n_cols)])

    u *= phases
    vh *= phases[:, np.newaxis]

    return u, vh


def calculate_truncated_svd(X, k):
    """Calculate the truncated SVD of a 2D array.

    Given an array X with shape (M, N), the SVD of X is computed and the
    leading K = min(k, min(M, N)) singular values are retained.

    The singular values are returned as a 1D array in non-increasing
    order, and the singular vectors are defined such that the array X
    is decomposed as ```u @ np.diag(s) @ vh```.

    Parameters
    ----------
    X : array, shape (M, N)
        The matrix to calculate the SVD of.

    k : integer
        Number of singular values to retain in truncated decomposition.
        If k > min(M, N), then all singular values are retained.

    Returns
    -------
    u : array, shape (M, K)
        Unitary array containing the retained left-singular vectors of X
        as columns.

    s : array, shape (K)
        Array containing the leading K singular vectors of X.

    vh : array, shape (K, N)
        Unitary array containing the retained right-singular vectors of
        X as rows.
    """

    max_modes = min(X.shape)

    if is_dask_array(X):
        dsvd = da.linalg.svd(X)

        u, s, vh = (x.compute() for x in dsvd)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vh = vh[:min(k, len(s)), :]
    elif k < max_modes:
        u, s, vh = svds(X, k=k)

        # Note that svds returns the singular values with the
        # opposite (i.e., non-decreasing) ordering convention.
        u = u[:, ::-1]
        s = s[::-1]
        vh = vh[::-1]
    else:
        u, s, vh = sl.svd(X, full_matrices=False)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vh = vh[:min(k, len(s)), :]

    # Impose a fixed phase convention on the singular vectors
    # to avoid phase ambiguity.
    u, vh = _fix_svd_phases(u, vh)

    return u, s, vh


def downsample_data(data, frequency=None, time_name=None):
    """Perform down-sampling of data."""

    if frequency is None:
        return data

    if frequency not in ('daily', 'monthly'):
        raise ValueError('Unrecognized down-sampling frequency %r' %
                         frequency)

    time_name = time_name if time_name is not None else get_time_field(data)

    current_frequency = pd.infer_freq(data[time_name].values[:3])
    target_frequency = '1D' if frequency == 'daily' else '1MS'

    current_timestep = (pd.to_datetime(data[time_name].values[0]) +
                        pd.tseries.frequencies.to_offset(current_frequency))
    target_timestep = (pd.to_datetime(data[time_name].values[0]) +
                       pd.tseries.frequencies.to_offset(target_frequency))

    if target_timestep < current_timestep:
        raise ValueError('Downsampling frequency appears to be higher'
                         ' than current frequency')

    return data.resample({time_name: target_frequency}).mean(time_name)


def get_named_regions(hemisphere):
    """Get names of predefined region for given hemisphere."""

    if hemisphere == 'WG':
        return ['all']

    if hemisphere == 'NH':
        return ['all', 'eurasia', 'pacific', 'atlantic', 'atlantic_eurasia']

    if hemisphere == 'SH':
        return ['all', 'indian', 'south_america', 'pacific',
                'full_pacific', 'australian']

    raise RuntimeError("Invalid hemisphere '%s'" % hemisphere)


def get_region_lon_bounds(hemisphere, region):
    """Get longitude bounds associated with a given region."""

    if hemisphere in ('WG', 'NH'):

        if region == 'all':
            return np.array([0, 360])

        if region == 'atlantic':
            return np.array([250, 360])

        if region == 'atlantic_eurasia':
            return np.array([250, 120])

        if region == 'eurasia':
            return np.array([0, 120])

        if region == 'pacific':
            return np.array([120, 250])

        raise ValueError("Invalid region '%s'" % region)

    if hemisphere == 'SH':

        if region == 'all':
            return np.array([0, 360])

        if region == 'australian':
            return np.array([110, 210])

        if region == 'full_pacific':
            return np.array([150, 300])

        if region == 'indian':
            return np.array([0, 120])

        if region == 'pacific':
            return np.array([120, 250])

        if region == 'south_america':
            return np.array([240, 360])

        raise ValueError("Invalid region '%s'" % region)

    raise ValueError("Invalid hemisphere '%s'" % hemisphere)


def select_latlon_box(data, lat_bounds, lon_bounds,
                      lat_name=None, lon_name=None):
    """Select data in given latitude-longitude box."""

    lat_name = lat_name if lat_name is not None else get_lat_field(data)
    lon_name = lon_name if lon_name is not None else get_lon_field(data)

    if len(lat_bounds) != 2:
        raise ValueError('Latitude bounds must be a list of length 2')

    if len(lon_bounds) != 2:
        raise ValueError('Longitude bounds must be a list of length 2')

    lat_bounds = np.array(lat_bounds)
    lon_bounds = np.array(lon_bounds)

    if data[lat_name].values[0] > data[lat_name].values[-1]:
        lat_bounds = lat_bounds[::-1]

    region_data = data.sel({lat_name : slice(lat_bounds[0], lat_bounds[1])})

    lon = data[lon_name]

    if np.any(lon_bounds < 0) & np.any(lon < 0):
        # Both bounds and original coordinates are given in convention
        # [-180, 180]

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 180)) |
                        ((lon >= -180) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    elif np.any(lon_bounds < 0) & ~np.any(lon < 0):
        # Bounds are given in convention [-180, 180] but data is in
        # convention [0, 360] -- convert bounds to [0, 360]
        lon_bounds = np.where(lon_bounds < 0, lon_bounds + 360, lon_bounds)

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 360)) |
                        ((lon >= 0) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    elif ~np.any(lon_bounds < 0) & np.any(lon < 0):
        # Bounds are given in convention [0, 360] but data is in
        # convention [-180, 180] -- convert bounds to [-180, 180]
        lon_bounds = np.where(lon_bounds > 180, lon_bounds - 360, lon_bounds)

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 180)) |
                        ((lon >= -180) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    else:
        # Bounds and data are given in convention [0, 360]

        if lon_bounds[0] > lon_bounds[1]:
            lon_mask = (((lon >= lon_bounds[0]) & (lon <= 360)) |
                        ((lon >= 0) & (lon <= lon_bounds[1])))
        else:
            lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    region_data = region_data.where(lon_mask, drop=True)

    return region_data


def select_named_region(data, hemisphere, region, lat_name=None, lon_name=None):
    """Select data within named region."""

    if hemisphere == 'WG':
        lat_bounds = [-90.0, 90.0]
    elif hemisphere == 'NH':
        lat_bounds = [0.0, 90.0]
    elif hemisphere == 'SH':
        lat_bounds = [-90.0, 0.0]
    else:
        raise ValueError("Invalid hemisphere '%s'" % hemisphere)

    lon_bounds = get_region_lon_bounds(hemisphere, region)

    return select_latlon_box(data, lat_bounds=lat_bounds, lon_bounds=lon_bounds,
                             lat_name=lat_name, lon_name=lon_name)


def standardized_anomalies(data, base_period=None, standardize_by=None,
                           time_name=None):
    """Calculate standardized anomalies."""

    time_name = time_name if time_name is not None else get_time_field(data)

    base_period = check_base_period(data, base_period=base_period,
                                    time_name=time_name)

    base_period_da = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    if standardize_by == 'dayofyear':
        base_period_groups = base_period_da[time_name].dt.dayofyear
        groups = data[time_name].dt.dayofyear
    elif standardize_by == 'month':
        base_period_groups = base_period_da[time_name].dt.month
        groups = data[time_name].dt.month
    elif standardize_by == 'season':
        base_period_groups = base_period_da[time_name].dt.season
        groups = data[time_name].dt.season
    else:
        base_period_groups = None
        groups = None

    if base_period_groups is not None:

        clim_mean = base_period_da.groupby(base_period_groups).mean(time_name)
        clim_std = base_period_da.groupby(base_period_groups).std(time_name)

        std_anom = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s, data.groupby(groups),
            clim_mean, clim_std, dask='allowed')

    else:

        clim_mean = base_period_da.mean(time_name)
        clim_std = base_period_da.std(time_name)

        std_anom = ((data - clim_mean) / clim_std)

    return std_anom
