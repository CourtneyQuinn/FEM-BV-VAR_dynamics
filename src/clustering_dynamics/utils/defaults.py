"""
Provides helper routines for handling defaults.
"""

# License: MIT

from __future__ import absolute_import, division


def get_time_field(da):
    """Get name of time coordinate."""

    default_time_names = ['time', 'initial_time0_hours']

    for d in default_time_names:

        if d in da.dims:
            return d

    raise RuntimeError('Could not deduce time coordinate for given data')


def get_lat_field(da):
    """Get name of latitude coordinate."""

    default_lat_names = ['lat', 'lat_2', 'g0_lat_2']

    for d in default_lat_names:

        if d in da.dims:
            return d

    raise RuntimeError('Could not deduce latitude coordinate for given data')


def get_lon_field(da):
    """Get name of longitude coordinate."""

    default_lon_names = ['lon', 'lon_2', 'g0_lon_3']

    for d in default_lon_names:

        if d in da.dims:
            return d

    raise RuntimeError('Could not deduce longitude coordinate for given data')
