"""Set-up routines for clustering dynamics study code."""

from setuptools import setup, find_packages


setup(
    name='clustering_dynamics',
    version='0.0.1',
    description='Code accompanying clustering dynamics study',
    long_description='',
    install_requires=['cftime', 'cvxpy', 'cytoolz', 'dask', 'matplotlib', 'netCDF4',
                      'numpy', 'pandas', 'scikit-learn', 'scipy', 'xarray'],
    setup_requires=['pytest-runner', 'pytest-pylint'],
    tests_require=['pytest', 'pytest-cov', 'pylint'],
    packages=find_packages('src'),
    package_dir={'':'src'},
    test_suite='tests',
    zip_safe=False
)
