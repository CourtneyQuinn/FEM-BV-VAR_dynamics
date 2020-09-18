FEM-BV-VAR dynamics
===================
[![DOI](https://zenodo.org/badge/296521603.svg)](https://zenodo.org/badge/latestdoi/296521603)

This repository contains code and notebooks for
analyzing the dynamics of FEM-BV-VAR models of
the North Atlantic Oscillation.

Routines to fit FEM-BV-VAR models, and to calculate
the dynamical properties of the resulting
reduced-order models, are provided in the
`clustering_dynamics` package. Notebooks
performing the EOF and dynamics analyses
are provided under the `notebooks/` directory,
and scripts demonstrating fitting the FEM-BV-VAR
models and calculating covariant Lyapunov vectors
are provided under the `bin/` directory.

To install from source, run:

    python setup.py install

It is recommended that the package be installed into a custom
environment. For example, to install into a custom conda
environment, first create the environment via

    conda create -n clustering-dynamics-env python=3.7
    conda activate clustering-dynamics-env

The package may then be installed using

    cd /path/to/package/directory
    python setup.py install
    
Additional packages required to run the provided jupyter notebooks 
can be installed in the same enviroment using

    conda install cartopy jupyter seaborn statsmodels -c conda-forge

Optionally, a set of unit tests may be run by executing

    python setup.py test


