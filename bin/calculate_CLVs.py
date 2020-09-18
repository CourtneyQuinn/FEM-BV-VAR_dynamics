"""
Calculate covariant Lyapunov vectors for FEM-BV-VARX models of NCEPv1 500 hPa geopotential height anomalies.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

## Packages

from copy import deepcopy
import itertools
import os
import time
import argparse

import cartopy.crs as ccrs
import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.linalg as linalg
import scipy.stats as stats
import pandas as pd

from cartopy.util import add_cyclic_point
from scipy.signal import correlate
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from statsmodels.nonparametric.smoothers_lowess import lowess

from clustering_dynamics.models import FEMBVVARX
from clustering_dynamics.dynamics import *

DEFAULT_N_COMPONENTS = 2
DEFAULT_ORDER = 0
DEFAULT_STATE_LENGTH = 0
DEFAULT_PUSH_FORWARD = None
DEFAULT_N_TRUNC = 15
DEFAULT_TOL = 1e-3

def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Calculate CLVs of FEM-BV-VARX model.")
    
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=DEFAULT_N_COMPONENTS,
                        help='number of FEM-BV-VARX states')
    parser.add_argument('--order', dest='order', type=int,
                        default=DEFAULT_ORDER,
                        help='AR order')
    parser.add_argument('--state-length', dest='state_length', type=float,
                        default=DEFAULT_STATE_LENGTH,
                        help='average length of residence in states')
    
    parser.add_argument('--push-forward', dest='push_forward', nargs='+', type=int,
                        default=DEFAULT_PUSH_FORWARD,
                        help='Push forwards lengths for CLV calculation')
            
    parser.add_argument('--n-trunc', dest='n_trunc', type=float,
                        default=DEFAULT_N_TRUNC,
                        help='CLVs truncation number')

    args = parser.parse_args()
    

    if args.n_components < 1:
        raise ValueError(
            'Number of components must be an integer greater than 1')

    if args.order < 1:
        raise ValueError(
            'AR order must be at least 1 for CLV calculation')

    if args.state_length is not None and args.state_length < 0:
        raise ValueError(
            'State length must be non-negative')

    if args.push_forward is None:
        raise ValueError(
            'At least 1 push forward length must be specified')
        
    if args.n_trunc is not None and args.n_trunc < 0:
        raise ValueError(
            'CLV truncation number must be a non-negative integer')


    return args

## File paths

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
RESULTS_DIR = os.path.join(PROJECT_DIR,'results')
FEM_BV_VAR_DIR = os.path.join(RESULTS_DIR,'fembv_varx_fits')

## Parameters for loading experiments

reanalysis = 'nnr1'
var_name = 'hgt'
var_lev = '500'
var_ext = 'anom'
timespan = '1948_2018'
base_period = [np.datetime64('1979-01-01'), np.datetime64('2018-12-31')]
hemisphere = 'NH'
region = 'atlantic'
season = 'ALL'
pc_scaling = 'unit'
max_eofs = 200
lat_weights = 'scos'

base_period_str = '{}_{}'.format(pd.to_datetime(base_period[0]).strftime('%Y%m%d'),
                                pd.to_datetime(base_period[1]).strftime('%Y%m%d'))


## Load model

args = parse_cmd_line_args()

k = args.n_components
m = args.order
p = args.state_length
n_PCs = 20

model_filename = '.'.join([var_name, var_lev, timespan, base_period_str, 'anom', hemisphere, region, base_period_str,
                           season, 'max_eofs_{:d}'.format(max_eofs), lat_weights, pc_scaling, 'fembv_varx',
                           'n_pcs20','k{:d}'.format(k),'m{:d}'.format(m),'state_length{:d}'.format(p),'nc'])
model_file = os.path.join(FEM_BV_VAR_DIR, model_filename)
model = xr.open_dataset(model_file)

## Compute matrix cocycle

time_len = model.weights.shape[0]-5
state_space = m*n_PCs
A = np.array(model.A)
gammas = np.array(model.weights)

AT = np.matmul(gammas[:,:],A[:,0,:,:].transpose(1, 0, 2)).transpose(0,2,1)
for mm in np.arange(1,m):
    AT = np.concatenate((AT,np.matmul(gammas[:,:],A[:,mm,:,:].transpose(1, 0, 2)).transpose(0,2,1)),axis=1)

I0 = np.concatenate((np.eye(n_PCs*(m-1)),np.zeros((n_PCs*(m-1),n_PCs))),axis=1)
I0 = np.repeat(I0[:, :, np.newaxis], AT.shape[2], axis=2)

matrix_cocycle = np.concatenate((AT,I0),axis=0)
matrix_cocycle = matrix_cocycle[:,:,5:]

## Calculate CLVs

Push_forwards = args.push_forward

print("Calculating truncated CLVs for k = {}, m = {}, p = {}.".format(k, m, p))

for M in Push_forwards:
    Nk = np.arange(0,M+1,1)
    num_CLVs = matrix_cocycle.shape[2]-(2*M)
    n_trunc = args.n_trunc
    CLVs = np.array(np.zeros((state_space,n_trunc,num_CLVs),dtype=np.float))

    start = time.time()
    for t in np.arange(0,num_CLVs):
        CLVs[:,:,t] = calculate_CLV_numerically(state_space,matrix_cocycle[:,:,t:],Nk,M,nCLVs = n_trunc)
    
    end = time.time()
    elapsed = (end-start)/60
    print("Calculated CLVs for push-forward step of M = {}.  Calculation time: {} min".format(M, round(elapsed,3)))

    
    ## convert to xarray
    time_CLVs = model.time[4+M:-M-1]
    
    CLVs = xr.DataArray(CLVs, coords=[np.arange(1,CLVs.shape[0]+1),np.arange(1,CLVs.shape[1]+1), time_CLVs],
                               dims=['coordinate','CLV', 'time'])
    
    ## save to netCDF file
    CLVs_filename = '.'.join([var_name, var_lev, timespan, base_period_str, 'anom', hemisphere, region, season, 
                         'max_eofs_{:d}'.format(max_eofs), lat_weights, pc_scaling, 'm{:d}'.format(m),
                          'state_length{:d}'.format(p),'CLVs', 'M{:d}'.format(M),'orth1','nc'])

    if n_trunc == None:
        CLVS_DIR = os.path.join(FEM_BV_VAR_DIR, 'CLVs')
    else:
        CLVS_DIR = os.path.join(FEM_BV_VAR_DIR, 'CLVs', 'truncated')
    
    if not os.path.exists(CLVS_DIR):
        os.makedirs(CLVS_DIR)
    
    CLVs_file = os.path.join(CLVS_DIR, CLVs_filename)
    
    CLVs.to_dataset(name='CLVs').to_netcdf(CLVs_file)
