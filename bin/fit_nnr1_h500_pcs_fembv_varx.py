"""
Fit FEM-BV-VARX model to PCs of NCEPv1 500 hPa geopotential height anomalies.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state

from clustering_dynamics.models import aic, aicc, bic, FEMBVVARX


DEFAULT_N_FEATURES = 20
DEFAULT_N_COMPONENTS = 2
DEFAULT_ORDER = 0
DEFAULT_STATE_LENGTH = None
DEFAULT_PRESAMPLE_LENGTH = None
DEFAULT_TOL = 1e-3
DEFAULT_MAX_ITER = 1000
DEFAULT_REG_COVAR = 1e-6
DEFAULT_N_INIT = 20
DEFAULT_RANDOM_SEED = None
DEFAULT_LOSS = 'least_squares'
DEFAULT_TIME_NAME = 'time'
DEFAULT_COMPONENT_NAME = 'component'
REQUIRE_MONOTONIC_COST_DECREASE = True
WEIGHTS_SOLVER_KWARGS = {
    'solver': 'ECOS',
    'max_iters': 10000,
    'abstol': 1e-7,
    'reltol': 1e-15
    }


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Fit FEM-BV-VARX model to PCs data.")

    parser.add_argument('input_file', help='file containing PCs data')
    parser.add_argument('output_file', help='file to write fitted model to')
    parser.add_argument('--n-features', dest='n_features', type=int,
                        default=DEFAULT_N_FEATURES,
                        help='number of principal components to retain')
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=DEFAULT_N_COMPONENTS,
                        help='number of FEM-BV-VARX states')
    parser.add_argument('--order', dest='order', type=int,
                        default=DEFAULT_ORDER,
                        help='AR order')
    parser.add_argument('--state-length', dest='state_length', type=float,
                        default=DEFAULT_STATE_LENGTH,
                        help='average length of residence in states')
    parser.add_argument('--presample-length', dest='presample_length', type=int,
                        default=DEFAULT_PRESAMPLE_LENGTH,
                        help='presample length')
    parser.add_argument('--tol', dest='tol', type=float,
                        default=DEFAULT_TOL, help='convergence tolerance')
    parser.add_argument('--max-iter', dest='max_iter', type=int,
                        default=DEFAULT_MAX_ITER,
                        help='maximum number of iterations allowed')
    parser.add_argument('--reg-covar', dest='reg_covar', type=float,
                        default=DEFAULT_REG_COVAR,
                        help='regularization for variances')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=DEFAULT_N_INIT,
                        help='number of initializations')
    parser.add_argument('--fit-period-start', dest='fit_period_start',
                        default=None, help='start of fit period')
    parser.add_argument('--fit-period-end', dest='fit_period_end',
                        default=None, help='end of fit period')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=DEFAULT_RANDOM_SEED,
                        help='random seed')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='if given, generate verbose output')
    parser.add_argument('--loss', dest='loss',
                        choices=['log_likelihood', 'least_squares'],
                        default=DEFAULT_LOSS,
                        help='type of loss function')
    parser.add_argument('--pickle', dest='pickle', action='store_true',
                        help='if given, also pickle fitted model')
    parser.add_argument('--standardize', dest='standardize', action='store_true',
                        help='if given, standardize PC values')
    parser.add_argument('--standardize-by', dest='standardize_by',
                        choices=['dayofyear', 'month', 'season'],
                        default='dayofyear',
                        help='grouping interval to use for standardization')
    parser.add_argument('--cross-validate', dest='cross_validate',
                        action='store_true',
                        help='if given, perform cross-validated fits')
    parser.add_argument('--n-folds', dest='n_folds', type=int,
                        default=10, help='number of cross-validation folds')
    parser.add_argument('--time-name', dest='time_name',
                        default=DEFAULT_TIME_NAME,
                        help='name of time dimension')
    parser.add_argument('--component-name', dest='component_name',
                        default=DEFAULT_COMPONENT_NAME,
                        help='name of component dimension')

    args = parser.parse_args()

    if args.n_features < 1:
        raise ValueError(
            'Number of principal components must be at least 1')

    if args.n_components < 1:
        raise ValueError(
            'Number of components must be an integer greater than 1')

    if args.order < 0:
        raise ValueError(
            'AR order must be a non-negative integer')

    if args.state_length is not None and args.state_length < 0:
        raise ValueError(
            'State length must be non-negative')

    if args.presample_length is not None and args.presample_length < 0:
        raise ValueError(
            'Presample length must be a non-negative integer')

    if args.tol <= 0:
        raise ValueError(
            'Stopping tolerance must be positive')

    if args.max_iter < 1:
        raise ValueError(
            'Maximum number of iterations must be at least 1')

    if args.reg_covar < 0:
        raise ValueError(
            'Covariance regularization must be non-negative')

    if args.n_init < 1:
        raise ValueError(
            'Number of initializations must be at least 1')

    if args.cross_validate and args.n_folds < 2:
        raise ValueError(
            'Number of cross-validation folds must be at least 2')

    return args


def get_principal_components(input_file, fit_period=None,
                             standardize=False, standardize_by=None,
                             time_name=DEFAULT_TIME_NAME,
                             pcs_name='principal_components'):

    with xr.open_dataset(input_file) as eofs_ds:
        pcs_da = eofs_ds[pcs_name].astype('f8')

        if pcs_da.get_axis_num(time_name) != 0:
            pcs_da = pcs_da.transpose(
                *([time_name] + [d for d in pcs_da.dims if d != time_name]))

        if fit_period is not None:
            if fit_period[0] is None:
                fit_period_start = pcs_da[time_name].min().values
            else:
                fit_period_start = fit_period[0]

            if fit_period[1] is None:
                fit_period_end = pcs_da[time_name].max().values
            else:
                fit_period_end = fit_period[1]

        else:
            fit_period_start = pcs_da[time_name].min().values
            fit_period_end = pcs_da[time_name].max().values

        fit_period = [fit_period_start, fit_period_end]

        pcs_da = pcs_da.where(
            (pcs_da[time_name] >= fit_period[0]) &
            (pcs_da[time_name] <= fit_period[1]), drop=True)

        if standardize:

            if standardize_by is not None:

                means = pcs_da.groupby(
                    time_name + '.' + standardize_by).mean(time_name)
                stds = pcs_da.groupby(
                    time_name + '.' + standardize_by).std(time_name)

                pcs_da = xr.apply_ufunc(
                    lambda x, m, s: (x - m) / s,
                    pcs_da.groupby(time_name + '.' + standardize_by), means, stds)

            else:
                means = pcs_da.mean(time_name)
                stds = pcs_da.std(time_name)

                pcs_da = (pcs_da - means) / stds

        return pcs_da, fit_period


def fit_cross_validated_fembv_varx_model(endog, exog=None, n_folds=10, **kwargs):
    """Evaluate cross-validation metrics for fitted model."""

    model = FEMBVVARX(**kwargs)

    tscv = TimeSeriesSplit(n_splits=n_folds)

    test_fold_sizes = np.zeros((n_folds,))
    test_costs = np.zeros((n_folds,))
    test_log_likelihoods = np.zeros((n_folds,))
    test_rmses = np.zeros((n_folds,))

    fold = 0
    for train_indices, test_indices in tscv.split(endog):

        train_data = endog[train_indices]
        test_data = endog[test_indices]

        if exog is not None:
            train_exog = exog[train_indices]
            test_exog = exog[test_indices]
        else:
            train_exog = None
            test_exog = None

        train_weights = model.fit_predict(train_data, exog=train_exog)
        test_weights, test_cost, test_log_like, test_rmse = model.predict(
            test_data, exog=test_exog)

        test_fold_sizes[fold] = len(test_indices)
        test_costs[fold] = test_cost
        test_log_likelihoods[fold] = test_log_like
        test_rmses[fold] = test_rmse

        fold += 1

    cv_result = xr.Dataset(
        {
            'test_fold_size': (['fold'], test_fold_sizes),
            'test_cost': (['fold'], test_costs),
            'test_log_likelihood': (['fold'], test_log_likelihoods),
            'test_rmse': (['fold'], test_rmses)
        },
        coords={'fold': np.arange(n_folds)})

    return cv_result


def fit_fembv_varx_model(endog, dates, exog=None, time_name=DEFAULT_TIME_NAME,
                         cross_validate=False, n_folds=10, **kwargs):
    """Fit FEM-BV-VARX model."""

    if cross_validate:
        cv_result = fit_cross_validated_fembv_varx_model(
            endog, exog=exog, n_folds=n_folds, **kwargs)
    else:
        cv_result = None

    n_samples, n_features = endog.shape

    model = FEMBVVARX(**kwargs)

    weights = model.fit_predict(endog, exog=exog)

    # sanity check
    if model._max_tv_norm is not None:

        tv_norms = np.sum(np.abs(np.diff(weights, axis=0)), axis=0)

        if np.any(tv_norms > 1.01 * model._max_tv_norm):
            raise RuntimeError(
                'TV-norm constraint is not satisfied: '
                'required TV-norm < %.5f but solution has TV-norm=%.5f' %
                (model._max_tv_norm, np.max(tv_norms)))

    n_components = model.n_components
    presample_length = model.presample_length
    max_order = model.max_order_

    intcpt_da = xr.DataArray(
        model.means_,
        coords={'fembv_state': np.arange(n_components),
                'endog': np.arange(n_features)},
        dims=['fembv_state', 'endog'])

    covariances_da = xr.DataArray(
        model.covariances_,
        coords={'fembv_state': np.arange(n_components),
                'endog': np.arange(n_features),
                'lagged_endog': np.arange(n_features)},
        dims=['fembv_state', 'endog', 'lagged_endog'])

    precisions_da = xr.DataArray(
        model.precisions_,
        coords={'fembv_state': np.arange(n_components),
                'endog': np.arange(n_features),
                'lagged_endog': np.arange(n_features)},
        dims=['fembv_state', 'endog', 'lagged_endog'])

    # Pad with NaNs for presample values
    weights = np.vstack(
        [np.full((presample_length, n_components), np.NaN), weights])
    weights_da = xr.DataArray(
        weights,
        coords={time_name: dates, 'fembv_state': np.arange(n_components)},
        dims=[time_name, 'fembv_state'])

    data_vars = {'mu': intcpt_da,
                 'covariances': covariances_da,
                 'precisions': precisions_da,
                 'weights': weights_da}

    if model.endog_coef_ is not None:

        endog_coefs_da = xr.DataArray(
            model.endog_coef_,
            coords={'fembv_state': np.arange(n_components),
                    'lag': np.arange(1, max_order + 1),
                    'endog': np.arange(n_features),
                    'lagged_endog': np.arange(n_features)},
            dims=['fembv_state', 'lag', 'endog', 'lagged_endog'])

        data_vars['A'] = endog_coefs_da

    if model.exog_coef_ is not None:
        n_exog = model.exog_coef_.shape[-1]

        exog_coefs_da = xr.DataArray(
            model.exog_coef_,
            coords={'fembv_state': np.arange(n_components),
                    'endog': np.arange(n_features),
                    'exog': np.arange(n_features)},
            dims=['fembv_state', 'endog', 'exog'])

        data_vars['B0'] = exog_coefs_da

    else:
        n_exog = 0

    df = (n_components * n_features + np.sum(model.orders) * n_features ** 2 +
          n_components * n_features * n_exog)
    df += (n_components - 1) * np.sum(np.abs(np.diff(weights[presample_length:, :-1], axis=0)))

    if cv_result is not None:
        for v in cv_result.data_vars:
            data_vars[v] = cv_result[v]

    model_ds = xr.Dataset(data_vars)

    model_ds.attrs['n_components'] = '{:d}'.format(n_components)
    model_ds.attrs['max_order'] = '{:d}'.format(max_order)
    model_ds.attrs['presample_length'] = '{:d}'.format(model.presample_length)
    model_ds.attrs['reg_covar'] = '{:16.8e}'.format(model.reg_covar)
    model_ds.attrs['cost'] = '{:16.8e}'.format(model.cost_)
    model_ds.attrs['log_likelihood'] = '{:16.8e}'.format(model.log_likelihood_)
    model_ds.attrs['df'] = '{:16.8e}'.format(df)
    model_ds.attrs['aic_df'] = '{:16.8e}'.format(aic(model.log_likelihood_, df))
    model_ds.attrs['aicc_df'] = '{:16.8e}'.format(
        aicc(model.log_likelihood_, df, n_samples - model.presample_length))
    model_ds.attrs['bic_df'] = '{:16.8e}'.format(
        bic(model.log_likelihood_, df, n_samples - model.presample_length))
    model_ds.attrs['n_iter'] = '{:d}'.format(model.n_iter_)
    model_ds.attrs['n_init'] = '{:d}'.format(model.n_init)
    model_ds.attrs['tol'] = '{:16.8e}'.format(model.tol)
    model_ds.attrs['max_iter'] = '{:d}'.format(model.max_iter)
    model_ds.attrs['warm_start'] = '{}'.format(model.warm_start)
    model_ds.attrs['converged'] = '{}'.format(model.converged_)
    if model.state_length is None:
        model_ds.attrs['state_length'] = '0'
    else:
        model_ds.attrs['state_length'] = '{:16.8e}'.format(model.state_length)
    if model._max_tv_norm is not None:
        model_ds.attrs['max_tv_norm'] = '{:16.8e}'.format(model._max_tv_norm)
    else:
        model_ds.attrs['max_tv_norm'] = 'inf'

    return model_ds, model


def main():
    """Fit FEM-BV-VARX model to PCs data."""

    args = parse_cmd_line_args()

    random_state = check_random_state(args.random_seed)

    if args.fit_period_start is not None:
        fit_period_start = np.datetime64(args.fit_period_start)
    else:
        fit_period_start = None

    if args.fit_period_end is not None:
        fit_period_end = np.datetime64(args.fit_period_end)
    else:
        fit_period_end = None

    pcs_da, fit_period = get_principal_components(
        args.input_file, fit_period=[fit_period_start, fit_period_end],
        standardize=args.standardize, standardize_by=args.standardize_by,
        time_name=args.time_name)

    pcs = pcs_da.isel({args.component_name: slice(0, args.n_features)}).data

    model_ds, fitted_model = fit_fembv_varx_model(
        pcs,
        dates=pcs_da[args.time_name].data,
        time_name=args.time_name,
        n_components=args.n_components,
        orders=args.order,
        state_length=args.state_length,
        presample_length=args.presample_length,
        tol=args.tol, max_iter=args.max_iter, reg_covar=args.reg_covar,
        n_init=args.n_init, random_state=random_state, warm_start=False,
        verbose=args.verbose, weights_solver_kwargs=WEIGHTS_SOLVER_KWARGS,
        require_monotonic_cost_decrease=REQUIRE_MONOTONIC_COST_DECREASE,
        loss=args.loss, cross_validate=args.cross_validate,
        n_folds=args.n_folds)

    model_ds.attrs['input_file'] = args.input_file
    model_ds.attrs['fit_period_start'] = '{}'.format(
        pd.to_datetime(fit_period[0]).strftime('%Y%m%d'))
    model_ds.attrs['fit_period_end'] = '{}'.format(
        pd.to_datetime(fit_period[1]).strftime('%Y%m%d'))

    if args.pickle:

        basename, ext = os.path.splitext(args.output_file)
        pickle_output = os.path.join(basename, '.pickle')

        with open(pickle_output, 'wb') as ofs:
            pickle.dump(fitted_model, ofs, protocol=pickle.HIGHEST_PROTOCOL)

    model_ds.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
