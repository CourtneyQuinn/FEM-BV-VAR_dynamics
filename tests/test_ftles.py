"""
Provides unit tests for calculation of FTLEs.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np

from clustering_dynamics.dynamics import calculate_FTLEs


def test_ftles_with_constant_jacobian():
    """Test FTLEs calculation gives correct result with constant Jacobian."""

    tolerance = 1e-2

    delta_t = 1.0
    exact_clvs = np.array([[1.0, 1.0 / np.sqrt(3.0), 0.0],
                           [0.0, 1.0 / np.sqrt(3.0), 0.0],
                           [0.0, 1.0 / np.sqrt(3.0), 1.0]])
    D = np.diag([np.exp(delta_t), np.exp(-delta_t), np.exp(-3 * delta_t)])

    A = np.matmul(exact_clvs, np.matmul(D, np.linalg.inv(exact_clvs)))

    T = 1000
    At = np.repeat(A[:, :, np.newaxis], T, axis=-1)

    dim = A.shape[0]

    N = 50
    Nk = np.arange(N + 1)

    ftles = calculate_FTLEs(dim, At, Nk)

    assert ftles.shape == (dim,)
    assert np.abs(ftles[0] - 1) < tolerance
    assert np.abs(ftles[1] + 1) < tolerance
    assert np.abs(ftles[2] + 3) < tolerance


def test_ftles_with_time_dependent_system():
    """Test FTLEs calculation gives correct result for time-dependent system."""

    random_state = np.random.default_rng(0)

    tolerance = 1e-2

    d = 8
    lambdas = np.log(np.arange(1, d + 1)[::-1])
    epsilon = 0.1
    R = np.diag(np.exp(lambdas))

    N = 200
    n_window = 0

    Sn = np.zeros((2 * N + 2 + 2 * n_window, d, d))

    idx = 0
    for n in range(-(N + 1 + n_window), N + n_window + 1):

        if n == -1:
            z = random_state.uniform(size=(d - 1),)
            Sn[idx] = np.eye(d) + np.diag(z, k=-1)
        else:
            Z = random_state.uniform(size=(d, d))
            Sn[idx] = np.eye(d) + epsilon * Z

        idx += 1

    An = np.full((Sn.shape[0], d, d), np.NaN)
    for n in range(1, Sn.shape[0]):
        An[n] = np.matmul(Sn[n], np.matmul(R, np.linalg.inv(Sn[n - 1])))
    An = An[1:]

    Sn = Sn.transpose((1, 2, 0))
    An = An.transpose((1, 2, 0))

    Nk = np.arange(N + 1)

    ftles = calculate_FTLEs(d, An, Nk)

    assert ftles.shape == (d,)
    assert np.allclose(ftles, lambdas, rtol=tolerance, atol=tolerance)
