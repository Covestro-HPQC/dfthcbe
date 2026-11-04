r"""Tests for :mod:`dfthcbe.utils`, on the decomposed Hamiltonians in ``data/dfthc``."""

import json
from pathlib import Path

import numpy as np
import pytest

from dfthcbe import (
    hamiltonian_matrix, majoranas, reconstruct, walk_e_sos, walk_inputs, walk_integrals,
)

DATA = Path(__file__).parents[1] / 'data' / 'dfthc'
KEYS = ('U', 'W', 'w_B', 'w_plus', 'u_plus', 'w_minus', 'u_minus')
# How far each decomposed Hamiltonian is from the true one, in FCI energy (Ha): the tolerance of
# every comparison with its FCI numbers. The STO-3G factors are exact.
FACTORS_ERROR = {
    'naphthalene': 1e-5, 'h2_sto3g_R2_B2_C1': 1e-10, 'h2_631g_R2_B2_C2': 3e-2,
    'h2_631g_R2_B3_C3': 1e-3,
}

def load_factors(name):
    """``(factors, integrals, summary)`` of ``data/dfthc/<name>``."""
    return (np.load(DATA / name / 'factors.npz'), np.load(DATA / name / 'integrals.npz'),
            json.loads((DATA / name / 'summary.json').read_text()))


def random_factors(n, num_r, num_b, num_c, seed):
    """Random DFTHC factors as in ``factors.npz``, the D1/Q1 split from a random ``h^(1)'``."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((num_r, num_b, n))
    h1 = rng.standard_normal((n, n))
    e, v = np.linalg.eigh(h1 + h1.T)
    return dict(
        U=U / np.linalg.norm(U, axis=2, keepdims=True),
        W=rng.standard_normal((num_r, num_b, num_c)), w_B=rng.standard_normal((num_r, num_c)),
        w_plus=e[e >= 0], u_plus=v[:, e >= 0].T, w_minus=-e[e < 0], u_minus=v[:, e < 0].T,
    )


@pytest.fixture(scope='module', params=list(FACTORS_ERROR))
def case(request):
    return request.param, *load_factors(request.param)


# --- reconstruct: the physical Hamiltonian ------------------------------------------


def test_reconstruct_has_the_factors_error(case):
    """``eri`` misses the true integrals by the factors' ``pqrs_error``, with their 8-fold
    symmetry."""
    _, f, ints, summary = case
    h1e, eri = reconstruct(f['U'], f['W'], f['h_sym'], f['h1'])
    assert np.isclose(np.linalg.norm(eri - ints['eri']), summary['pqrs_error'], rtol=1e-10,
                      atol=1e-14)
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        assert np.allclose(eri, eri.transpose(perm))
    assert np.allclose(h1e, h1e.T)


def _fci(h1e, eri, ints):
    fci = pytest.importorskip('pyscf.fci')
    nelec = (int(ints['nalpha']), int(ints['nbeta']))
    return fci.direct_spin1.FCI().kernel(h1e, eri, len(h1e), nelec)[0]


def test_reconstruct_has_the_fci_energy(case):
    """FCI of the reconstruction is ``e_upper``, the FCI energy of the true integrals."""
    name, f, ints, summary = case
    assert np.isclose(_fci(ints['h1e'], ints['eri'], ints), summary['e_upper'], rtol=0, atol=1e-10)
    rec = reconstruct(f['U'], f['W'], f['h_sym'], f['h1'])
    assert np.isclose(_fci(*rec, ints), summary['e_upper'], rtol=0, atol=FACTORS_ERROR[name])


# --- the walk's view: what it encodes, in the BLISS frame ----------------------------


def test_walk_inputs_match_the_factors(case):
    """``Lambda = sum(outer) / 2`` is the factors' ``lambda_sos``, the D1/Q1 split is its
    ``h1_shifted``, and the inner table keeps the signs of ``w_b`` and ``w_B``."""
    _, f, _, summary = case
    outer, inner = walk_inputs(f['W'], f['w_B'], f['w_plus'], f['w_minus'])
    assert np.isclose(outer.sum() / 2, summary['lambda_sos'], rtol=1e-12)
    split = (f['u_plus'].T @ np.diag(f['w_plus']) @ f['u_plus']
             - f['u_minus'].T @ np.diag(f['w_minus']) @ f['u_minus'])
    assert np.allclose(split, f['h1_shifted'], atol=1e-12)
    assert np.array_equal(inner[..., :-1], f['W'].transpose(0, 2, 1))
    assert np.array_equal(inner[..., -1], f['w_B'])


def test_walk_integrals_have_the_factors_gap(case):
    """On ``eta`` electrons ``H_SA = H_enc - E_SOS >= 0``, and its ground state is the factors'
    ``e_gap = e_upper - e_sos``: the gap is the same in both frames."""
    name, f, ints, summary = case
    gap = _fci(*walk_integrals(**{k: f[k] for k in KEYS}), ints) - walk_e_sos(
        f['W'], f['w_B'], f['w_minus'])
    assert gap > 0
    assert np.isclose(gap, summary['e_gap'], rtol=0, atol=FACTORS_ERROR[name])


def test_walk_integrals_are_the_sum_of_squares():
    """``walk_integrals`` is ``sum O^dag O + walk_e_sos``, with the generators written out from
    the README's Eqs. 28-29 (``a = (gamma_0 + i gamma_1)/2``), on random ``N = 3`` factors."""
    n, num_c = 3, 2
    f = random_factors(n, 2, 2, num_c, seed=0)
    g = majoranas(n)
    g_u = lambda u, s, x: np.tensordot(u, g[:, s, x], axes=1)  # the Majorana along u
    a_u = lambda u, s: (g_u(u, s, 0) + 1j * g_u(u, s, 1)) / 2
    eye = np.eye(4**n)
    ops = [np.sqrt(w) * a_u(u, s) for w, u in zip(f['w_plus'], f['u_plus']) for s in (0, 1)]
    ops += [np.sqrt(w) * a_u(u, s).conj().T for w, u in zip(f['w_minus'], f['u_minus'])
            for s in (0, 1)]
    for r, c in np.ndindex(f['w_B'].shape):
        pair = sum(f['W'][r, b, c] * 1j / 2 * g_u(f['U'][r, b], s, 0) @ g_u(f['U'][r, b], s, 1)
                   for b in range(f['W'].shape[1]) for s in (0, 1))
        ops.append((f['w_B'][r, c] * eye + pair) / np.sqrt(2))
    sos = sum(o.conj().T @ o for o in ops)
    e_sos = walk_e_sos(f['W'], f['w_B'], f['w_minus'])
    assert np.allclose(hamiltonian_matrix(*walk_integrals(**f)), sos + e_sos * eye, atol=1e-10)
