r"""Helpers shared by several modules, and the integrals of a DFTHC fit."""

from __future__ import annotations

import functools
from typing import Tuple

import numpy as np
import sympy

from qualtran.symbolics import Shaped


# --- symbolic helpers ---------------------------------------------------------------


def num_blocks(a: int, b: int) -> int:
    """Aligned blocks ``[j 2^h, (j+1) 2^h)`` in the canonical decomposition of ``[a, b)``.

    Split at the midpoint ``m`` of the smallest aligned block containing ``[a, b)``: the part
    below climbs to ``m`` in ``s(m - a)`` blocks, the part above descends in ``s(b - m)``.
    """
    if b <= a:
        return 0
    h = (a ^ (b - 1)).bit_length()  # that smallest block has size 2^h
    start = a >> h << h
    if h == 0 or (a == start and b == start + (1 << h)):
        return 1
    m = start + (1 << (h - 1))
    return bin(m - a).count('1') + bin(b - m).count('1')


class popcount(sympy.Function):
    r"""``s(D)``, the number of ones in the binary expansion of ``D``; evaluates on integers."""

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_Integer and n >= 0:
            return sympy.Integer(bin(int(n)).count('1'))

    def _sympystr(self, printer):
        return f's({printer._print(self.args[0])})'

    def _latex(self, printer):
        return rf's\left({printer._print(self.args[0])}\right)'


class aligned_blocks(sympy.Function):
    r"""``beta(a, b)``, the aligned blocks of ``[a, b)`` (:func:`num_blocks`); evaluates on ints."""

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, a, b):
        if a.is_Integer and b.is_Integer:
            return sympy.Integer(num_blocks(int(a), int(b)))

    def _sympystr(self, printer):
        return f'beta({printer._print(self.args[0])}, {printer._print(self.args[1])})'

    def _latex(self, printer):
        a, b = (printer._print(x) for x in self.args)
        return rf'\beta\left({a}, {b}\right)'


# --- table helpers ------------------------------------------------------------------


def freeze_table(data, ndim: int, name: str):
    """Freeze an integer table into nested tuples, or pass a symbolic ``Shaped`` through."""
    if isinstance(data, Shaped):
        if len(data.shape) != ndim:
            raise ValueError(f"symbolic {name} must be {ndim}D; got {data.shape}")
        return data
    arr = np.asarray(data, dtype=int)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; got shape {arr.shape}")

    def freeze(x):
        return tuple(map(freeze, x)) if isinstance(x, list) else x

    return freeze(arr.tolist())


# --- DFTHC factors (pure numpy) -----------------------------------------------------
# Arguments are named as in ``factors.npz``.


def _check_unit_rows(name: str, vectors: np.ndarray) -> None:
    norms = np.linalg.norm(vectors, axis=-1)
    if norms.size and np.abs(norms - 1).max() > 1e-8:
        raise ValueError(f"the rows of {name} must have norm 1; got [{norms.min()}, {norms.max()}]")


def reconstruct(U, W, h_sym, h1) -> Tuple[np.ndarray, np.ndarray]:
    r"""The integrals ``(h1e, eri)`` of a DFTHC fit, chemists' order, as for an FCI solver.

    The fit ``h2' = sum_rc h_rc (x) h_rc``, ``h_rc = sum_b w_b u_b u_b^T`` (Eq. 24), is of the
    BLISS-shifted two-body tensor, so ``eri`` takes the shift ``(h_sym (x) I + I (x) h_sym) / 2``
    off again. ``h1e`` normal-orders ``sum h1 E + 1/2 sum eri E E``.

    Args:
        U: ``(R, B, N)`` unit vectors ``u_b^(r)``.
        W: ``(R, B, C)`` weights ``w_b^(rc)``.
        h_sym: ``(N, N)`` BLISS one-body shift.
        h1: ``(N, N)`` one-body term ``h^(1)`` of the ``E E`` form.
    """
    U, W, h_sym, h1 = (np.asarray(a, dtype=float) for a in (U, W, h_sym, h1))
    _check_unit_rows('U', U)
    eye = np.eye(len(h_sym))
    h_rc = np.einsum('rbp,rbq,rbc->rcpq', U, U, W)
    shift = (np.einsum('pq,rs->pqrs', h_sym, eye) + np.einsum('pq,rs->pqrs', eye, h_sym)) / 2
    eri = np.einsum('rcpq,rcst->pqst', h_rc, h_rc) - shift
    h1e = h1 + np.einsum('prrq->pq', eri) / 2
    return h1e, eri


def walk_inputs(W, w_B, w_plus, w_minus) -> Tuple[np.ndarray, np.ndarray]:
    r"""The LCU weights of a DFTHC fit, as the walk's two PREPAREs take them.

    Returns:
        outer: ``(N+RC,)``: ``lambda_SF,rc**2 = (sum_b |w_b| + |w_B|)**2 / 2`` at ``x_o = rC + c``
            (Eq. 32), then ``2 w_+`` (D1) and ``2 w_-`` (Q1), the factor 2 for the two spins.
        inner: ``(R, C, B+1)`` signed: ``W`` reordered, with ``w_B`` as ``b = B``.
    """
    W, w_B = np.asarray(W, dtype=float), np.asarray(w_B, dtype=float)
    inner = np.concatenate([W.transpose(0, 2, 1), w_B[..., None]], axis=-1)
    sf = np.abs(inner).sum(axis=-1).ravel() ** 2 / 2
    tail = 2 * np.concatenate([np.asarray(w_plus, float), np.asarray(w_minus, float)])
    return np.concatenate([sf, tail]), inner


def walk_e_sos(W, w_B, w_minus) -> float:
    r"""``E_SOS = -2 sum w_- - 1/2 sum_rc (w_B - sum_b w_b)**2`` (Eq. 30), in the walk's frame."""
    identity = np.asarray(w_B, dtype=float) - np.asarray(W, dtype=float).sum(axis=1)
    return float(-2 * np.sum(w_minus) - np.sum(identity**2) / 2)


def walk_integrals(U, W, w_B, w_plus, u_plus, w_minus, u_minus) -> Tuple[np.ndarray, np.ndarray]:
    r"""The integrals ``(h1e, eri)`` of what the walk encodes, ``sum O^dag O + E_SOS`` (Eq. 23).

    This is the BLISS-shifted Hamiltonian: on the fit's ``eta`` electrons it differs from
    :func:`reconstruct`'s by a constant. Its one-body part is ``h^(1)'`` from the D1/Q1 split,
    plus ``sum_rc W^(rc) h_rc`` with ``W^(rc) = w_B - sum_b w_b``.
    """
    U, W, w_B = (np.asarray(a, dtype=float) for a in (U, W, w_B))
    u_plus, u_minus = (np.asarray(u, dtype=float) for u in (u_plus, u_minus))
    _check_unit_rows('u_plus', u_plus)
    _check_unit_rows('u_minus', u_minus)
    h1_shifted = u_plus.T @ np.diag(w_plus) @ u_plus - u_minus.T @ np.diag(w_minus) @ u_minus
    h_rc = np.einsum('rbp,rbq,rbc->rcpq', U, U, W)
    h1 = h1_shifted + np.einsum('rc,rcpq->pq', w_B - W.sum(axis=1), h_rc)
    return reconstruct(U, W, np.zeros_like(h1), h1)


# --- the Hamiltonian on the system registers (pure numpy) ---------------------------


def majoranas(n: int) -> np.ndarray:
    r"""``gamma[p, sigma, x]`` on ``(psi_up, psi_dn)``, Jordan-Wigner per register (README):
    ``Z_0 .. Z_{p-1} (X|Y)_p``, qubit 0 the most significant, ``psi_up`` first."""
    X, Y = np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex)
    Z, eye2 = np.diag([1, -1]).astype(complex), np.eye(2, dtype=complex)
    one = [[functools.reduce(np.kron, [Z] * p + [P] + [eye2] * (n - 1 - p)) for P in (X, Y)]
           for p in range(n)]
    eye = np.eye(2**n)
    spin = (lambda m: np.kron(m, eye), lambda m: np.kron(eye, m))  # psi_up, psi_dn
    return np.array([[[spin[s](one[p][x]) for x in (0, 1)] for s in (0, 1)] for p in range(n)])


def hamiltonian_matrix(h1e, eri) -> np.ndarray:
    r"""The matrix of ``sum h1e E + 1/2 sum eri a^dag a^dag a a`` on ``(psi_up, psi_dn)``, the
    ``4**N`` Fock states of the walk's system registers, ``|1>`` occupied."""
    n = len(h1e)
    g = majoranas(n)
    a = (g[:, :, 0] + 1j * g[:, :, 1]) / 2
    E = np.array([[sum(a[p, s].conj().T @ a[q, s] for s in (0, 1)) for q in range(n)]
                  for p in range(n)])
    h1 = h1e - np.einsum('prrq->pq', eri) / 2  # the E E form
    return (np.einsum('pq,pqij->ij', h1, E)
            + np.einsum('pqrs,pqij,rsjk->ik', eri, E, E) / 2)
