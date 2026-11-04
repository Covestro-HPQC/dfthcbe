r"""Rotation-angle data of the inner SELECT: the two QROM loads and their erase.

Steps 10, 11 and 18 of the ``README.md`` circuit. The angles are read into one shared
``(N-1) x b_rot`` register by two mutually exclusive loads -- the ``x_o`` tail for the D1/Q1
one-body terms, the ``(b, r)`` table for SF.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import Bloq, BQUInt, QAny, QBit, Register, Side, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.symbolics import bit_length, is_symbolic, Shaped

from dfthcbe.alias_data import PrunedQROM
from dfthcbe.utils import popcount

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt


# --- angle helpers (pure numpy) -----------------------------------------------------


def angles_from_vector(u_vec) -> np.ndarray:
    r"""``N-1`` Givens angles (radians) rotating ``\gamma_{0,x}`` onto ``\gamma_{u_vec,x}``.
    Solves the one-column Givens factorisation. Rung ``k`` acts on modes ``(k, k+1)``.
    """
    u = np.asarray(u_vec, dtype=float)
    if u.ndim != 1 or u.size < 2:
        raise ValueError(f"u_vec must be 1D of length N>=2; got shape {u.shape}")
    norm = np.linalg.norm(u)
    if abs(norm - 1) > 1e-8:
        raise ValueError(f"u_vec must have norm 1. Got {norm}")
    theta = [np.arctan2(np.linalg.norm(u[k + 1:]), u[k]) for k in range(u.size - 2)]
    theta.append(np.arctan2(u[-1], u[-2]))  # last rung signed, to fix the final component
    #TODO(#28): Check if theta needs a representing bit less if its modulo 2*pi 
    return np.array(theta) % (2 * np.pi)


def discretize_angles(theta_vec, b_rot: int) -> np.ndarray:
    r"""Round radians to the ``b_rot``-bit integers the QROM loads."""
    per = 2 ** (b_rot - 1)
    return np.round(np.asarray(theta_vec, float) / (2 * np.pi) * per).astype(int) % per


def vector_from_angles(theta_ints, b_rot: int) -> np.ndarray:
    r"""Inverse of :func:`angles_from_vector`.
    """
    theta = 4 * np.pi * np.asarray(theta_ints, dtype=float) / 2**b_rot
    u = np.zeros(theta.size + 1)
    u[0] = 1.0
    for k, phi in enumerate(theta):
        c, s = np.cos(phi), np.sin(phi)
        u[k], u[k + 1] = c * u[k] + s * u[k + 1], s * u[k] - c * u[k + 1]
    return u


# --- the two angle loads ------------------------------------------------------------


def sf_rotation_qrom(angle_data, b_rot: 'SymbolicInt') -> QROM:
    r"""SF angle load, keyed ``(b, r)`` and controlled on ``G_0 = 0`` (step 11).

    ``b`` spans ``B+1`` values. Nesting **``b`` outer / ``r`` inner** lets ``_break_early`` prune
    the all-zero ``b = B`` row's inner walk to its ``s(R)`` aligned blocks, so the cost drops from
    ``R(B+1)-1`` to ``RB + s(R) - 1``.

    Targets are ``target0_ .. target{N-2}_`` (the angles) then ``target{N-1}_`` (b!=B flag).
    """
    if isinstance(angle_data, Shaped):
        num_r, num_b, n_rungs = angle_data.shape
        qrom = PrunedQROM.build_from_bitsize(
            (num_b + 1, num_r), (n_rungs * b_rot, 1), num_controls=1
        )
        return attrs.evolve(qrom, num_leaves=num_r * num_b + popcount(num_r))
    num_r, num_b, n_rungs = np.shape(angle_data)
    table = np.zeros((num_b + 1, num_r, n_rungs), dtype=int)
    table[:num_b] = np.asarray(angle_data, dtype=int).transpose(1, 0, 2)
    flag = np.ones((num_b + 1, num_r), dtype=int)
    flag[num_b] = 0
    columns = [table[..., k] for k in range(n_rungs)] + [flag]
    return QROM.build_from_data(
        *columns, target_bitsizes=(b_rot,) * n_rungs + (1,), num_controls=1
    )


def tail_rotation_qrom(
    tail_data, num_sf: 'SymbolicInt', num_pos: 'SymbolicInt', b_rot: 'SymbolicInt'
) -> QROM:
    r"""D1/Q1 angle load: the last ``N`` of ``x_o`` are the one-body generators.
    The leading ``RC`` rows are all zero, and ``_break_early`` prunes them to their ``s(RC)``
    aligned blocks, so the walk costs ``N + s(RC) - 2``.

    It also writes the family selector: ``G_0 = 1`` on the whole tail, ``G_1 = 1``
    on the ``Q1`` rows.

    Targets are ``target0_ .. target{N-2}_`` (the angles), then ``G_0``, ``G_1``.

    Args:
        tail_data: ``(N, N-1)`` integer angles, the ``N_+`` ``D1`` rows first, or a
            symbolic ``Shaped``, costed by its pruned walk (:class:`~dfthcbe.alias_data.PrunedQROM`).
        num_sf: ``RC`` -- the all-zero SF prefix, so ``x_o`` spans ``N + RC`` values.
        num_pos: ``N_+`` -- how many tail rows are ``D1``; the rest are ``Q1`` (``G_1 = 1``).
        b_rot: bits of precision per angle.
    """
    if isinstance(tail_data, Shaped):
        num_orb, n_rungs = tail_data.shape
        qrom = PrunedQROM.build_from_bitsize((num_sf + num_orb,), (n_rungs * b_rot, 1, 1))
        return attrs.evolve(qrom, num_leaves=num_orb + popcount(num_sf))
    num_orb, n_rungs = np.shape(tail_data)
    n_rows = num_sf + num_orb
    table = np.zeros((n_rows, n_rungs), dtype=int)
    table[num_sf:] = np.asarray(tail_data, dtype=int)
    family = np.zeros((2, n_rows), dtype=int)
    family[0, num_sf:] = 1                      # G_0: SF vs the D1/Q1 pair
    family[1, num_sf + num_pos:] = 1            # G_1: D1 vs Q1 (the X -+ iY sign)
    columns = [table[:, k] for k in range(n_rungs)] + [family[0], family[1]]
    return QROM.build_from_data(*columns, target_bitsizes=(b_rot,) * n_rungs + (1, 1))


# --- the cheap erase of the angle registers -----------------------------------------


@frozen
class RotDataErase(Bloq):
    r"""Step 18: the measurement-based erase of both angle loads, a cost-only bloq.

    One X-measurement, outcome ``xi``, frees the ``N-1`` angle registers, ``b != B`` and ``G``,
    whichever load wrote them, for free. It leaves the sign ``(-1)^{xi . d(l)}`` of the erased
    data ``d`` on each address ``l``, and one phase fixup per load pays it back. The cost is
    hard-coded, so here is where every term comes from. A walk over ``l`` leaves with ``k``
    controls costs ``l + k - 2`` Toffolis (``l - 1`` splits, the root one free without a
    control), and ``s(n)`` is the number of ones of ``n`` in binary.

    - ``(b, r)``, the SF table: ``R + B - 3``. Berry et al.'s fixup: ``b`` to one-hot, then a
      walk over ``r`` whose leaf ``r`` puts row ``r``'s signs on the one-hot as ``Z``s.

      - One-hot of ``b``, ``(B+1) - 2``. It has ``B + 1`` positions: the identity row
        ``b = B`` loaded no angles, but it is a branch of the superposition and must get no
        sign, so it needs a position of its own. Each position after the first is reached by
        one controlled swap, and the first swap is two CNOTs, as it meets the known pair
        ``(1, 0)``.
      - Walk over ``r``, ``R - 2``.
      - Undoing the one-hot is free: each swap now ends on a qubit known to be 0, released by
        measurement.

      The paper charges ``R + B``.
    - ``x_o``, the D1/Q1 tail: ``N + s(RC) - 2``. No one-hot: a walk over ``x_o`` with a ``Z``
      on each tail row. It is the tail load's own walk (:func:`tail_rotation_qrom`): ``N`` live
      leaves, plus ``s(RC)`` to get past the all-zero SF rows, which prune to that many aligned
      blocks. The paper charges a bare ``N``, as it continues step 21's walk over ``x_o``
      instead of starting its own.

    Neither fixup is controlled: on the D1/Q1 branch ``b = r = 0``, so the ``(b, r)`` fixup leaves
    the classical sign ``(-1)^{g_SF(0, 0)}`` there, undone for free by a ``Z`` on ``G0`` before
    ``G`` is measured.

    Cost-only because the measurement has no unitary decomposition. With ``exact_erase``,
    :class:`~dfthcbe.select.InnerSelect` reads both tables again instead.

    Args:
        num_r: ``R`` -- iteration length of the ``r`` selection register.
        num_b: ``B`` -- number of loaded ``b`` values; the register itself spans ``B+1``.
        num_orb: ``N``; ``N-1`` angle registers to free, and the tail fixup's address count.
        num_sf: ``RC`` -- the SF prefix, so ``x_o`` spans ``N + RC`` values.
        b_rot: bits per angle register (sets their width; does *not* affect the cost).

    Registers:
        x_o: the outer index, addressing the D1/Q1 tail fixup (THRU).
        b: the inner selection register, ``B+1`` values (THRU).
        r: the rank selection register (THRU).
        angles: the ``N-1`` loaded angle registers, freed (``Side.LEFT``).
        b_neq_B: the identity flag written by the load, freed (``Side.LEFT``).
        G0, G1: the family selector written by the tail, freed (``Side.LEFT``).

    References:
        [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank
        Factorization](https://arxiv.org/abs/1902.02134). Berry et al. (2019), App. C.
    """

    num_r: 'SymbolicInt'
    num_b: 'SymbolicInt'
    num_orb: 'SymbolicInt'
    num_sf: 'SymbolicInt'
    b_rot: 'SymbolicInt'

    @cached_property
    def signature(self) -> Signature:
        n_xo = self.num_sf + self.num_orb
        return Signature(
            [
                Register('x_o', BQUInt(bit_length(n_xo - 1), n_xo)),
                Register('b', BQUInt(bit_length(self.num_b), self.num_b + 1)),
                Register('r', BQUInt(bit_length(self.num_r - 1), self.num_r)),
                Register('angles', QAny(self.b_rot), shape=(self.num_orb - 1,), side=Side.LEFT),
                Register('b_neq_B', QBit(), side=Side.LEFT),
                Register('G0', QBit(), side=Side.LEFT),
                Register('G1', QBit(), side=Side.LEFT),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        s_rc = popcount(self.num_sf) if is_symbolic(self.num_sf) else self.num_sf.bit_count()
        sf_fixup = (self.num_b + 1 - 2) + (self.num_r - 2)  # one-hot of b, walk over r
        tail_fixup = self.num_orb + s_rc - 2                # walk over x_o, past RC rows
        return {Toffoli(): sf_fixup + tail_fixup}
