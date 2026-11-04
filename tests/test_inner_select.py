r"""Tests for :class:`InnerSelect` -- steps 10-18 of the DFTHC block encoding.

**Correctness** is the composite operator, by tensor simulation: pin every register but the two
``N``-qubit systems and one address contracts to a ``4**N`` matrix, checked against the rotated
Majorana generator it is supposed to apply. That needs ``exact_erase`` -- the measurement-based
erase has no decomposition -- and it needs ``N`` and ``b_rot`` tiny. The exhaustive sweeps
are marked ``intensive`` and deselected by default (``pyproject.toml``); run them with
``-m intensive``.

Also **structure** (the decomposition contains exactly the circuit's blocks), **routing** (the
spin plumbing as an exact classical permutation, and -- with the ladders dropped, so 9 qubits --
as an exact operator), and **cost** (the call graph, numeric at FeMoco-54 and symbolic). The
parts are checked in ``test_rot_data`` / ``test_majorana`` / ``test_rotations``.
"""

import functools
import itertools
import os

import attrs
import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder, DecomposeTypeError, QAny, QBit
from qualtran.bloqs.basic_gates import CSwap, IntEffect, IntState, XGate
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import is_symbolic

from dfthcbe import InnerSelect, Maj, vector_from_angles
from dfthcbe.utils import popcount

B_ROT = 5
SF = np.array([[[3, 9], [12, 6]], [[1, 25], [7, 2]]])  # (R=2, B=2, N-1=2)
TAIL = np.array([[5, 2], [8, 11], [4, 30]])            # (N=3, N-1=2): 2 x D1 then 1 x Q1
NUM_C, NUM_POS = 2, 2                                  # C = 2 (so RC = 4), N_+ = 2

# FeMoco-54, the README's worked example (N, R, B, C, b_rot).
FEMOCO = (54, 10, 27, 27, 15)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def _ccz(bloq):
    return get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()['n_ccz']


@pytest.fixture
def select():
    return InnerSelect(
        angle_data=SF, tail_data=TAIL, num_c=NUM_C, num_pos=NUM_POS, b_rot=B_ROT
    )


def _femoco(**kwargs):
    n, r, b, c, b_rot = FEMOCO
    rng = np.random.default_rng(0)
    return InnerSelect(
        angle_data=rng.integers(1, 2**b_rot, size=(r, b, n - 1)),
        tail_data=rng.integers(1, 2**b_rot, size=(n, n - 1)),
        num_c=c, num_pos=n // 2, b_rot=b_rot, **kwargs,
    )


# --- structure ----------------------------------------------------------------------


def test_signature(select):
    """The two success flags, the three index registers, the spins, and the system.

    The angle registers, ``b_neq_B`` and ``G`` are *not* here: steps 10-11 write them, ``Maj``
    reads them and step 18 erases them, so they are internal. ``r`` is, because inner PREPARE
    writes it and erases it at step 21.
    """
    assert [r.name for r in select.signature] == [
        'succ_x_o', 'succ_b', 'x_o', 'b', 'r', 'sigma_0', 'sigma_1',
        'psi_up', 'psi_dn', 'phase_grad',
    ]
    assert (select.num_r, select.num_b, select.num_orb, select.num_sf) == (2, 2, 3, 4)
    spans = {r.name: r.dtype.iteration_length for r in select.signature
             if hasattr(r.dtype, 'iteration_length')}
    assert spans['x_o'] == select.num_sf + select.num_orb  # N + RC
    assert spans['b'] == select.num_b + 1                  # b = B is the identity term
    assert spans['r'] == select.num_r


def test_decomposes_into_the_circuit_blocks(select):
    """Exactly the blocks of steps 10-18."""
    from qualtran.bloqs.bookkeeping import Allocate

    bloqs = [bi.bloq for bi in select.decompose_bloq().bloq_instances]
    assert bloqs.count(select.tail_qrom) == 1     # (10) D1/Q1 angles + G
    assert bloqs.count(select.sf_qrom) == 1       # (11) SF angles + b != B
    assert bloqs.count(XGate()) == 2              #      fired on G0 = 0
    assert bloqs.count(MultiControlX(cvs=(0, 1))) == 2  # (12, 17) sg = sigma_{G0} and back
    assert bloqs.count(MultiControlX(cvs=(1, 1))) == 2
    assert bloqs.count(select.uswap) == 2         # (13, 17) USwap, controlled on sg
    assert bloqs.count(select.ladder) == 1          # (14)
    assert bloqs.count(select.maj) == 1           # (15)
    assert bloqs.count(select.ladder.adjoint()) == 1  # (16)
    assert bloqs.count(select.eraser) == 1        # (18)
    # N-1 angle registers, the flag, G0, G1 and sg -- the two spin wires are THRU.
    assert sum(isinstance(b, Allocate) for b in bloqs) == select.num_orb + 3


@pytest.mark.parametrize("G0,s0,s1", [(g, a, b) for g in (0, 1) for a in (0, 1) for b in (0, 1)])
def test_spin_routing_is_a_classical_permutation(G0, s0, s1):
    """(12, 13) ``sg = sigma_{G0}`` drives the ``N``-qubit swap, and both spins survive.

    SF sums the spin on ``sigma_0``, D1/Q1 on ``sigma_1``; multiplexing into
    ``sg`` means the swap needs **one** control rather than an if-else, and leaves ``sigma_0``
    for ``Maj`` to use as its flavour ancilla. A permutation, so ``call_classically`` pins the
    convention exactly, without touching the ``b_rot``-heavy ladder.
    """
    n = 2
    bb = BloqBuilder()
    g = bb.add_register('G0', 1)
    a = bb.add_register('sigma_0', 1)
    b = bb.add_register('sigma_1', 1)
    sg = bb.add_register('sg', 1)
    pu = bb.add_register('psi_up', n)
    pd = bb.add_register('psi_dn', n)
    (g, a), sg = bb.add(MultiControlX(cvs=(0, 1)), controls=np.array([g, a]), target=sg)
    (g, b), sg = bb.add(MultiControlX(cvs=(1, 1)), controls=np.array([g, b]), target=sg)
    sg, pd, pu = bb.add(CSwap(bitsize=n), ctrl=sg, x=pd, y=pu)
    names = ['G0', 'sigma_0', 'sigma_1', 'sg', 'psi_up', 'psi_dn']
    cb = bb.finalize(G0=g, sigma_0=a, sigma_1=b, sg=sg, psi_up=pu, psi_dn=pd)

    out = dict(zip(names, cb.call_classically(
        G0=G0, sigma_0=s0, sigma_1=s1, sg=0, psi_up=1, psi_dn=2
    )))
    active = s1 if G0 else s0
    assert out['sg'] == active
    assert (out['sigma_0'], out['sigma_1']) == (s0, s1)
    assert (out['psi_up'], out['psi_dn']) == ((2, 1) if active else (1, 2))


def _spin_plumbing(n):
    """Steps 12-17 by hand: build ``sg`` -> USwap -> ``Maj`` -> USwap -> unbuild ``sg``.

    No Givens ladder -- it is a unitary conjugation of ``psi_dn`` alone, tested in
    ``test_rotations`` -- so this is ``3 + 2 + 2n`` qubits, cheap to contract exactly. Both success
    flags are pinned to 1.
    """
    bb = BloqBuilder()
    regs = {name: bb.add_register(name, 1)
            for name in ('G0', 'G1', 'b_neq_B', 'sigma_0', 'sigma_1')}
    G0, G1, nB, s0, s1 = regs.values()
    up, dn = bb.add_register('psi_up', n), bb.add_register('psi_dn', n)

    sg = bb.allocate(1, QBit())
    (G0, s0), sg = bb.add(MultiControlX(cvs=(0, 1)), controls=np.array([G0, s0]), target=sg)
    (G0, s1), sg = bb.add(MultiControlX(cvs=(1, 1)), controls=np.array([G0, s1]), target=sg)
    sg, dn, up = bb.add(CSwap(bitsize=n), ctrl=sg, x=dn, y=up)
    qs = list(bb.split(dn))
    succ_x_o, succ_b, G0, G1, nB, s0, qs[0] = bb.add_t(
        Maj(), succ_x_o=bb.add(IntState(1, 1)), succ_b=bb.add(IntState(1, 1)), G0=G0, G1=G1,
        b_neq_B=nB, flavour=s0, target=qs[0],
    )
    for flag in (succ_x_o, succ_b):
        bb.add(IntEffect(1, 1), val=flag)
    dn = bb.join(np.array(qs), dtype=QAny(n))
    sg, dn, up = bb.add(CSwap(bitsize=n), ctrl=sg, x=dn, y=up)
    (G0, s0), sg = bb.add(MultiControlX(cvs=(0, 1)), controls=np.array([G0, s0]), target=sg)
    (G0, s1), sg = bb.add(MultiControlX(cvs=(1, 1)), controls=np.array([G0, s1]), target=sg)
    bb.free(sg)  # contracts as <0|, so a dirty sg would show up as a non-unitary result
    return bb.finalize(
        G0=G0, G1=G1, b_neq_B=nB, sigma_0=s0, sigma_1=s1, psi_up=up, psi_dn=dn
    )


def test_sigma_0_can_be_both_the_spin_and_maj_s_flavour_ancilla():
    """``sigma_0`` does double duty across the ``G0`` branches without cross-contamination.

    SF sums the spin on ``sigma_0``; D1/Q1 sums it outside (on ``sigma_1``) and uses
    ``sigma_0`` as ``Maj``'s flavour ancilla instead. Both jobs are the *same* operation -- a
    uniform 2-term LCU discharged by projecting the ancilla onto ``|+>`` -- which is why the
    single ``REF_T2`` at step 23 is correct on both branches even with ``G0`` in superposition.
    Here: contract steps 12-17, project ``sigma_0`` onto ``|+>``, and read the
    operator off; ``sigma_1`` stays a free index, ``REF_aB`` projecting it at step 28.
    """
    n = 2
    eye = lambda k: np.eye(2**k, dtype=complex)
    # `op` on qubit 0 of psi_up / psi_dn, over the 2n system qubits.
    up_0 = lambda op: np.kron(np.kron(op, eye(n - 1)), eye(n))
    dn_0 = lambda op: np.kron(eye(n), np.kron(op, eye(n - 1)))

    U = _spin_plumbing(n).tensor_contract()
    assert np.allclose(U.conj().T @ U, eye(5 + 2 * n))  # unitary => sg came back to |0>

    def plus(G0, G1, nB, s1_out, s1_in):
        """``<+|U|+>`` on ``sigma_0``, at fixed controls and ``sigma_1``."""
        blocks = []
        for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            lo = ((((G0 << 1 | G1) << 1 | nB) << 1 | i) << 1 | s1_out) << 2 * n
            hi = ((((G0 << 1 | G1) << 1 | nB) << 1 | j) << 1 | s1_in) << 2 * n
            blocks.append(U[lo:lo + 4**n, hi:hi + 4**n])
        return sum(blocks) / 2

    # SF: sigma_0 is the spin, so |+> gives the spin *sum* -- and sigma_1 is untouched.
    for nB, expect in [(1, -(up_0(Z) + dn_0(Z)) / 2), (0, eye(2 * n))]:
        for s1 in (0, 1):
            assert np.allclose(plus(0, 0, nB, s1, s1), expect), (nB, s1)
        assert np.allclose(plus(0, 0, nB, 0, 1), 0) and np.allclose(plus(0, 0, nB, 1, 0), 0)
        assert np.allclose(plus(0, 0, nB, 0, 0), plus(0, 0, nB, 1, 1))

    # D1/Q1: sigma_0 is the flavour, sigma_1 the spin (which sector the Majorana lands on).
    for G1, sign in [(0, +1), (1, -1)]:
        for s1, sector in [(0, dn_0), (1, up_0)]:
            expect = sector((X + sign * 1j * Y) / 2)
            for nB in (0, 1):                      # D1/Q1 ignores b != B
                assert np.allclose(plus(1, G1, nB, s1, s1), expect), (G1, s1, nB)
        assert np.allclose(plus(1, G1, 1, 0, 1), 0) and np.allclose(plus(1, G1, 1, 1, 0), 0)


# --- correctness: the composite operator (tensor sim, pinned addresses) --------------
#
# Steps 10-18 are
#
#   sum_{x_o,b,r} |x_o b r><x_o b r| (x) sum_{s0,s1} |s0 s1><s0 s1| (x) U(u)^dag M U(u)
#
# the ladder on ``psi_{sigma_{g_0}}`` and ``M`` on its mode 0. Pinning everything but the two
# systems leaves a ``4**N`` matrix per address.

B_ROT_SIM = 10


def _sim(num_orb, num_pos, seed=7):
    """A contractible :class:`InnerSelect`, built the production way: from *directions*, which
    ``angles_from_vector`` + ``discretize_angles`` round to ``B_ROT_SIM`` bits before the circuit
    ever sees them. Returns it with the unit vectors it was asked for -- the ground truth.

    ``exact_erase`` because the measurement-based erase has no decomposition.
    """
    rng = np.random.default_rng(seed)
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    u_sf = unit(rng.standard_normal((2, 1, num_orb)))         # (R=2, B=1, N)
    u_tail = unit(rng.standard_normal((num_orb, num_orb)))    # N rows, the D1 ones first
    sel = InnerSelect.from_vectors(
        u_sf, u_tail[:num_pos], u_tail[num_pos:], num_c=1, b_rot=B_ROT_SIM, exact_erase=True
    )
    return sel, u_sf, u_tail


def _atol(sel):
    """The rounding floor: ``N-1`` angles, each off by up to ``2 pi / 2**b_rot``."""
    return 8 * sel.num_orb * 2.0 ** -(sel.b_rot - 1)


def _branch_op(sel, x_o, b, r, succ=(1, 1), s0=0, s1=0):
    """The ``2N``-qubit operator on ``(psi_up, psi_dn)`` at one fixed address.

    ``IntState``/``IntEffect`` pin the addresses, flags and spins and ``PhaseGradientState``
    supplies the catalyst, so the dense matrix is ``4**N`` wide -- 16 at ``N = 2``, against
    ``2**22`` for the whole signature.
    """
    width = {reg.name: reg.dtype.num_qubits for reg in sel.signature}
    pinned = dict(x_o=x_o, b=b, r=r, succ_x_o=succ[0], succ_b=succ[1], sigma_0=s0, sigma_1=s1)
    bb = BloqBuilder()
    psi_up = bb.add_register('psi_up', QAny(sel.num_orb))
    psi_dn = bb.add_register('psi_dn', QAny(sel.num_orb))
    ins = {name: bb.add(IntState(val=val, bitsize=width[name])) for name, val in pinned.items()}
    out = bb.add_d(sel, psi_up=psi_up, psi_dn=psi_dn, **ins,
                   phase_grad=bb.add(PhaseGradientState(bitsize=sel.b_rot)))
    bb.add(PhaseGradientState(bitsize=sel.b_rot).adjoint(), phase_grad=out['phase_grad'])
    for name, val in pinned.items():
        bb.add(IntEffect(val=val, bitsize=width[name]), val=out[name])
    return bb.finalize(psi_up=out['psi_up'], psi_dn=out['psi_dn']).tensor_contract()


def _majorana(j, n, flavour):
    """``gamma_{j,f} = Z_0 .. Z_{j-1} (X|Y)_j`` (Jordan-Wigner, qubit 0 = MSB).

    The flavour is spelled x/y, never 0/1, so it cannot be misread as the registers
    ``sigma_0``/``sigma_1``.
    """
    return functools.reduce(
        np.kron, [Z] * j + [X if flavour == 'x' else Y] + [I2] * (n - 1 - j)
    )


def _generator(u, kind, on_up):
    """The operator steps 14-16 apply for direction ``u``, on one spin register.

    Per flavour ``U^dag gamma_{0,f} U = sum_j u_j gamma_{j,f}``, the Majorana along ``u``. On one
    mode ``gamma_x gamma_y = i Z``, so the SF branch is ``-U^dag Z_0 U = i gamma_x gamma_y``,
    the README's sign. For D1/Q1 the ``|+>`` on ``sigma_0`` sums the two flavours, leaving
    ``a_u`` (``D1``) or ``a_u^dag`` (``Q1``).
    """
    n = len(u)
    gamma_x = sum(u[j] * _majorana(j, n, 'x') for j in range(n))
    gamma_y = sum(u[j] * _majorana(j, n, 'y') for j in range(n))
    if kind == 'SF':
        op = 1j * gamma_x @ gamma_y
    else:
        op = (gamma_x + (1j if kind == 'D1' else -1j) * gamma_y) / 2
    # psi_up is the first register of the contraction, so it is the most significant.
    return np.kron(op, np.eye(2**n)) if on_up else np.kron(np.eye(2**n), op)


def test_sf_branch_applies_the_rotated_majorana_pair():
    """(10-18) SF: ``-U(u)^dag Z_0 U(u)``, the Majorana pair along the direction we asked for.

    ``sigma_0`` is the spin here, so it also decides *which* system register is hit -- the swap
    control is ``sg = sigma_{G_0}`` and ``G_0 = 0`` on SF.
    """
    sel, u_sf, _ = _sim(num_orb=2, num_pos=1)
    for s0 in (0, 1):
        got = _branch_op(sel, x_o=0, b=0, r=0, s0=s0)
        assert np.allclose(got, _generator(u_sf[0, 0], 'SF', s0), atol=_atol(sel)), s0


def test_the_identity_row_does_nothing():
    """``b = B`` is the inner LCU's identity term: no angles loaded, no Majorana."""
    sel, _, _ = _sim(num_orb=2, num_pos=1)
    got = _branch_op(sel, x_o=0, b=sel.num_b, r=0)
    assert np.allclose(got, np.eye(4**sel.num_orb))


def test_tail_branch_applies_the_one_body_generator():
    """(10-18) D1/Q1: with ``sigma_0`` projected onto ``|+>`` -- its job on this branch is
    ``Maj``'s flavour ancilla -- the tail applies ``a_u`` / ``a_u^dag`` on ``psi_{sigma_1}``.
    """
    sel, _, u_tail = _sim(num_orb=2, num_pos=1)
    for j, kind in enumerate(['D1', 'Q1']):
        plus = sum(_branch_op(sel, sel.num_sf + j, 0, 0, s0=s0) for s0 in (0, 1)) / 2
        assert np.allclose(plus, _generator(u_tail[j], kind, False), atol=_atol(sel)), kind


@pytest.mark.parametrize("succ", [(0, 1), (1, 0), (0, 0)])
def test_success_flags_gate_the_majorana(succ):
    """(15) On a D1/Q1 row ``Maj`` fires on ``succ_x_o`` alone, ``succ_b`` is ignored. Off, the two
    ladders cancel and nothing else acts: the identity."""
    sel, _, _ = _sim(num_orb=2, num_pos=1)
    got = _branch_op(sel, x_o=sel.num_sf, b=0, r=0, succ=succ)
    want = _branch_op(sel, x_o=sel.num_sf, b=0, r=0) if succ[0] else np.eye(4**sel.num_orb)
    assert np.allclose(got, want)


def test_branch_operator_is_unitary():
    """``exact_erase`` is exact: the angles, ``b != B``, ``G`` and ``sg`` all come home to ``|0>``.

    ``bb.free`` contracts as ``<0|``, so any wire left dirty shows up as a non-unitary result.
    """
    sel, _, _ = _sim(num_orb=2, num_pos=1)
    got = _branch_op(sel, x_o=sel.num_sf, b=0, r=0, s0=1, s1=1)
    assert np.allclose(got.conj().T @ got, np.eye(4**sel.num_orb))


@pytest.mark.intensive
@pytest.mark.parametrize("num_orb,num_pos", [(2, 1), (3, 2)])
def test_sf_branch_sweep(num_orb, num_pos):
    """Every SF address and both spins. ``sigma_1`` is swept too: it must not matter here.

    ``N = 3`` is why this tier exists -- two Givens rungs, so it pins the rung *order*, which
    ``N = 2`` (one rung) cannot.
    """
    sel, u_sf, _ = _sim(num_orb, num_pos)
    for x_o, b, r, s0, s1 in itertools.product(
        range(sel.num_sf), range(sel.num_b + 1), range(sel.num_r), (0, 1), (0, 1)
    ):
        want = (np.eye(4**num_orb) if b == sel.num_b     # b = B is the identity term
                else _generator(u_sf[r, b], 'SF', s0))
        got = _branch_op(sel, x_o, b, r, s0=s0, s1=s1)
        assert np.allclose(got, want, atol=_atol(sel)), (x_o, b, r, s0, s1)


@pytest.mark.intensive
@pytest.mark.parametrize("num_orb,num_pos", [(2, 1), (3, 2)])
def test_tail_branch_sweep(num_orb, num_pos):
    """Every D1/Q1 row and both spins; ``b`` and ``r`` are dead on this branch."""
    sel, _, u_tail = _sim(num_orb, num_pos)
    for j, s1 in itertools.product(range(num_orb), (0, 1)):
        kind = 'D1' if j < num_pos else 'Q1'
        plus = sum(_branch_op(sel, sel.num_sf + j, 0, 0, s0=s0, s1=s1) for s0 in (0, 1)) / 2
        assert np.allclose(plus, _generator(u_tail[j], kind, s1), atol=_atol(sel)), (j, kind, s1)


# --- cost ---------------------------------------------------------------------------


# TODO(#17): never decomposes; compare against select.decompose_bloq() as test_prepare does.
def test_cost_is_the_block_sum(select):
    """No cost/decomposition divergence: the call graph is exactly its blocks."""
    blocks = (
        _ccz(select.tail_qrom) + _ccz(select.sf_qrom) + 2 * _ccz(MultiControlX(cvs=(0, 1)))
        + 2 * _ccz(MultiControlX(cvs=(1, 1))) + 2 * _ccz(select.uswap)
        + 2 * _ccz(select.ladder) + _ccz(select.maj) + _ccz(select.eraser)
    )
    assert _ccz(select) == blocks


def test_femoco_cost_by_block():
    """Block by block at FeMoco-54: 3502 Toffolis.

    The Givens atom costs ``2*b_rot - 2`` per rung: a ``b``-bit addition into the phase gradient
    needs ``b-1`` Toffolis because the top carry is never used. The two pruned loads walk
    ``s(RC) - 2`` and ``s(R) - 1`` leaves beyond their ``N`` and ``RB`` live ones, and the
    erase's ``x_o`` fixup walks the tail load's tree again.
    """
    n, r, b, _c, b_rot = FEMOCO
    s = _femoco()
    mcx = _ccz(MultiControlX(cvs=(0, 1))) + _ccz(MultiControlX(cvs=(1, 1)))
    assert _ccz(s.tail_qrom) == 56                     # N + s(RC) - 2
    assert _ccz(s.sf_qrom) == 271                      # RB + s(R) - 1
    assert 2 * mcx == 4                                # sg built and unbuilt
    assert 2 * _ccz(s.uswap) == 2 * n == 108           # 2N
    assert _ccz(s.maj) == 5                            # Maj and its gating
    assert _ccz(s.eraser) == (r + b - 3) + _ccz(s.tail_qrom) == 90  # (b, r), then x_o
    assert 2 * _ccz(s.ladder) == 2 * (n - 1) * (2 * b_rot - 2) == 2968
    assert _ccz(s) == 3502 == 56 + 271 + 4 + 108 + 5 + 90 + 2968


def test_symbolic_cost_is_the_closed_form():
    """Symbolically the total is ``4N + RB + R + B + 1 + 2 s(RC) + s(R) + 2(N-1)(2 b_rot - 2)``.

    Every term is exact, the two loads included: the dead SF prefix of the tail costs ``s(RC)``
    leaves, once in the load and once in the erase, and the zero row of the SF table ``s(R)``
    (``popcount``), so substituting FeMoco-54 gives the built circuit's count.
    """
    n, r, b, c, b_rot = sympy.symbols('N R B C b_rot', positive=True, integer=True)
    s = InnerSelect.symbolic(num_orb=n, num_r=r, num_b=b, num_c=c, b_rot=b_rot)
    assert is_symbolic(s.angle_data)
    assert sympy.simplify(_ccz(s.tail_qrom) - (n + popcount(r * c) - 2)) == 0
    assert sympy.simplify(_ccz(s.sf_qrom) - (r * b + popcount(r) - 1)) == 0
    assert sympy.simplify(_ccz(s.uswap) - n) == 0                   # N per USwap
    assert sympy.simplify(_ccz(s.ladder) - (n - 1) * (2 * b_rot - 2)) == 0
    assert sympy.simplify(_ccz(s.eraser) - (r + b - 3 + n + popcount(r * c) - 2)) == 0
    assert _ccz(s.maj) == 5

    total = (4 * n + r * b + r + b + 1 + 2 * popcount(r * c) + popcount(r)
             + 2 * (n - 1) * (2 * b_rot - 2))
    assert sympy.simplify(_ccz(s) - total) == 0
    assert _ccz(_femoco()) == total.subs(dict(zip((n, r, b, c, b_rot), FEMOCO)))


def test_symbolic_cannot_decompose():
    n, r, b, c, b_rot = sympy.symbols('N R B C b_rot', positive=True, integer=True)
    with pytest.raises(DecomposeTypeError):
        InnerSelect.symbolic(num_orb=n, num_r=r, num_b=b, num_c=c, b_rot=b_rot).decompose_bloq()


def test_exact_erase_pays_a_second_read(select):
    """``exact_erase`` swaps the measurement-based erase for the honest unitary inverse.

    That keeps the bloq simulable -- everything in it decomposes to gates -- at the price of
    re-reading both tables, dead rows included: step 18 then costs
    ``RB + s(R) - 1 + N + s(RC) - 2`` (327 at FeMoco-54) instead of the phase fixups.
    """
    exact = InnerSelect(
        angle_data=SF, tail_data=TAIL, num_c=NUM_C, num_pos=NUM_POS, b_rot=B_ROT,
        exact_erase=True,
    )
    exact.decompose_bloq()
    expected = _ccz(select) - _ccz(select.eraser) + _ccz(select.sf_qrom) + _ccz(select.tail_qrom)
    assert _ccz(exact) == expected

    n, r, b, c, b_rot = sympy.symbols('N R B C b_rot', positive=True, integer=True)
    sym = InnerSelect.symbolic(num_orb=n, num_r=r, num_b=b, num_c=c, b_rot=b_rot)
    step_21 = _ccz(attrs.evolve(sym, exact_erase=True)) - _ccz(sym) + _ccz(sym.eraser)
    assert sympy.simplify(step_21 - (r * b + popcount(r) - 1 + n + popcount(r * c) - 2)) == 0
    femoco = _femoco()
    assert _ccz(_femoco(exact_erase=True)) - _ccz(femoco) + _ccz(femoco.eraser) == 271 + 56


@pytest.mark.parametrize("exact_erase", [False, True])
def test_adjoint_is_the_same_circuit_with_maj_daggered(select, exact_erase):
    """``SEL = V^dag Maj V``, so ``SEL^dag = V^dag Maj^dag V``: the loads still come first and the
    erase last. Reversing the circuit instead would run the measurement-based erase backwards."""
    sel = attrs.evolve(select, exact_erase=exact_erase)
    boxes = lambda b: [i.bloq for i in sorted(b.decompose_bloq().bloq_instances, key=lambda i: i.i)]
    assert boxes(sel.adjoint()) == [sel.maj.adjoint() if b == sel.maj else b for b in boxes(sel)]
    assert sel.adjoint().adjoint() == sel
    assert _ccz(sel.adjoint()) == _ccz(sel)


# --- constructors / validation ------------------------------------------------------


def test_from_vectors_reproduces_the_directions():
    """``from_vectors`` normalises and discretises; each ladder redirects ``e_0`` onto ``u``."""
    b_rot = 16
    u_vecs = np.array([                                        # (R=2, B=2, N=4)
        [[1.0, 2.0, -0.5, 0.3], [0.0, 1.0, 1.0, 0.0]],
        [[-1.0, 0.2, 0.0, 3.0], [2.0, -1.0, 0.5, 0.1]],
    ])
    u_plus = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]])       # 2 x D1
    u_minus = np.array([[0.1, 0.2, 1.0, 0.3], [-1.0, 0.4, 0.2, 2.0]])     # 2 x Q1
    sel = InnerSelect.from_vectors(u_vecs, u_plus, u_minus, num_c=1, b_rot=b_rot)
    assert (sel.num_r, sel.num_b, sel.num_orb, sel.num_pos) == (2, 2, 4, 2)
    tol = 4 * 2 ** -(b_rot - 1)
    for i in range(2):
        for j in range(2):
            u = u_vecs[i, j]
            v = vector_from_angles(sel.angle_array[i, j], b_rot)
            assert np.allclose(v, u / np.linalg.norm(u), atol=tol)
    for j, u in enumerate(np.vstack([u_plus, u_minus])):
        v = vector_from_angles(sel.tail_array[j], b_rot)
        assert np.allclose(v, u / np.linalg.norm(u), atol=tol)


def test_from_vectors_matches_fe2s2_factors():
    """Real DFTHC factors round-trip: one SF direction and one ``D1`` eigenvector."""
    path = "data/dfthc/fe2s2/factors.npz"
    if not os.path.exists(path):
        pytest.skip("fe2s2 factors.npz not available")
    d = np.load(path)
    b_rot = 20
    sel = InnerSelect.from_vectors(d["U"], d["u_plus"], d["u_minus"], num_c=1, b_rot=b_rot)
    n = d["U"].shape[2]
    assert (sel.num_orb, sel.num_pos) == (n, len(d["u_plus"]))
    v = vector_from_angles(sel.angle_array[0, 0], b_rot)
    assert np.allclose(v, d["U"][0, 0] / np.linalg.norm(d["U"][0, 0]), atol=1e-4)
    v = vector_from_angles(sel.tail_array[0], b_rot)
    assert np.allclose(v, d["u_plus"][0], atol=1e-4)


def test_from_angles_roundtrip():
    angles = np.array([[[0.0, np.pi]], [[np.pi / 2, 2 * np.pi * 0.1]]])  # (R=2, B=1, N-1=2)
    tail = np.array([[0.0, 0.3], [1.0, np.pi], [0.5, 2.0]])              # (N=3, N-1=2)
    sel = InnerSelect.from_angles(angles, tail, num_c=3, num_pos=1, b_rot=8)
    per = 2 ** (8 - 1)
    assert np.array_equal(sel.angle_array, np.round(angles / (2 * np.pi) * per).astype(int) % per)
    assert np.array_equal(sel.tail_array, np.round(tail / (2 * np.pi) * per).astype(int) % per)
    assert (sel.num_r, sel.num_b, sel.num_orb, sel.num_sf) == (2, 1, 3, 6)


@pytest.mark.parametrize("kwargs", [
    dict(angle_data=np.array([[[0, 2**B_ROT], [1, 2]]])),      # angle wider than b_rot
    dict(angle_data=np.array([[[1, 2]]])),                     # R = B = 1: nothing to select
    dict(angle_data=np.array([[1, 2], [3, 4]])),               # SF table not 3D
    dict(tail_data=np.array([[1, 2], [3, 4]])),                # tail not (N, N-1)
    dict(num_pos=99),                                          # more D1 rows than there are
])
def test_rejects_bad_tables(kwargs):
    args = dict(angle_data=SF, tail_data=TAIL, num_c=NUM_C, num_pos=NUM_POS, b_rot=B_ROT)
    with pytest.raises(ValueError):
        InnerSelect(**(args | kwargs))
