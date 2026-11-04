r"""Tests for :class:`DFTHCBlockEncoding` (steps 1-27) and :class:`DFTHCWalk` (all 28 steps).

**Correctness** by exact contraction at ``N = 2``: with the ancillas pinned to ``|0>`` on both
sides, ``<0|BE|0>`` is ``sum_x p_x (2 A_x^dag A_x - I)``, ``A_x`` being the inner LCU of the
generator ``x_o`` selects. The reference is built from the alias tables and angles the bloqs
actually load, so it is exact and ``b_rot`` can be tiny. It needs ``exact_erase`` on all three
sub-bloqs, as the measurement-based erases have no decomposition. Built from a decomposed
Hamiltonian, the block is ``H_SA / Lambda - I`` for the ``H_SA`` of the decomposed Hamiltonian it
encodes (``encoded_factors``): the default tier checks it on H2 in STO-3G, where ``Uniform(b)``
fails. The ``intensive`` tier adds the reference above, a random decomposed Hamiltonian where
``Uniform(x_o)`` fails and one where neither does, and checks that the controlled block encoding is
the identity on ``ctrl = 0`` and the block on ``ctrl = 1``. ``REF_T2`` is also contracted on its
own, the reflection about ``|0>_B |1>_succ_x_o``. The encoded factors' error is bounded by that of
the input factors rounded to the same bits. **Cost** by the call graph, at FeMoco-54. The
reflections are pinned in ``test_reflections``.
"""

import itertools

import numpy as np
import pytest
import sympy

from qualtran import Adjoint, BloqBuilder, DecomposeTypeError, QAny
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.resource_counting import get_cost_value, QECGatesCost

from dfthcbe import (
    alias_probabilities, DFTHCBlockEncoding, DFTHCWalk, InnerPrepare, InnerSelect, OuterPrepare,
    hamiltonian_matrix, RotDataErase, vector_from_angles, walk_e_sos, walk_inputs, walk_integrals,
)
from dfthcbe.block_encoding import INNER_ANCILLAS, OUTER_ANCILLAS
from test_inner_select import _generator
from test_utils import FACTORS_ERROR, KEYS, load_factors, random_factors

# FeMoco-54, the README's worked example.
FEMOCO = dict(num_orb=54, num_r=10, num_b=27, num_c=27, b_rot=15, b_k=15, k_1=2, k_2=4)


def _toffoli(bloq):
    """Toffoli count: ``CCZ``, ``CSwap`` and every 4 raw ``T`` (as in ``test_prepare``)."""
    c = get_cost_value(bloq, QECGatesCost())
    return c.toffoli + c.cswap + c.and_bloq + c.t / 4


def _walk(num_orb, num_r, num_b, num_c, b_rot, b_k, k_1=None, k_2=None, seed=0):
    """Random tables at the given sizes."""
    rng = np.random.default_rng(seed)
    uniform = dict(s=min(7, b_rot), phase_grad_bitsize=b_rot)
    return DFTHCWalk(DFTHCBlockEncoding(
        outer=OuterPrepare.from_weights(
            rng.random(num_orb + num_r * num_c) + 0.1, b_k1=b_k, log_block_size=k_1, **uniform
        ),
        inner=InnerPrepare.from_coeffs(
            rng.standard_normal((num_r, num_c, num_b + 1)), num_orb=num_orb, b_k2=b_k,
            log_block_size=k_2, **uniform,
        ),
        select=InnerSelect(
            angle_data=rng.integers(1, 2**b_rot, (num_r, num_b, num_orb - 1)),
            tail_data=rng.integers(1, 2**b_rot, (num_orb, num_orb - 1)),
            num_c=num_c, num_pos=num_orb // 2, b_rot=b_rot,
        ),
    ))


@pytest.fixture
def walk():
    return _walk(num_orb=3, num_r=2, num_b=2, num_c=2, b_rot=5, b_k=3)


def test_signature(walk):
    """The ancillas ``a u B`` that ``REF_aB`` reflects, the system and the catalyst, and ``ctrl``
    first when controlled. Every flag and loaded table is computed and uncomputed inside."""
    names = [*OUTER_ANCILLAS, *INNER_ANCILLAS, 'psi_up', 'psi_dn', 'phase_grad']
    for bloq in (walk, walk.block_encoding):
        assert [r.name for r in bloq.signature] == names
        assert [r.name for r in bloq.controlled().signature] == ['ctrl', *names]


def test_no_erase_runs_backwards(walk):
    """A measurement-based erase cannot be run in reverse: ``RotDataErase`` is never adjointed."""
    graph, _ = walk.call_graph()
    reversed_erase = lambda b: isinstance(b, Adjoint) and isinstance(b.subbloq, RotDataErase)
    assert not any(reversed_erase(b) for b in graph.nodes)


def test_rejects_mismatched_sub_bloqs(walk):
    be = walk.block_encoding
    inner = InnerPrepare.from_coeffs(np.ones((2, 2, 4)), num_orb=3, b_k2=3)  # B = 3, not 2
    with pytest.raises(ValueError):
        DFTHCBlockEncoding(outer=be.outer, inner=inner, select=be.select)


# --- correctness: the signal block, by exact contraction ----------------------------


def _tiny(seed=0):
    """``N = 2, R = 2, B = 1, C = 1``: two SF rows, one D1 and one Q1, signed inner weights.

    ``N = 2`` is one Givens angle per direction, so the loaded direction is exactly
    ``(cos phi, sin phi)``.
    """
    rng = np.random.default_rng(seed)
    uniform = dict(s=4, phase_grad_bitsize=4, exact_erase=True)  # 4 and 2 values: Hadamards only
    return DFTHCBlockEncoding(
        outer=OuterPrepare.from_weights(rng.random(4) + 0.1, b_k1=2, **uniform),
        inner=InnerPrepare.from_coeffs(
            rng.standard_normal((2, 1, 2)), num_orb=2, b_k2=2, **uniform
        ),
        select=InnerSelect.from_angles(
            rng.uniform(0, 2 * np.pi, (2, 1, 1)), rng.uniform(0, 2 * np.pi, (2, 1)),
            num_c=1, num_pos=1, b_rot=4, exact_erase=True,
        ),
    )


def _signal_block(bloq, **pinned):
    """``<0|bloq|0>`` on the ancillas, the ``pinned`` qubits too: a ``4**N`` matrix on
    ``(psi_up, psi_dn)``."""
    num_orb, b_rot = bloq.select.num_orb, bloq.select.b_rot
    bb = BloqBuilder()
    psi = {name: bb.add_register(name, QAny(num_orb)) for name in ('psi_up', 'psi_dn')}
    width = {name: bloq.signature.get_left(name).total_bits()
             for name in OUTER_ANCILLAS + INNER_ANCILLAS + tuple(pinned)}
    grad = PhaseGradientState(b_rot)
    out = bb.add_d(bloq, **psi, phase_grad=bb.add(grad),
                   **{name: bb.add(IntState(pinned.get(name, 0), n)) for name, n in width.items()})
    bb.add(grad.adjoint(), phase_grad=out.pop('phase_grad'))
    for name, n in width.items():
        bb.add(IntEffect(pinned.get(name, 0), n), val=out.pop(name))
    return bb.finalize(**out).tensor_contract()


def _reference(be):
    """``sum_x p_x avg_{sigma_1} (2 A^dag A - I)``, with ``A_SF = sum_b q_b sign(w_b) M_b``
    (``M_b`` the spin-summed Majorana pair, ``M_B = I``) and ``A = a_u, a_u^dag`` on D1/Q1."""
    o, i, s = be.outer, be.inner, be.select
    eye = np.eye(4**s.num_orb)
    u_sf, u_tail = (np.apply_along_axis(vector_from_angles, -1, a, s.b_rot)
                    for a in (s.angle_array, s.tail_array))
    p = alias_probabilities(o.alt, o.keep, o.b_k1)
    h = 0
    for x in range(o.num_x):
        if x < s.num_sf:
            u, q = u_sf[x // s.num_c], alias_probabilities(i.alt[x], i.keep[x], i.b_k2)
            terms = [(_generator(u[b], 'SF', True) + _generator(u[b], 'SF', False)) / 2
                     for b in range(s.num_b)] + [eye]
            a = sum(q[b] * (-1) ** i.sgn[x][b] * terms[b] for b in range(s.num_b + 1))
            per_spin = [a, a]
        else:
            j = x - s.num_sf
            kind = 'D1' if j < s.num_pos else 'Q1'
            per_spin = [_generator(u_tail[j], kind, on_up) for on_up in (False, True)]
        h = h + p[x] * sum(2 * a.conj().T @ a - eye for a in per_spin) / 2
    return h


@pytest.mark.intensive
def test_signal_block_is_the_block_encoding():
    """(1-27) ``<0|BE|0>`` is ``sum_x p_x (2 A_x^dag A_x - I)``, to machine precision.

    With ``O_x = sqrt(w_x) A_x`` that is ``H_SA / Lambda - I``, ``Lambda = sum(w) / 2``.
    """
    be = _tiny()
    # The signs are exercised: a sign shared by a whole row drops out of A^dag A.
    assert any(len(set(row)) > 1 for row in be.inner.sgn)
    assert np.allclose(_signal_block(be), _reference(be), atol=1e-10)


def _factors(name):
    return {k: v for k, v in load_factors(name)[0].items() if k in KEYS}


# (N, R, B, C) = (2, 3, 1, 1): X_o = 5, so Uniform(x_o) fails; H2 has B + 1 = 3, so Uniform(b) does.
FACTORS = {
    'x_o_fails': lambda: random_factors(2, 3, 1, 1, seed=3),
    'none_fail': lambda: random_factors(2, 2, 1, 1, seed=3),
    'h2': lambda: _factors('h2_sto3g_R2_B2_C1'),
}


def _from_factors(name):
    return DFTHCBlockEncoding.from_factors(
        **FACTORS[name](), b_rot=4, b_k1=2, b_k2=2, s=4, exact_erase=True
    )


def _h_sa(be):
    """``H_SA`` of the decomposed Hamiltonian ``be`` encodes, a ``16 x 16`` Fock matrix."""
    enc = be.encoded_factors()
    e_sos = walk_e_sos(enc['W'], enc['w_B'], enc['w_minus'])
    return hamiltonian_matrix(*walk_integrals(**enc)) - e_sos * np.eye(16)


def _integrals(matrix, n):
    """The ``(h1e, eri)`` and constant whose ``hamiltonian_matrix`` is ``matrix``: least squares
    over symmetric ``h1e`` and 8-fold symmetric ``eri``, which it maps one to one."""
    pairs = list(itertools.combinations_with_replacement(range(n), 2))
    basis = []
    for p, q in pairs:
        h1e = np.zeros((n, n))
        h1e[p, q] = h1e[q, p] = 1
        basis.append((h1e, np.zeros((n,) * 4)))
    for (p, q), (r, s) in itertools.combinations_with_replacement(pairs, 2):
        eri = np.zeros((n,) * 4)
        for i, j, k, l in itertools.product((p, q), (q, p), (r, s), (s, r)):
            if {i, j} == {p, q} and {k, l} == {r, s}:
                eri[i, j, k, l] = eri[k, l, i, j] = 1
        basis.append((np.zeros((n, n)), eri))
    columns = [hamiltonian_matrix(h1e, eri).ravel() for h1e, eri in basis] + [np.eye(4**n).ravel()]
    fit = np.linalg.lstsq(np.array(columns).T, matrix.ravel(), rcond=None)
    coeffs = fit[0].real
    assert fit[1].size == 0 or fit[1].max() < 1e-16
    h1e = sum(c * h for c, (h, _) in zip(coeffs, basis))
    eri = sum(c * g for c, (_, g) in zip(coeffs, basis))
    return h1e, eri, coeffs[-1]


@pytest.mark.parametrize('name', [
    'h2',
    pytest.param('x_o_fails', marks=pytest.mark.intensive),
    pytest.param('none_fail', marks=pytest.mark.intensive),
])
def test_from_factors_encodes_the_hamiltonian(name):
    """(1-27) Built from a decomposed Hamiltonian, ``<0|BE|0> = H_SA / Lambda - I`` to machine
    precision, ``H_SA`` the Fock matrix of ``walk_integrals - walk_e_sos`` of the decomposed
    Hamiltonian it encodes, rounding included: the tables, their signs, the SF ``-Z``, the D1/Q1
    split, a failed ``Uniform(b)``, a failed ``Uniform(x_o)`` and ``Lambda`` in one check. The
    integrals and ``E_SoS`` read back off the block are those of the encoded factors."""
    be = _from_factors(name)
    assert any(len(set(row)) > 1 for row in be.inner.sgn)  # a row that mixes signs
    fails = [prep.uniform.failure_probability() > 0 for prep in (be.outer, be.inner)]
    expected = dict(x_o_fails=[True, False], none_fail=[False, False], h2=[False, True])
    assert fails == expected[name]
    block = _signal_block(be)
    assert np.allclose(block, _h_sa(be) / be.Lambda - np.eye(16), atol=1e-10)
    enc = be.encoded_factors()
    h1e, eri, minus_e_sos = _integrals(be.Lambda * (block + np.eye(16)), n=2)
    for got, want in zip((h1e, eri), walk_integrals(**enc)):
        assert np.allclose(got, want, atol=1e-8)
    assert np.isclose(-minus_e_sos, walk_e_sos(enc['W'], enc['w_B'], enc['w_minus']), atol=1e-8)


@pytest.mark.parametrize('name', list(FACTORS_ERROR))
def test_encoded_factors_tend_to_the_input(name):
    """At 24 bits of ``keep``, of angle and of the ``Uniform`` rotations, the walk encodes its input
    to about ``1e-7``; the encoded weights keep ``Lambda`` exactly."""
    f = _factors(name)
    be = DFTHCWalk.from_factors(**f, b_rot=24, b_k1=24, b_k2=24, s=24).block_encoding
    enc = be.encoded_factors()
    for got, want in zip(walk_integrals(**enc), walk_integrals(**f)):
        assert np.allclose(got, want, atol=1e-6)
    e_sos = (walk_e_sos(g['W'], g['w_B'], g['w_minus']) for g in (enc, f))
    assert np.isclose(*e_sos, rtol=0, atol=1e-6)
    outer, _ = walk_inputs(enc['W'], enc['w_B'], enc['w_plus'], enc['w_minus'])
    p_x = be.outer.uniform.failure_probability()
    assert np.isclose(outer.sum() / 2, be.Lambda * (1 - p_x), rtol=1e-12)


def _cut(f, b_rot, b_k):
    """The factors with each component of ``u`` rounded to ``b_rot`` bits (rows normalized again)
    and each ``w`` to ``b_k`` bits of its array's largest."""
    out = {}
    for key in ('U', 'u_plus', 'u_minus'):
        u = np.round(np.asarray(f[key]) * 2.0 ** (b_rot - 1)) / 2.0 ** (b_rot - 1)
        out[key] = u / np.linalg.norm(u, axis=-1, keepdims=True)
    for key in ('W', 'w_B', 'w_plus', 'w_minus'):
        step = np.abs(f[key]).max() * 2.0**-b_k
        out[key] = np.round(np.asarray(f[key]) / step) * step
    return out


def _integral_errors(f, g):
    """``h1e`` and ``eri`` of ``g`` against those of ``f``: spectral (``eri`` as an ``N^2 x N^2``
    matrix), Frobenius and max norm."""
    errors = {}
    for name, got, want in zip(('h1e', 'eri'), walk_integrals(**g), walk_integrals(**f)):
        d = got - want
        m = d.reshape(len(d) ** (d.ndim // 2), -1)
        errors |= {(name, 'spectral'): np.linalg.norm(m, 2), (name, 'Frobenius'): np.linalg.norm(m),
                   (name, 'max'): np.abs(d).max()}
    return errors


# The encoded error over that of the input factors rounded to the same bits stays below these. Most
# of it is the angles: b_rot-bit Givens angles move a direction about 5 times a rounded component's.
ROUNDING_BOUNDS = {('h1e', 'spectral'): 25, ('h1e', 'Frobenius'): 20, ('h1e', 'max'): 25,
                   ('eri', 'spectral'): 20, ('eri', 'Frobenius'): 20, ('eri', 'max'): 35}


@pytest.mark.parametrize('name', list(FACTORS_ERROR))
def test_encoded_errors_are_bounded_by_the_rounded_input(name):
    """With exact equal superpositions (``exact_prep``), the encoded factors are off the input by at
    most ``ROUNDING_BOUNDS`` times the input rounded to the same bits, ``u`` to ``b_rot`` and ``w``
    to ``b_k1 = b_k2``, for ``b_rot, b_k`` in ``{6, 8, ..., 14}``."""
    f = _factors(name)
    for b_rot, b_k in itertools.product(range(6, 15, 2), repeat=2):
        be = DFTHCBlockEncoding.from_factors(**f, b_rot=b_rot, b_k1=b_k, b_k2=b_k, s=4)
        enc = _integral_errors(f, be.encoded_factors(exact_prep=True))
        cut = _integral_errors(f, _cut(f, b_rot, b_k))
        for key, bound in ROUNDING_BOUNDS.items():
            assert enc[key] <= bound * cut[key], (b_rot, b_k, key, enc[key] / cut[key])


def test_exact_prep_leaves_out_only_the_failed_uniform():
    """On H2, where ``Uniform(b)`` fails, ``exact_prep`` takes its ``1 - P(succ_b = 0)`` off ``W``
    and changes ``w_B``; the directions and ``w_pm`` stay."""
    be = _from_factors('h2')
    fail = be.inner.uniform.failure_probability()
    exact, enc = be.encoded_factors(exact_prep=True), be.encoded_factors()
    assert fail > 0 and np.allclose(enc['W'], (1 - fail) * exact['W'])
    assert not np.allclose(enc['w_B'], exact['w_B'])
    for key in ('U', 'u_plus', 'u_minus', 'w_plus', 'w_minus'):
        assert np.array_equal(enc[key], exact[key])


def test_from_factors_takes_bits_or_a_precision():
    """Per PREPARE, its bits or a precision: here ``b_k1`` from ``precision_outer`` over the
    ``N + RC = 4`` outer values, and ``b_k2`` as given."""
    f = random_factors(2, 2, 1, 1, seed=3)
    be = DFTHCBlockEncoding.from_factors(**f, b_rot=4, precision_outer=1e-2, b_k2=3, s=4)
    assert (be.outer.b_k1, be.inner.b_k2) == (5, 3)
    with pytest.raises(ValueError):
        DFTHCBlockEncoding.from_factors(**f, b_rot=4, b_k1=3, precision_outer=1e-2, b_k2=3, s=4)


@pytest.mark.intensive
def test_controlled_block_encoding_is_the_identity_on_ctrl_0():
    """With ``REF_T2`` skipped, the rest of the circuit is ``U^dag U = I``."""
    be = _tiny()
    assert np.allclose(_signal_block(be.controlled(), ctrl=0), np.eye(16), atol=1e-10)


@pytest.mark.intensive
def test_controlled_block_encoding_is_the_block_on_ctrl_1():
    """On ``ctrl = 1``, ``REF_T2``'s ``-1`` moves onto ``ctrl``: the block of the uncontrolled
    one, the failed ``Uniform(x_o)`` included."""
    be = _from_factors('x_o_fails')
    got = _signal_block(be.controlled(), ctrl=1)
    assert np.allclose(got, _h_sa(be) / be.Lambda - np.eye(16), atol=1e-10)


@pytest.mark.parametrize('controlled', [False, True])
def test_ref_t2_is_the_reflection_about_succ_x_o_1(walk, controlled):
    """(23) ``REF_T2`` as the block encoding adds it, contracted on its own: ``2|0><0|_B - I`` on
    ``succ_x_o = 1``, ``-I`` on ``succ_x_o = 0``; controlled, also the identity on ``ctrl = 0``."""
    be = walk.block_encoding.controlled() if controlled else walk.block_encoding
    names = ('ctrl',) * controlled + ('succ_x_o',) + INNER_ANCILLAS
    width = {name: 1 for name in ('ctrl', 'succ_x_o')}
    width |= {name: be.signature.get_left(name).total_bits() for name in INNER_ANCILLAS}
    bb = BloqBuilder()
    soqs = {name: bb.add_register(name, width[name]) for name in names}
    be._add_ref_t2(bb, soqs)
    got = bb.finalize(**{name: soqs[name] for name in names}).tensor_contract()
    n_b = sum(width[name] for name in INNER_ANCILLAS)
    want = []
    for idx in range(2 ** (n_b + 1 + controlled)):
        ctrl, succ_x_o, b = idx >> (n_b + 1), idx >> n_b & 1, idx % 2**n_b  # MSB first
        want.append(1 if controlled and not ctrl else 1 if succ_x_o and b == 0 else -1)
    assert np.allclose(got, np.diag(want))


# --- cost ---------------------------------------------------------------------------


def _block_sum(be):
    return (
        _toffoli(be.outer) + _toffoli(be.outer.adjoint())
        + 2 * (_toffoli(be.inner) + _toffoli(be.inner.adjoint()))
        + _toffoli(be.select) + _toffoli(be.select.adjoint())
        + _toffoli(be.ref_t2)
    )


def _boxes(bloq):
    return [inst.bloq for inst in bloq.decompose_bloq().bloq_instances]


def test_cost_is_the_block_sum(walk):
    """No cost/decomposition divergence: the call graph is exactly the 28 steps."""
    be = walk.block_encoding
    assert _toffoli(be) == _toffoli(be.decompose_bloq()) == _block_sum(be)
    walk_sum = _block_sum(be) + _toffoli(walk.ref_ab)
    assert _toffoli(walk) == _toffoli(walk.decompose_bloq()) == walk_sum


def test_only_the_reflections_take_the_control(walk):
    """Controlled, ``REF_T2`` merges ``ctrl`` with ``succ_x_o`` for one Toffoli, its ``-1``
    becomes a ``Z`` on ``ctrl``, and ``REF_aB`` takes it for free: controlled-``W`` is one Toffoli
    more than ``W``."""
    be, cbe, cwalk = walk.block_encoding, walk.block_encoding.controlled(), walk.controlled()
    assert [b for b in _boxes(cbe) if b not in _boxes(be)] == [cbe.ref_t2, cbe.ref_t2_phase]
    assert _boxes(cwalk) == [cbe, cwalk.ref_ab]
    assert _toffoli(cbe) == _toffoli(cbe.decompose_bloq()) == _toffoli(be) + 1
    assert _toffoli(cwalk) == _toffoli(cwalk.decompose_bloq()) == _toffoli(walk) + 1


def test_femoco_cost_by_block():
    """Block by block at FeMoco-54, as in the notebook's walk table: 9641 Toffolis, no ``Rz``."""
    w = _walk(**FEMOCO)
    be = w.block_encoding
    ours = dict(
        outer=_toffoli(be.outer) + _toffoli(be.outer.adjoint()),
        inner=_toffoli(be.inner) + _toffoli(be.inner.adjoint()),
        select=_toffoli(be.select), ref_t2=_toffoli(be.ref_t2), ref_ab=_toffoli(w.ref_ab),
    )
    calls = dict(outer=1, inner=2, select=2, ref_t2=1, ref_ab=1)
    assert ours == dict(outer=306, inner=1133, select=3502, ref_t2=20, ref_ab=45)
    assert _toffoli(be.select.adjoint()) == ours['select']
    assert _toffoli(be) == 9596 and _toffoli(be.controlled()) == 9597
    assert _toffoli(w) == sum(calls[k] * ours[k] for k in calls) == 9641
    assert _toffoli(w.controlled()) == 9642
    assert get_cost_value(w, QECGatesCost()).rotation == 0


def test_symbolic_cost_is_the_block_sum():
    w = DFTHCWalk.symbolic(
        *sympy.symbols('N R B C b_rot b_k1 b_k2 k_1 k_2 k_e1 k_e2', positive=True)
    )
    be = w.block_encoding
    assert sympy.simplify(_toffoli(w) - _block_sum(be) - _toffoli(w.ref_ab)) == 0
    assert sympy.simplify(_toffoli(w.controlled()) - _toffoli(w) - 1) == 0
    assert get_cost_value(w, QECGatesCost()).rotation == 0
    for bloq in (be, w):
        with pytest.raises(DecomposeTypeError):
            bloq.decompose_bloq()
