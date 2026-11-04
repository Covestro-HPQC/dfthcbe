r"""Tests for the two PREPAREs -- steps 1-4 and 5-8 of the DFTHC block encoding -- and the
signs of step 9.

**Correctness** is the prepared state, by tensor simulation at tiny ``(L, b_k)``: contract from
``|0>`` and marginalize over the alias garbage, which must give the ``b_k``-bit approximation of
``w / lambda``. For :class:`InnerPrepare` followed by :class:`InnerSigns` also the *signs*: every
branch of a given ``b`` must carry ``sign(w_b)``, and since the garbage differs per branch that
cannot be read off a marginal -- it is read off the amplitudes directly. The signs of a failed
``Uniform(b)`` by phased classical simulation of steps 6-9.

**Cost** is the call graph at FeMoco-54, block by block, plus the symbolic closed forms of the
loads. The reverse pass is ``.adjoint()``, whose QROAM erase is a cost-only bloq --
measurement-based uncomputation has no unitary decomposition -- so there only the cost is
checked.
"""

import itertools

import attrs
import cirq
import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder, BQUInt, DecomposeTypeError, Side
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan
from qualtran.bloqs.basic_gates import CSwap, Hadamard, IntEffect, IntState, OnEach
from qualtran.linalg.lcu_util import sub_bit_prec_from_epsilon
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.simulation.classical_sim import do_phased_classical_simulation
from qualtran.symbolics import log2

from dfthcbe.alias_data import QROAMCleanHetero
from dfthcbe.prepare import (
    alias_table, InnerPrepare, InnerSigns, OuterPrepare, signed_alias_table,
    UniformViaPhaseGradient,
)
from dfthcbe.utils import aligned_blocks, num_blocks, popcount

# FeMoco-54, the README's worked example: N, R, B, C, b_k1 = b_k2, k_1, k_2.
FEMOCO = dict(num_orb=54, num_r=10, num_b=27, num_c=27, b_k=15, k_1=2, k_2=4)

W_OUT = np.array([1.0, 2.0, 3.0, 0.5])                 # 4 outer weights, all non-negative
W_SF = np.array([[[1.0, -2.0, 3.0, 0.5]],              # (R=2, C=1, B+1=4), signed
                 [[-1.0, 1.0, -0.25, 2.0]]])


def _toffoli(bloq):
    """Toffoli count: ``CCZ``, ``CSwap`` and every 4 raw ``T``. Rotations are counted apart, by
    ``_rotations``: Qualtran leaves the two ``Rz`` of ``Uniform`` unsynthesized.
    """
    c = get_cost_value(bloq, QECGatesCost())
    return c.toffoli + c.cswap + c.and_bloq + c.t / 4


def _rotations(bloq):
    return get_cost_value(bloq, QECGatesCost()).rotation


def _prepared(bloq, **pinned):
    """The state ``bloq`` makes from ``|0>``, with the named inputs pinned and projected back.

    Register order in the result is signature order minus the pinned ones, so the selection
    register is the most significant index.
    """
    bb = BloqBuilder()
    width = {reg.name: reg.dtype.num_qubits for reg in bloq.signature.lefts()}
    out = bb.add_d(bloq, **{name: bb.add(IntState(pinned.get(name, 0), n))
                            for name, n in width.items()})
    for name, val in pinned.items():
        bb.add(IntEffect(val, width[name]), val=out.pop(name))
    return bb.finalize(**out).tensor_contract()


def _femoco_outer(**kwargs):
    n_x = FEMOCO['num_orb'] + FEMOCO['num_r'] * FEMOCO['num_c']
    weights = np.random.default_rng(0).random(n_x) + 0.1
    return OuterPrepare.from_weights(
        weights, b_k1=FEMOCO['b_k'], log_block_size=FEMOCO['k_1'], **kwargs
    )


def _femoco_inner(**kwargs):
    shape = (FEMOCO['num_r'], FEMOCO['num_c'], FEMOCO['num_b'] + 1)
    w = np.random.default_rng(1).standard_normal(shape)
    return InnerPrepare.from_coeffs(
        w, num_orb=FEMOCO['num_orb'], b_k2=FEMOCO['b_k'], log_block_size=FEMOCO['k_2'], **kwargs
    )


# --- structure ----------------------------------------------------------------------


def test_outer_signature():
    """``x_o`` and the outer spin, plus the alias garbage the adjoint has to clear."""
    prep = OuterPrepare.from_weights(W_OUT, b_k1=4, log_block_size=1)
    assert [r.name for r in prep.signature] == [
        'x_o', 'succ_x_o', 'sigma_1', 'm_x_o', 'less_than_x_o', 'alt_x_o', 'keep_x_o',
        'junk_alt_x_o', 'junk_keep_x_o', 'phase_grad',
    ]
    assert prep.signature.get_left('x_o').dtype.iteration_length == prep.num_x == 4
    # The loaded tables and the 2**k - 1 rows the block swap keeps around are RIGHT.
    assert [r.name for r in prep.signature if r.side is Side.RIGHT] == [
        'alt_x_o', 'keep_x_o', 'junk_alt_x_o', 'junk_keep_x_o'
    ]
    assert prep.signature.get_right('junk_alt_x_o').shape == (1,)


def test_inner_signature():
    """``b`` and the inner spin; ``x_o`` is read-only, and ``r`` is an *output*."""
    prep = InnerPrepare.from_coeffs(W_SF, num_orb=2, b_k2=4)
    assert [r.name for r in prep.signature] == [
        'x_o', 'b', 'succ_b', 'sigma_0', 'm_b', 'less_than_b',
        'alt_b', 'keep_b', 'sgn', 'd_sgn', 'r', 'phase_grad',
    ]
    assert (prep.num_sf, prep.num_r, prep.num_b, prep.num_x) == (2, 2, 3, 4)
    spans = {r.name: r.dtype.iteration_length for r in prep.signature
             if hasattr(r.dtype, 'iteration_length')}
    assert spans == {'x_o': 4, 'b': 4}          # N + RC, and B+1 including the identity term
    assert prep.signature.get_right('r').dtype.num_qubits == prep.r_bitsize == 1


# --- correctness: the prepared state ------------------------------------------------


@pytest.mark.parametrize("log_block_size", [0, 1])
def test_outer_prepares_the_weights(log_block_size):
    """(1-4) The marginal over the garbage is the ``b_k1``-bit approximation of ``w / lambda``, and
    no amplitude is negative, as no weight is: the read's swap at ``k_1 = 1`` leaves no sign."""
    b_k1 = 3
    prep = OuterPrepare.from_weights(W_OUT, b_k1=b_k1, log_block_size=log_block_size)
    psi = _prepared(prep)
    prob = (np.abs(psi) ** 2).reshape(len(W_OUT), -1).sum(axis=1)
    assert np.allclose(prob, W_OUT / W_OUT.sum(), atol=2.0**-b_k1)
    assert np.allclose(psi.imag, 0) and np.all(psi.real > -1e-9)


def test_outer_flags_success_and_the_outer_spin():
    """(1) ``Uniform`` is exact here, so ``succ_x_o`` is deterministically 1; ``sigma_1``
    leaves in ``|+>``, uncorrelated with anything.
    """
    prep = OuterPrepare.from_weights(W_OUT, b_k1=3, log_block_size=0)
    psi = _prepared(prep).reshape(len(W_OUT), 2, 2, -1)      # x_o, succ_x_o, sigma_1, rest
    assert np.allclose(psi[:, 0], 0)                         # never flags failure
    assert np.allclose(psi[:, 1, 0], psi[:, 1, 1])           # |+> on sigma_1


@pytest.mark.parametrize("x_o", range(2))
@pytest.mark.parametrize("k_2, b_k2", [(None, 4), (1, 2)])
def test_inner_prepares_the_signed_weights(x_o, k_2, b_k2):
    """(5-9) Per SF row: the weights are ``|w_b|`` and *every* branch of ``b`` carries
    ``sign(w_b)`` -- ``Z(sgn)`` and ``CZ(less_than, d_sgn)`` together. At ``k_2 = 1`` the read's
    swap is in the circuit, with ``r`` loaded flat.
    """
    prep = InnerPrepare.from_coeffs(W_SF, num_orb=2, b_k2=b_k2, log_block_size=k_2)
    w = W_SF.reshape(-1, W_SF.shape[-1])[x_o]
    bb, soqs = BloqBuilder.from_signature(prep.signature)
    soqs = bb.add_d(prep, **soqs)
    soqs |= bb.add_d(InnerSigns(), **{n: soqs.pop(n) for n in ('sgn', 'd_sgn', 'less_than_b')})
    psi = _prepared(bb.finalize(**soqs), x_o=x_o).reshape(prep.num_b + 1, -1)

    prob = (np.abs(psi) ** 2).sum(axis=1)
    assert np.allclose(prob, np.abs(w) / np.abs(w).sum(), atol=2.0**-b_k2)
    for b, amps in enumerate(psi):
        live = amps[np.abs(amps) > 1e-9]
        assert np.allclose(live.imag, 0)
        assert np.all(np.sign(live.real) == np.sign(w[b])), b


def test_inner_tail_rows_load_nothing():
    """The ``N`` D1/Q1 rows have no ``b`` sum, so the load is a no-op there.

    ``Uniform(b)`` still runs, uncontrolled, and the alias swap then
    collapses ``b`` onto ``alt = 0``. Harmless: ``SEL`` is ``b``-independent on this branch, so
    ``PREP_b^dag SEL PREP_b = SEL`` whatever ``PREP_b`` made.
    """
    prep = InnerPrepare.from_coeffs(W_SF, num_orb=2, b_k2=4)
    for x_o in range(prep.num_sf, prep.num_x):
        psi = _prepared(prep, x_o=x_o).reshape(prep.num_b + 1, -1)
        prob = (np.abs(psi) ** 2).sum(axis=1)
        assert np.allclose(prob, [1, 0, 0, 0]), x_o


def _round_trip(prep, **pinned):
    """``PREP^dag PREP`` from ``|0>``, with the named inputs pinned and projected back."""
    width = {reg.name: reg.dtype.num_qubits for reg in prep.signature.lefts()}
    bb = BloqBuilder()
    out = bb.add_d(prep, **{name: bb.add(IntState(pinned.get(name, 0), n))
                            for name, n in width.items()})
    out = bb.add_d(prep.adjoint(), **out)
    for name, val in pinned.items():
        bb.add(IntEffect(val, width[name]), val=out.pop(name))
    return bb.finalize(**out).tensor_contract()


@pytest.mark.parametrize("k_1", [None, 1])
def test_exact_erase_outer_adjoint_is_its_inverse(k_1):
    """``exact_erase``: ``.adjoint()`` is the read's exact inverse, so ``PREP^dag PREP |0> = |0>``,
    the swap included at ``k_1 = 1``."""
    psi = _round_trip(
        OuterPrepare.from_weights(W_OUT, b_k1=3, log_block_size=k_1, exact_erase=True)
    )
    assert np.isclose(psi[0], 1)


@pytest.mark.parametrize("x_o", range(2))
@pytest.mark.parametrize("k_2", [None, 1])
def test_exact_erase_inner_adjoint_is_its_inverse(x_o, k_2):
    """The same for steps 5-8, on each SF row."""
    prep = InnerPrepare.from_coeffs(W_SF, num_orb=2, b_k2=4, log_block_size=k_2, exact_erase=True)
    assert np.isclose(_round_trip(prep, x_o=x_o)[0], 1)


def test_exact_erase_warns():
    """The exact inverse is for simulation: it costs a second read, not the erase we report."""
    with pytest.warns(UserWarning, match='exact unitary adjoint'):
        OuterPrepare.from_weights(W_OUT, b_k1=3, exact_erase=True).qroam.adjoint()


def _cirq_read(read, x, b):
    """The targets ``read``'s decomposition writes on ``|x, b>|0>``, from a cirq statevector."""
    quregs = get_named_qubits(read.signature.lefts())
    circuit = read.decompose_bloq().to_cirq_circuit(cirq_quregs=quregs)
    named = [q for reg in read.signature.lefts() for q in quregs[reg.name].flat]
    qubits = named + sorted(circuit.all_qubits() - set(named))
    bits = dict.fromkeys(qubits, 0)
    for name, val in (('selection0', x), ('selection1', b)):
        bits.update(zip(quregs[name].flat, read.signature.get_left(name).dtype.to_bits(val)))
    psi = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, qubit_order=qubits, initial_state=int(''.join(map(str, bits.values())), 2)
    ).final_state_vector
    out = dict(zip(qubits, map(int, format(int(np.argmax(np.abs(psi))), f'0{len(qubits)}b'))))
    return [int(''.join(str(out[q]) for q in quregs[f'target{t}_'].flat), 2)
            for t in range(len(read.target_registers))]


@pytest.mark.parametrize("shape, k", [((3, 2), (0, 1)), ((3, 4), (0, 2)), ((3, 4), (0, 1))])
def test_a_block_as_wide_as_its_axis_reads_right(shape, k):
    """With ``2**k`` the whole axis, one batch is left: the read still costs what it decomposes
    to, and writes ``data[x, b]`` at every address (the half block as the control)."""
    rng = np.random.default_rng(1)
    data = [rng.integers(0, 4, size=shape), rng.integers(0, 2, size=shape)]
    read = QROAMCleanHetero.build_from_data(*data, target_bitsizes=(2, 1), log_block_sizes=k)
    assert _toffoli(read) == _toffoli(read.decompose_bloq())
    for x, b in np.ndindex(shape):
        assert _cirq_read(read, x, b) == [int(d[x, b]) for d in data], (x, b)


# --- a failed Uniform(b): what steps 6-9 do with b >= B + 1 ----------------------------


@attrs.frozen
class _AnyUInt(type(BQUInt(1, 2)._bit_encoding)):
    """``BQUInt``'s encoding without its bound, as a failed ``Uniform`` leaves ``b``; no bits read
    0 (the offset of an unblocked read). Only the simulator's checks change, not the gates."""

    def from_bits(self, bits):
        return int(''.join(map(str, bits)) or '0', 2)


STEP_5 = (UniformViaPhaseGradient, Hadamard, OnEach)


def _steps_5_to_9(prep):
    """Inner PREP, then the signs, flattened to gates but for step 5."""
    bb = BloqBuilder()
    out = bb.add_d(prep, **{reg.name: bb.add_register(reg) for reg in prep.signature.lefts()})
    out['sgn'], out['d_sgn'], out['less_than_b'] = bb.add_t(
        InnerSigns(), sgn=out['sgn'], d_sgn=out['d_sgn'], less_than_b=out['less_than_b']
    )
    return bb.finalize(**out).flatten(lambda binst: not isinstance(binst.bloq, STEP_5))


FAILED_UNIFORM = [(5, 1)] + [
    pytest.param(num_b1, k_2, marks=pytest.mark.intensive)
    for num_b1 in (3, 5, 6, 7, 11, 12) for k_2 in range(num_b1.bit_length())
    if (num_b1, k_2) != (5, 1)
]


@pytest.mark.parametrize("num_b1, k_2", FAILED_UNIFORM)
def test_failure_signs_are_what_steps_6_to_9_give(num_b1, k_2, monkeypatch):
    """Phased classical simulation of steps 6-9 on the basis states step 5 prepares,
    ``(b, succ_b = [b <= B])``, over every ``b < 2**lg(B+1)`` and ``m_b`` of four signed SF rows: a
    valid ``b`` gets the sign of the weight it selects, the failed ones on average
    ``failure_signs()``. ``B + 1 = 5`` at ``k_2 = 1``: one failed ``b`` reads entry 4, two the zero
    padding."""
    rng = np.random.default_rng(3)
    prep = InnerPrepare.from_coeffs(
        rng.standard_normal((2, 2, num_b1)), num_orb=2, b_k2=2, log_block_size=k_2
    )
    steps = _steps_5_to_9(prep)
    assert not any(isinstance(i.bloq, QROAMCleanHetero) for i in steps.bloq_instances)  # gates
    for bloq in STEP_5:  # step 5 is given by the inputs
        monkeypatch.setattr(bloq, 'on_classical_vals', lambda self, **vals: vals)
    unbounded = property(lambda dtype: _AnyUInt(dtype.bitsize, 2**dtype.bitsize))
    monkeypatch.setattr(BQUInt, '_bit_encoding', unbounded)
    sign, num_fail = 1 - 2 * np.array(prep.sgn), 2**prep.b_bitsize - num_b1
    mean = np.zeros(prep.num_sf)
    for x_o, b, m_b in itertools.product(
        range(prep.num_sf), range(2**prep.b_bitsize), range(2**prep.b_k2)
    ):
        out, phase = do_phased_classical_simulation(steps, dict(
            x_o=x_o, b=b, succ_b=int(b < num_b1), sigma_0=0, m_b=m_b, less_than_b=0, phase_grad=0
        ))
        if b < num_b1:
            assert phase == sign[x_o, out['b']], (x_o, b, m_b)
        else:
            mean[x_o] += phase.real / (num_fail * 2**prep.b_k2)
    assert np.allclose(mean, prep.failure_signs())


# --- cost ---------------------------------------------------------------------------


@pytest.mark.parametrize("prep", [
    OuterPrepare.from_weights(W_OUT, b_k1=4, log_block_size=1),
    InnerPrepare.from_coeffs(W_SF, num_orb=2, b_k2=4, log_block_size=1),
    InnerPrepare.from_coeffs(W_SF, num_orb=2, b_k2=4, log_block_size=1).adjoint(),
])
def test_cost_is_the_block_sum(prep):
    """No cost/decomposition divergence: the call graph is exactly what gets built."""
    assert get_cost_value(prep, QECGatesCost()) == get_cost_value(
        prep.decompose_bloq(), QECGatesCost()
    )


def test_outer_cost():
    """At FeMoco-54: 212 + 94 = 306 Toffolis, no ``Rz``.

    The load is ``ceil(324/4) + (2**k_1-1) b_1 - 2 = 151``, the ``-2`` being QROM's free root;
    the inequality test and the alias swap are ``b_1 = b_k1 + ceil lg(N+RC) = 24``; the erase is
    33. ``Uniform(x_o)`` and its flag are 37: as ``324 = 4 x 81``, 20 for the amplitude
    amplification on the top 7 qubits, ``2 (s - 2) = 10`` for its two rotations at ``s = 7``, and
    7 for the flag, comparing those 7 qubits with 81.
    """
    prep, n_x, b_k1 = _femoco_outer(), 324, FEMOCO['b_k']
    assert prep.num_x == n_x and prep.sel_bitsize == 9
    assert _toffoli(prep.uniform) == 20 + 10 + 7
    assert _toffoli(prep.qroam) == 151
    assert _toffoli(LinearDepthGreaterThan(b_k1)) + _toffoli(CSwap(9)) == b_k1 + 9 == 24
    assert _toffoli(prep.qroam.adjoint()) == 33

    assert (_toffoli(prep), _rotations(prep)) == (212, 0)
    assert (_toffoli(prep.adjoint()), _rotations(prep.adjoint())) == (94, 0)


def test_inner_cost():
    """At FeMoco-54: 913 + 220 = 1133 Toffolis, no ``Rz``.

    The inequality test and alias swap are ``b_k2 + ceil lg(B+1) = 20``. The read writes
    ``alt | keep | sgn | d_sgn | r`` from ``RC ceil((B+1)/2**k_2) = 540`` live leaves; ``r`` is
    flat, so only the other 22 bits go through the swap: ``540 + 15 * 22 = 870``, plus 4 for the
    pruned zero rows of the D1/Q1 tail, less QROM's free root, is 872. The erase's one-hot holds
    all of ``b`` and the low ``k_e2 = 2`` bits of ``x_o``, ``4 * 28 = 112`` values; its walk takes
    the ``ceil(270 / 4) = 68`` SF leaves and ``beta(68, 81) = 3`` past the zero rows:
    ``68 + 3 + 112 - 4 = 179``. ``Uniform(b)`` and its flag are 21, as for ``Uniform(x_o)`` on
    the top 3 qubits of ``28 = 4 x 7``.
    """
    prep, b_k2 = _femoco_inner(), FEMOCO['b_k']
    assert (prep.num_sf, prep.num_r, prep.num_b, prep.num_x) == (270, 10, 27, 324)
    assert _toffoli(prep.uniform) == 8 + 10 + 3                        # 28 = 4 x 7
    assert _toffoli(prep.qroam) == 872
    assert _toffoli(LinearDepthGreaterThan(b_k2)) + _toffoli(CSwap(5)) == b_k2 + 5 == 20
    assert prep.qroam.erase_log_block_sizes == (2, 0)
    assert _toffoli(prep.qroam.adjoint()) == 179

    assert (_toffoli(prep), _rotations(prep)) == (913, 0)
    assert (_toffoli(prep.adjoint()), _rotations(prep.adjoint())) == (220, 0)


def test_the_default_block_size_is_the_cheapest_read():
    """``log_block_size=None`` takes the cheapest read; at FeMoco that is the paper's ``k_1 = 2``,
    ``k_2 = 4``."""
    rng = np.random.default_rng(0)
    n_x = FEMOCO['num_orb'] + FEMOCO['num_r'] * FEMOCO['num_c']
    w_out = rng.random(n_x) + 0.1
    w_sf = rng.standard_normal((FEMOCO['num_r'], FEMOCO['num_c'], FEMOCO['num_b'] + 1))
    for build, k_max, k_paper in [
        (lambda k: OuterPrepare.from_weights(w_out, b_k1=FEMOCO['b_k'], log_block_size=k), 8, 2),
        (lambda k: InnerPrepare.from_coeffs(w_sf, num_orb=FEMOCO['num_orb'], b_k2=FEMOCO['b_k'],
                                            log_block_size=k), 4, 4),
    ]:
        reads = [_toffoli(build(k).qroam) for k in range(k_max + 1)]
        assert _toffoli(build(None).qroam) == min(reads) == reads[k_paper]


def test_the_erase_is_free_of_the_loaded_width():
    """Measurement-based uncomputation costs the same however wide the tables are.

    That asymmetry -- ``ceil(L/2**k) + 2**k`` against the load's ``ceil(L/2**k) + (2**k-1) b`` --
    is the whole reason PREPARE's garbage is carried to the end of the walk instead of unloaded.
    """
    wide, narrow = _femoco_inner(), _femoco_inner()
    wide = InnerPrepare(
        alt=narrow.alt, keep=narrow.keep, sgn=narrow.sgn, d_sgn=narrow.d_sgn,
        num_orb=narrow.num_orb, num_c=narrow.num_c, b_k2=2 * narrow.b_k2,
        log_block_size=narrow.log_block_size,
    )
    assert _toffoli(wide.qroam) > _toffoli(narrow.qroam)
    assert _toffoli(wide.qroam.adjoint()) == _toffoli(narrow.qroam.adjoint())


def _symbolic():
    """Both PREPAREs at symbolic ``N, R, B, C, b_k``, block size ``k`` and erase block ``k_e``."""
    n, r, b, c, bk, k, ke = sympy.symbols('N R B C b_k k k_e', positive=True, integer=True)
    return (
        OuterPrepare.symbolic(num_x=n + r * c, b_k1=bk, log_block_size=k, erase_log_block_size=ke),
        InnerPrepare.symbolic(
            num_orb=n, num_r=r, num_b=b, num_c=c, b_k2=bk, log_block_size=k, erase_log_block_size=ke
        ),
        (n, r, b, c, bk, k),
    )


def test_symbolic_cost_is_the_closed_form():
    """Both loads are ``leaves - 2 + (2**k - 1) * width``.

    The ``-2`` is QROM's free root. The outer load has ``ceil(X_o / 2**k)`` leaves; the inner one
    ``RC ceil((B+1)/2**k)`` on the SF rows plus ``beta(RC, X_o) s(ceil((B+1)/2**k))`` for the dead
    D1/Q1 rows, so substituting FeMoco-54 gives the built circuit's count.
    """
    outer, inner, (n, r, b, c, bk, k) = _symbolic()
    n_x, n_b = n + r * c, b + 1
    lg = lambda x: sympy.ceiling(log2(x))

    load = sympy.ceiling(n_x / 2**k) + (2**k - 1) * (bk + lg(n_x))
    assert sympy.simplify(_toffoli(outer.qroam) - (load - 2)) == 0

    width = bk + lg(n_b) + 2                               # alt | keep | sgn | d_sgn; r is flat
    b_hat = sympy.ceiling(n_b / 2**k)
    load = r * c * b_hat + aligned_blocks(r * c, n_x) * popcount(b_hat) + (2**k - 1) * width
    assert sympy.simplify(_toffoli(inner.qroam) - (load - 2)) == 0

    femoco = dict(zip((n, r, b, c, bk, k), (54, 10, 27, 27, 15, 4)))
    assert int((load - 2).subs(femoco)) == _toffoli(_femoco_inner().qroam)


def test_symbolic_erase_is_the_closed_form():
    """Both erases are ``leaves - 2`` for the walk over ``x_o`` plus ``2**k_e M - 2`` for the
    one-hot, ``M`` the values of the other addresses. The inner walk stops after the ``RC`` SF
    rows and passes the zero D1/Q1 rows in ``beta`` aligned blocks.
    """
    n, r, b, c, bk, k, ke = sympy.symbols('N R B C b_k k k_e', positive=True, integer=True)
    outer = OuterPrepare.symbolic(
        num_x=n + r * c, b_k1=bk, log_block_size=k, erase_log_block_size=ke
    )
    inner = InnerPrepare.symbolic(
        num_orb=n, num_r=r, num_b=b, num_c=c, b_k2=bk, log_block_size=k, erase_log_block_size=ke
    )
    leaves = lambda rows: sympy.ceiling(rows / 2**ke)

    erase = leaves(n + r * c) + 2**ke - 4
    assert sympy.simplify(_toffoli(outer.qroam.adjoint()) - erase) == 0

    erase = leaves(r * c) + aligned_blocks(leaves(r * c), leaves(n + r * c)) + 2**ke * (b + 1) - 4
    assert sympy.simplify(_toffoli(inner.qroam.adjoint()) - erase) == 0
    femoco = {n: 54, r: 10, b: 27, c: 27, ke: 2}
    assert int(erase.subs(femoco)) == _toffoli(_femoco_inner().qroam.adjoint()) == 179


# (R, C, N, B, k_2): beta(RC, X_o) = 1, 2, 3, 4, 1 and s(Bhat) = 1, 1, 2, 1, 3.
INNER_WALKS = [(2, 1, 2, 3, 1), (3, 1, 5, 6, 1), (2, 3, 5, 5, 0), (2, 7, 9, 12, 2), (3, 2, 2, 12, 0)]


@pytest.mark.parametrize("num_r,num_c,num_orb,num_b,k_2", INNER_WALKS)
def test_inner_load_walk_is_exact(num_r, num_c, num_orb, num_b, k_2):
    """(6) ``RC Bhat + beta(RC, X_o) s(Bhat) - 2`` ANDs, ``Bhat = ceil((B+1)/2**k_2)``: the dead
    D1/Q1 rows cost one leaf per aligned block of ``[RC, X_o)``, each a constant row of
    ``s(Bhat)`` leaves. The symbolic bloq gives the same once substituted.
    """
    w = np.random.default_rng(0).standard_normal((num_r, num_c, num_b + 1))
    built = InnerPrepare.from_coeffs(w, num_orb=num_orb, b_k2=10, log_block_size=k_2)
    rc, b_hat = num_r * num_c, -(-(num_b + 1) // 2**k_2)
    want = rc * b_hat + num_blocks(rc, rc + num_orb) * popcount(b_hat) - 2
    assert get_cost_value(built.qroam, QECGatesCost()).and_bloq == want

    n, r, b, c, k, ke = sympy.symbols('N R B C k k_e', positive=True, integer=True)
    symbolic = InnerPrepare.symbolic(
        num_orb=n, num_r=r, num_b=b, num_c=c, b_k2=10, log_block_size=k, erase_log_block_size=ke
    )
    ands = get_cost_value(symbolic.qroam, QECGatesCost()).and_bloq
    assert ands.subs({n: num_orb, r: num_r, b: num_b, c: num_c, k: k_2}) == want


FEMOCO_WALK = (10, 27, 54, 27, 4)


@pytest.mark.parametrize("num_r,num_c,num_orb,num_b,k", INNER_WALKS + [FEMOCO_WALK])
def test_symbolic_reads_and_erases_are_the_built_ones(num_r, num_c, num_orb, num_b, k):
    """Both alias reads and their erases, symbolic and substituted, cost what the built ones do.

    The erase's block size is the built one's cheapest, substituted for ``k_e``.
    """
    rng = np.random.default_rng(0)
    built = (
        OuterPrepare.from_weights(
            rng.random(num_orb + num_r * num_c) + 0.1, b_k1=6, log_block_size=k
        ),
        InnerPrepare.from_coeffs(
            rng.standard_normal((num_r, num_c, num_b + 1)), num_orb=num_orb, b_k2=6,
            log_block_size=k,
        ),
    )
    n, r, b, c, bk, k_, ke = sympy.symbols('N R B C b_k k k_e', positive=True, integer=True)
    symbolic = (
        OuterPrepare.symbolic(num_x=n + r * c, b_k1=bk, log_block_size=k_, erase_log_block_size=ke),
        InnerPrepare.symbolic(
            num_orb=n, num_r=r, num_b=b, num_c=c, b_k2=bk, log_block_size=k_,
            erase_log_block_size=ke,
        ),
    )
    for want, sym in zip(built, symbolic):
        sizes = {n: num_orb, r: num_r, b: num_b, c: num_c, bk: 6, k_: k}
        sizes[ke] = want.qroam.erase_log_block_sizes[0]
        for part in (lambda p: p.qroam, lambda p: p.qroam.adjoint()):
            assert sympy.sympify(_toffoli(part(sym))).subs(sizes) == int(_toffoli(part(want)))


@pytest.mark.parametrize("which", [0, 1])
def test_symbolic_cannot_decompose(which):
    with pytest.raises(DecomposeTypeError):
        _symbolic()[which].decompose_bloq()


def test_symbolic_tables_need_a_symbolic_block_size():
    """Qualtran cannot compare an integer ``2**k`` against a symbolic table length."""
    with pytest.raises(ValueError):
        OuterPrepare.symbolic(
            num_x=sympy.Symbol('X', positive=True), b_k1=8, log_block_size=2,
            erase_log_block_size=sympy.Symbol('k_e', positive=True),
        )


# --- the alias tables ---------------------------------------------------------------


def test_alias_table_reproduces_the_distribution():
    """Sampling ``l`` uniformly then keeping it with probability ``keep_l / 2**b_k`` gives ``w``."""
    b_k, w = 8, np.array([1.0, 2.0, 3.0, 0.5, 4.0, 1.5, 0.25, 2.5])
    alt, keep = alias_table(w, b_k)
    prob = np.zeros(len(w))
    for l, (a, k) in enumerate(zip(alt, keep)):
        prob[l] += k / 2**b_k / len(w)
        prob[a] += (1 - k / 2**b_k) / len(w)
    assert np.allclose(prob, w / w.sum(), atol=2.0**-b_k)


def test_signed_alias_table_flags_the_sign_difference():
    """``sgn`` flags ``w_l < 0``, ``d_sgn`` a sign change under the swap; a zero row loads zero."""
    w = np.array([1.0, -2.0, 3.0, -0.5])
    alt, keep, sgn, d_sgn = signed_alias_table(w, b_k=6)
    assert list(sgn) == [0, 1, 0, 1]
    assert np.array_equal(d_sgn, sgn ^ sgn[alt])
    assert all(np.array_equal(col, np.zeros(4, int))
               for col in signed_alias_table(np.zeros(4), b_k=6))


def test_bits_or_a_precision():
    """A precision gives Qualtran's bits for the values it samples, ``N + RC = 4`` outer and
    ``B + 1 = 4`` inner (not ``N + RC = 8``); bits and a precision are one or the other."""
    assert OuterPrepare.from_weights(W_OUT, precision=1e-2).b_k1 == 5
    assert InnerPrepare.from_coeffs(W_SF, num_orb=6, precision=1e-2).b_k2 == 5
    assert sub_bit_prec_from_epsilon(4, 1e-2) == 5 != sub_bit_prec_from_epsilon(8, 1e-2)
    for kwargs in (dict(), dict(b_k1=3, precision=1e-2)):
        with pytest.raises(ValueError):
            OuterPrepare.from_weights(W_OUT, **kwargs)


# --- validation ---------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(keep=np.array([0, 1, 2])),                    # alt and keep disagree
    dict(alt=np.array([0]), keep=np.array([0])),       # a single value of x_o
    dict(keep=np.array([0, 1, 2, 2**4])),              # keep wider than b_k1
    dict(alt=np.array([[0, 1], [1, 0]])),              # not 1D
])
def test_outer_rejects_bad_tables(kwargs):
    args = dict(alt=np.arange(4), keep=np.array([1, 2, 3, 4]), b_k1=4)
    with pytest.raises(ValueError):
        OuterPrepare(**(args | kwargs))


@pytest.mark.parametrize("kwargs", [
    dict(num_c=3),                                     # RC not a multiple of C
    dict(num_c=2),                                     # ... leaving R = 1, nothing to select
    dict(keep=np.array([[0, 2**4], [1, 2]])),          # keep wider than b_k2
    dict(sgn=np.array([0, 1])),                        # not 2D
])
def test_inner_rejects_bad_tables(kwargs):
    ones = np.ones((2, 2), int)
    args = dict(alt=0 * ones, keep=ones, sgn=0 * ones, d_sgn=0 * ones,
                num_orb=2, num_c=1, b_k2=4)
    with pytest.raises(ValueError):
        InnerPrepare(**(args | kwargs))
