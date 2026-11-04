r"""The two coherent-alias-sampling PREPAREs of the DFTHC block encoding, and the inner signs.

Steps 1-4 (:class:`OuterPrepare`, over ``x_o``), 5-8 (:class:`InnerPrepare`, over ``b``) and 9
(:class:`InnerSigns`) of the ``README.md`` circuit. The UNPREPAREs -- steps 24-27 and 19-22 -- are
the PREPAREs' ``.adjoint()``: the alias garbage is left dirty on purpose so the reverse pass undoes
the swap, clears the inequality flag, and erases the alias data by X-measurement
(``QROAMCleanHetero.adjoint()``) -- or, with ``exact_erase``, by the read's exact inverse.
"""

from __future__ import annotations

from functools import cached_property
from typing import Dict, Iterator, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray
from scipy.linalg import hadamard

from qualtran import (
    Bloq,
    BloqBuilder,
    BQUInt,
    DecomposeTypeError,
    QAny,
    QBit,
    QFxp,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.arithmetic import LessThanConstant
from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan
from qualtran.bloqs.basic_gates import CSwap, CZ, Hadamard, OnEach, XGate, ZGate
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.bloqs.rotations.zpow_via_phase_gradient import ZPowConstViaPhaseGradient
from qualtran.linalg.lcu_util import (
    preprocess_probabilities_for_reversible_sampling, sub_bit_prec_from_epsilon,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import (
    acos, bit_length, ceil, floor, HasLength, is_symbolic, log2, Shaped, slen,
)

from dfthcbe.alias_data import optimal_erase_log_block_size, QROAMCleanHetero
from dfthcbe.utils import aligned_blocks, freeze_table, popcount

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt


# --- alias tables (pure numpy) ------------------------------------------------------


def keep_bits(b_k: Optional[int], precision: Optional[float], num_coeffs: int) -> int:
    r"""``b_k`` as given, or the bits that make ``|w_l - w~_l| / lambda <= precision`` for each of
    ``num_coeffs`` weights: Qualtran's ``sub_bit_prec_from_epsilon``. Give exactly one."""
    if (b_k is None) == (precision is None):
        raise ValueError(f"give b_k or a precision, exactly one; got {b_k} and {precision}.")
    return b_k if precision is None else sub_bit_prec_from_epsilon(num_coeffs, precision)


def alias_table(weights, b_k: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""``(alt, keep)`` for one row of non-negative weights, ``keep`` to ``b_k`` bits.

    Index ``l`` survives with probability ``keep_l / 2**b_k`` and is replaced by ``alt_l``
    otherwise, so the sampled distribution is a ``b_k``-bit approximation of ``w / sum(w)``.
    An all-zero row loads nothing -- that is the D1/Q1 tail, which has no ``b`` sum.
    """
    w = np.asarray(weights, float)
    if not w.sum():
        return np.zeros(w.size, int), np.zeros(w.size, int)
    alt, keep, _ = preprocess_probabilities_for_reversible_sampling(list(w), sub_bit_precision=b_k)
    return np.array(alt), np.array(keep)


def signed_alias_table(weights, b_k: int) -> Tuple[np.ndarray, ...]:
    r"""``(alt, keep, sgn, d_sgn)`` for signed weights: the alias table and two sign columns.

    ``sgn_l`` is the sign of ``w_l`` itself, ``d_sgn_l`` flags ``sign(w_l) != sign(w_alt_l)``. A
    ``Z`` on the first and a ``CZ`` between the second and the inequality flag leave *every*
    branch carrying the sign of whichever index survived the swap.
    """
    w = np.asarray(weights, float)
    alt, keep = alias_table(np.abs(w), b_k)
    sgn = (w < 0).astype(int)
    return alt, keep, sgn, sgn ^ sgn[alt]


def alias_probabilities(alt, keep, b_k: int) -> np.ndarray:
    r"""The distribution an alias table of ``L`` entries prepares.

    ``p_l = (keep_l + sum_{j: alt_j = l} (2**b_k - keep_j)) / (L 2**b_k)``.
    """
    keep = np.asarray(keep) / 2**b_k
    p = keep.copy()
    np.add.at(p, np.asarray(alt), 1 - keep)
    return p / len(keep)


def _toffolis(bloq: Bloq) -> int:
    """Toffolis of a read: ``And``, ``Toffoli``, ``CSwap``, and every 4 ``T`` (it has no ``Rz``)."""
    c = get_cost_value(bloq, QECGatesCost())
    return c.and_bloq + c.toffoli + c.cswap + c.t // 4


def _check_symbolic_block_size(log_block_size) -> None:
    """``SelectSwapQROM`` asserts ``2**k <= len(table)``, which needs both sides symbolic."""
    if not is_symbolic(log_block_size):
        raise ValueError(f"symbolic tables need a symbolic log_block_size; got {log_block_size}.")


def _leaf(v: int, n: int, length: int) -> int:
    """The leaf of Qualtran's unary iteration over ``[0, length)`` on ``n`` bits that ``v`` reaches,
    also for ``v >= length``: a half starting at or past ``length`` is never entered, so its bit is
    not read."""
    lo = 0
    for j in reversed(range(n)):
        if lo + 2**j < length and v >> j & 1:
            lo += 2**j
    return lo


# --- the shared alias-sampling skeleton ----------------------------------------------


class _AliasPrepare(Bloq):
    """Shared plumbing of the two PREPAREs: the ``QROAMCleanHetero`` read and its RIGHT registers.

    Subclasses supply ``qroam`` (the read), ``data_names`` (what to call its columns) and
    ``build_composite_bloq``; this allocates the read's THRU outputs, renames them, and keeps
    the signature in step with the decomposition. The erase is the read's ``.adjoint()``: by
    X-measurement, or with ``exact_erase`` the read's exact unitary inverse, for simulation.
    """

    log_block_size: Optional['SymbolicInt']
    exact_erase: bool
    erase_log_block_size: Optional['SymbolicInt']

    def _read(self, k: 'SymbolicInt') -> QROAMCleanHetero:
        """The read, in blocks of ``2**k`` along one axis."""
        raise NotImplementedError

    @property
    def _max_log_block_size(self) -> int:
        raise NotImplementedError

    @cached_property
    def qroam(self) -> QROAMCleanHetero:
        """The read with its erase; by default the block size with the fewest Toffolis."""
        k = self.log_block_size
        if k is None:
            if is_symbolic(self.alt):
                raise ValueError("symbolic tables need a symbolic log_block_size.")
            k = min(range(self._max_log_block_size + 1), key=lambda k: _toffolis(self._read(k)))
        return self._with_erase(self._read(k))

    @property
    def data_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def _with_erase(self, qroam: QROAMCleanHetero) -> QROAMCleanHetero:
        """The read with its erase; by default the ``k_e`` with the fewest Toffolis."""
        k = self.erase_log_block_size
        if k is None:
            k = optimal_erase_log_block_size(qroam)
        blocks = (k,) + (0,) * (len(qroam.data_shape) - 1)
        return attrs.evolve(qroam, erase_log_block_sizes=blocks, exact_adjoint=self.exact_erase)

    @cached_property
    def _names(self) -> Dict[str, str]:
        return {f'target{i}_': name for i, name in enumerate(self.data_names)}

    @cached_property
    def data_registers(self) -> Tuple[Register, ...]:
        """The read's outputs, renamed and RIGHT, plus the ``2**k - 1`` swap leftovers it keeps."""
        targets = [
            attrs.evolve(reg, name=self._names[reg.name], side=Side.RIGHT)
            for reg in self.qroam.target_registers
        ]
        junk = [
            attrs.evolve(reg, name='junk_' + self._names[reg.name[5:]], side=Side.RIGHT)
            for reg in self.qroam.junk_registers
        ]
        return tuple(targets + junk)

    def add_qroam(self, bb: BloqBuilder, **selection: SoquetT) -> Dict[str, SoquetT]:
        """Allocate the read's outputs in ``|0>``, read, and return them as the ``data_names``."""
        for reg in self.qroam.target_registers:
            selection[reg.name] = bb.allocate(dtype=reg.dtype)
        for reg in self.qroam.junk_registers:
            junk = [bb.allocate(dtype=reg.dtype) for _ in range(reg.shape[0])]
            selection[reg.name] = np.array(junk)
        out = bb.add_d(self.qroam, **selection)
        renamed = {}
        for target, name in self._names.items():
            renamed[name] = out.pop(target)
            if 'junk_' + target in out:
                renamed['junk_' + name] = out.pop('junk_' + target)
        return out | renamed


# --- steps 1-4: the outer sum over x_o ----------------------------------------------


@frozen
class OuterPrepare(_AliasPrepare):
    r"""Outer PREPARE: steps 1-4 of the ``README.md`` circuit::

        (1) Uniform(x_o) over N + RC values, and flag its success;
            |+> on m_x_o and on sigma_1, the outer spin             (Clifford)
        (2) QROAM(x_o) -> alt, keep
        (3) keep <= m_x_o -> less_than_x_o
        (4) cswap alt -> x_o

    leaving ``sum_x_o sqrt(w_x_o / lambda) |x_o>|junk_x_o>``. The outer weights are all
    non-negative -- ``lambda_SF,rc**2`` on the ``RC`` SF values and ``2 w_pm^(r)`` on the ``N``
    D1/Q1 ones -- so unlike :class:`InnerPrepare` there is no sign column.

    Args:
        alt: ``(N+RC,)`` alternate indices.
        keep: ``(N+RC,)`` keep numerators, ``b_k1`` bits each.
        b_k1: bits of the keep comparison, i.e. of the prepared probabilities.
        log_block_size: ``k_1``; ``None`` takes the cheapest read.
        exact_erase: make ``.adjoint()`` the read's exact inverse instead of the
            measurement-based erase, keeping it unitary and simulable.
        erase_log_block_size: ``k_e1``, the bits of ``x_o`` in the erase's one-hot; ``None`` takes
            the cheapest.
        s: bits of ``Uniform(x_o)``'s two rotations.
        phase_grad_bitsize: width of ``phase_grad``, ``b_rot`` in a walk; by default ``s``.

    Registers:
        x_o: the outer index, ``N + RC`` values.
        succ_x_o: flags that ``Uniform(x_o)`` landed in range; gates ``Maj`` and ``REF_T2``.
        sigma_1: the outer spin qubit, left in ``|+>``.
        m_x_o: the ``b_k1`` uniform bits ``keep`` is compared against.
        less_than_x_o: the comparison result, left dirty for the adjoint to clear.
        alt_x_o, keep_x_o: the read's outputs (with ``junk_*``, the QROAM's swap leftovers).
        phase_grad: the phase gradient of ``Uniform(x_o)``'s rotations, only catalysed.
    """

    alt: Union[Tuple[int, ...], Shaped] = field(converter=lambda d: freeze_table(d, 1, 'alt'))
    keep: Union[Tuple[int, ...], Shaped] = field(converter=lambda d: freeze_table(d, 1, 'keep'))
    b_k1: 'SymbolicInt'
    log_block_size: Optional['SymbolicInt'] = None
    exact_erase: bool = False
    erase_log_block_size: Optional['SymbolicInt'] = None
    s: 'SymbolicInt' = 7
    phase_grad_bitsize: 'SymbolicInt' = field(
        default=attrs.Factory(lambda self: self.s, takes_self=True)
    )

    def __attrs_post_init__(self):
        if is_symbolic(self.alt):
            return
        if len(self.alt) != len(self.keep):
            raise ValueError(f"alt and keep must match; got {len(self.alt)}, {len(self.keep)}.")
        if len(self.alt) < 2:
            raise ValueError(f"need at least 2 values of x_o; got {len(self.alt)}.")
        if max(self.keep, default=0) >= 2**self.b_k1:
            raise ValueError(f"keep values must fit in b_k1={self.b_k1} bits.")

    @property
    def num_x(self) -> 'SymbolicInt':
        """``N + RC`` -- how many values ``x_o`` takes."""
        return self.alt.shape[0] if is_symbolic(self.alt) else len(self.alt)

    @property
    def sel_bitsize(self) -> 'SymbolicInt':
        return bit_length(self.num_x - 1)

    @property
    def data_names(self) -> Tuple[str, ...]:
        return ('alt_x_o', 'keep_x_o')

    @cached_property
    def uniform(self) -> UniformViaPhaseGradient:
        """(1) ``Uniform(x_o)`` and ``succ_x_o``."""
        return UniformViaPhaseGradient(
            self.num_x, s=self.s, phase_grad_bitsize=self.phase_grad_bitsize
        )

    def _read(self, k: 'SymbolicInt') -> QROAMCleanHetero:
        """(2) The alias tables, output width ``b_1 = ceil lg(N+RC) + b_k1``."""
        bitsizes = (self.sel_bitsize, self.b_k1)
        if is_symbolic(self.alt):
            return QROAMCleanHetero.build_from_bitsize((self.num_x,), bitsizes, log_block_sizes=k)
        return QROAMCleanHetero.build_from_data(
            np.array(self.alt), np.array(self.keep), target_bitsizes=bitsizes, log_block_sizes=k
        )

    @property
    def _max_log_block_size(self) -> int:
        return int(self.num_x).bit_length() - 1

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x_o', BQUInt(self.sel_bitsize, self.num_x)),
                Register('succ_x_o', QBit()),
                Register('sigma_1', QBit()),
                Register('m_x_o', QAny(self.b_k1)),
                Register('less_than_x_o', QBit()),
                *self.data_registers,
                Register('phase_grad', QFxp(self.phase_grad_bitsize, self.phase_grad_bitsize)),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, x_o: SoquetT, succ_x_o: SoquetT, sigma_1: SoquetT,
        m_x_o: SoquetT, less_than_x_o: SoquetT, phase_grad: SoquetT,
    ) -> Dict[str, SoquetT]:
        if is_symbolic(self.alt):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")
        # (1) An equal superposition over the N + RC values, succ_x_o flagging the rare x_o out
        # of range that its rounded rotations leave; the uniform comparison register; and the
        # outer spin sum, which D1/Q1 read and SF ignores.
        x_o, succ_x_o, phase_grad = bb.add(
            self.uniform, target=x_o, succ=succ_x_o, phase_grad=phase_grad
        )
        m_x_o = bb.add(OnEach(self.b_k1, Hadamard()), q=m_x_o)
        sigma_1 = bb.add(Hadamard(), q=sigma_1)
        # (2) the tables.
        out = self.add_qroam(bb, selection=x_o)
        x_o = out.pop('selection')
        alt, keep = out.pop('alt_x_o'), out.pop('keep_x_o')
        # (3, 4) alias sampling proper: take alt whenever m_x_o >= keep.
        keep, m_x_o, less_than_x_o = bb.add(
            LinearDepthGreaterThan(self.b_k1), a=keep, b=m_x_o, target=less_than_x_o
        )
        less_than_x_o = bb.add(XGate(), q=less_than_x_o)
        less_than_x_o, alt, x_o = bb.add(
            CSwap(self.sel_bitsize), ctrl=less_than_x_o, x=alt, y=x_o
        )
        return {
            'x_o': x_o, 'succ_x_o': succ_x_o, 'sigma_1': sigma_1, 'm_x_o': m_x_o,
            'less_than_x_o': less_than_x_o, 'alt_x_o': alt, 'keep_x_o': keep, **out,
            'phase_grad': phase_grad,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """``Uniform`` + flag + load + test + cswap (steps 1-4); the Hadamards are Clifford."""
        return {
            self.uniform: 1,                                   # (1) and succ_x_o
            OnEach(self.b_k1, Hadamard()): 1,                    # (1) m_x_o
            Hadamard(): 1,                                     # (1) sigma_1
            self.qroam: 1,                                     # (2)
            LinearDepthGreaterThan(self.b_k1): 1,                # (3) b_k1
            XGate(): 1,                                        #     keep <= m
            CSwap(self.sel_bitsize): 1,                        # (4) ceil lg(N+RC)
        }

    @classmethod
    def from_weights(
        cls, weights, b_k1: Optional[int] = None, log_block_size: Optional[int] = None, *,
        precision: Optional[float] = None, **kwargs,
    ) -> "OuterPrepare":
        r"""Build from the ``N + RC`` outer weights, the ``RC`` SF ones first, to ``b_k1`` bits or
        to the bits a ``precision`` needs (:func:`keep_bits`)."""
        w = np.asarray(weights, float)
        if w.ndim != 1:
            raise ValueError(f"weights must be 1D; got shape {w.shape}")
        b_k1 = keep_bits(b_k1, precision, len(w))
        alt, keep = alias_table(w, b_k1)
        return cls(alt=alt, keep=keep, b_k1=b_k1, log_block_size=log_block_size, **kwargs)

    @classmethod
    def symbolic(
        cls, num_x: 'SymbolicInt', b_k1: 'SymbolicInt', log_block_size: 'SymbolicInt',
        erase_log_block_size: 'SymbolicInt', s: 'SymbolicInt' = 7,
        phase_grad_bitsize: Optional['SymbolicInt'] = None,
    ) -> "OuterPrepare":
        r"""A costing-only placeholder at symbolic ``N + RC``, ``b_k1``, ``k_1`` and ``k_e1``."""
        _check_symbolic_block_size(log_block_size)
        _check_symbolic_block_size(erase_log_block_size)
        shape = Shaped((num_x,))
        return cls(
            alt=shape, keep=shape, b_k1=b_k1, log_block_size=log_block_size,
            erase_log_block_size=erase_log_block_size, s=s,
            phase_grad_bitsize=s if phase_grad_bitsize is None else phase_grad_bitsize,
        )


# --- steps 5-8: the inner sum over b ------------------------------------------------


@frozen
class InnerPrepare(_AliasPrepare):
    r"""Inner PREPARE: steps 5-8 of the ``README.md`` circuit::

        (5) Uniform(b) over B+1 values, flagged; |+> on sigma_0 and on m_b      (Clifford)
        (6) unary(x_o) x QROAM(b) -> alt, keep, sgn, d_sgn, r
        (7) keep <= m_b -> less_than_b
        (8) cswap alt -> b

    The tables are keyed by ``(x_o, b)``, so one unary iteration over ``x_o`` both drives the
    load and writes ``r = x_o / C``. The ``N`` D1/Q1 rows are all zero: those operators have no
    ``b`` sum, and ``Uniform(b)`` runs unconditionally anyway because ``SEL`` is
    ``b``-independent there, so the inverse cleans up whatever it prepared.

    ``b = B`` is the inner LCU's identity term; it is prepared like any other value, and it is
    :class:`~dfthcbe.select.InnerSelect`'s ``b != B`` flag that makes it act as the identity.
    The loaded signs are applied by :class:`InnerSigns`.

    Args:
        alt: ``(RC, B+1)`` alternate ``b`` indices, keyed ``x_o = rC + c``.
        keep: ``(RC, B+1)`` keep numerators, ``b_k2`` bits each.
        sgn: ``(RC, B+1)`` sign of ``w_b``; ``d_sgn``: sign of ``w_b`` vs ``w_alt``.
        num_orb: ``N`` -- the D1/Q1 tail, so ``x_o`` spans ``N + RC``.
        num_c: ``C`` -- copies per rank, so ``r = x_o / C`` on the SF rows.
        b_k2: bits of the keep comparison.
        log_block_size: ``k_2``, applied to the ``b`` axis only; ``None`` takes the cheapest read.
        exact_erase: as for :class:`OuterPrepare`.
        erase_log_block_size: ``k_e2``, as for :class:`OuterPrepare`; the one-hot also holds all of
            ``b``, and the erase skips the zero D1/Q1 rows.
        s, phase_grad_bitsize: as for :class:`OuterPrepare`, for ``Uniform(b)``.

    Registers:
        x_o: the outer index, read-only here: it addresses the load.
        b: the inner index, ``B+1`` values.
        succ_b: flags that ``Uniform(b)`` landed in range; gates ``Maj``.
        sigma_0: the inner spin qubit, left in ``|+>`` -- also :class:`~dfthcbe.majorana.Maj`'s
            flavour ancilla.
        m_b: the ``b_k2`` uniform bits ``keep`` is compared against.
        less_than_b: the comparison result, left dirty for the adjoint to clear.
        alt_b, keep_b, sgn, d_sgn, r: the loaded columns (and ``junk_*``, the swap leftovers).
        phase_grad: the phase gradient of ``Uniform(b)``'s rotations, only catalysed.
    """

    alt: Union[Tuple[Tuple[int, ...], ...], Shaped] = field(
        converter=lambda d: freeze_table(d, 2, 'alt')
    )
    keep: Union[Tuple[Tuple[int, ...], ...], Shaped] = field(
        converter=lambda d: freeze_table(d, 2, 'keep')
    )
    sgn: Union[Tuple[Tuple[int, ...], ...], Shaped] = field(
        converter=lambda d: freeze_table(d, 2, 'sgn')
    )
    d_sgn: Union[Tuple[Tuple[int, ...], ...], Shaped] = field(
        converter=lambda d: freeze_table(d, 2, 'd_sgn')
    )
    num_orb: 'SymbolicInt'
    num_c: 'SymbolicInt'
    b_k2: 'SymbolicInt'
    log_block_size: Optional['SymbolicInt'] = None
    exact_erase: bool = False
    erase_log_block_size: Optional['SymbolicInt'] = None
    s: 'SymbolicInt' = 7
    phase_grad_bitsize: 'SymbolicInt' = field(
        default=attrs.Factory(lambda self: self.s, takes_self=True)
    )

    def __attrs_post_init__(self):
        if is_symbolic(self.alt):
            return
        shapes = {np.shape(t) for t in (self.alt, self.keep, self.sgn, self.d_sgn)}
        if len(shapes) != 1:
            raise ValueError(f"the four columns must have one shape; got {shapes}.")
        if self.num_sf % self.num_c:
            raise ValueError(f"RC={self.num_sf} must be a multiple of C={self.num_c}.")
        if self.num_r < 2 or self.num_b < 1:
            raise ValueError(f"need R>=2 and B>=1; got R={self.num_r}, B={self.num_b}.")
        if max(max(row) for row in self.keep) >= 2**self.b_k2:
            raise ValueError(f"keep values must fit in b_k2={self.b_k2} bits.")

    # --- shape helpers -----------------------------------------------------------

    @property
    def num_sf(self) -> 'SymbolicInt':
        """``RC`` -- the SF prefix of ``x_o``."""
        return self.alt.shape[0] if is_symbolic(self.alt) else len(self.alt)

    @property
    def num_b(self) -> 'SymbolicInt':
        """``B``; the register spans ``B+1``, the last value being the identity term."""
        n = self.alt.shape[1] if is_symbolic(self.alt) else len(self.alt[0])
        return n - 1

    @property
    def num_r(self) -> 'SymbolicInt':
        """``R = RC / C`` -- the DFTHC rank this load writes out."""
        return self.num_sf // self.num_c

    @property
    def num_x(self) -> 'SymbolicInt':
        """``N + RC`` -- SF first, then the D1/Q1 tail."""
        return self.num_orb + self.num_sf

    @property
    def sel_bitsize(self) -> 'SymbolicInt':
        return bit_length(self.num_x - 1)

    @property
    def b_bitsize(self) -> 'SymbolicInt':
        return bit_length(self.num_b)

    @property
    def r_bitsize(self) -> 'SymbolicInt':
        return bit_length(self.num_r - 1)

    @property
    def data_names(self) -> Tuple[str, ...]:
        return ('alt_b', 'keep_b', 'sgn', 'd_sgn', 'r')

    @cached_property
    def uniform(self) -> UniformViaPhaseGradient:
        """(5) ``Uniform(b)`` over ``B + 1`` values and ``succ_b``."""
        return UniformViaPhaseGradient(
            self.num_b + 1, s=self.s, phase_grad_bitsize=self.phase_grad_bitsize
        )

    def failure_signs(self) -> np.ndarray:
        """Per SF row, the mean sign steps 6-9 give a failed ``Uniform(b)``, which lands evenly on
        ``b >= B + 1``: the read loads the entry of the leaf that ``b`` reaches, zeros past the
        table."""
        n, k = self.b_bitsize, self.qroam.log_block_sizes[1]
        failed = range(self.num_b + 1, 2**n)
        if not failed:
            return np.ones(self.num_sf)
        kept = np.array(self.keep) / 2**self.b_k2  # the m_b that keep b; the rest take alt
        signs = (1 - 2 * np.array(self.sgn)) * (kept + (1 - kept) * (1 - 2 * np.array(self.d_sgn)))
        blocks = -(-(self.num_b + 1) // 2**k)
        loaded = [_leaf(b >> k, n - k, blocks) * 2**k + b % 2**k for b in failed]
        return np.mean(
            [signs[:, e] if e <= self.num_b else np.ones(self.num_sf) for e in loaded], axis=0
        )

    def _read(self, k_2: 'SymbolicInt') -> QROAMCleanHetero:
        """(6) The ``(x_o, b)`` read, in blocks along ``b`` only: ``RC Bhat`` live leaves,
        ``Bhat = ceil((B+1)/2**k_2)``.

        ``r`` is the same for every ``b``, so it is flat: written by the walk over ``x_o``, it
        skips the swap. The ``N`` tail rows are zero, so ``_break_early`` prunes them to their
        ``beta(RC, N+RC)`` aligned blocks of ``x_o``, each a constant row of ``s(Bhat)`` leaves.
        The walk is ``RC Bhat + beta s(Bhat) - 2`` ANDs.
        """
        bitsizes = (self.b_bitsize, self.b_k2, 1, 1, self.r_bitsize)  # x_o one at a time: r is flat
        if is_symbolic(self.alt):
            qroam = QROAMCleanHetero.build_from_bitsize(
                (self.num_x, self.num_b + 1), bitsizes, log_block_sizes=(0, k_2)
            )
            b_hat = qroam.batched_qrom_shape[1]
            dead = aligned_blocks(self.num_sf, self.num_x) * popcount(b_hat)
            qroam = attrs.evolve(qroam, num_leaves=self.num_sf * b_hat + dead)
        else:
            x_o = np.arange(self.num_x)
            r = np.where(x_o < self.num_sf, x_o // self.num_c, 0)  # r = x_o // C on the SF rows
            tails = ((0, self.num_orb), (0, 0))  # the D1/Q1 rows load nothing
            data = [np.pad(t, tails) for t in (self.alt, self.keep, self.sgn, self.d_sgn)]
            data.append(np.repeat(r[:, None], self.num_b + 1, axis=1))  # the same r for every b
            qroam = QROAMCleanHetero.build_from_data(
                *data, target_bitsizes=bitsizes, log_block_sizes=(0, k_2)
            )
        return attrs.evolve(qroam.with_flat_target_idxs((4,)), num_live_rows=self.num_sf)

    @property
    def _max_log_block_size(self) -> int:
        return int(self.num_b + 1).bit_length() - 1

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x_o', BQUInt(self.sel_bitsize, self.num_x)),
                Register('b', BQUInt(self.b_bitsize, self.num_b + 1)),
                Register('succ_b', QBit()),
                Register('sigma_0', QBit()),
                Register('m_b', QAny(self.b_k2)),
                Register('less_than_b', QBit()),
                *self.data_registers,
                Register('phase_grad', QFxp(self.phase_grad_bitsize, self.phase_grad_bitsize)),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, x_o: SoquetT, b: SoquetT, succ_b: SoquetT, sigma_0: SoquetT,
        m_b: SoquetT, less_than_b: SoquetT, phase_grad: SoquetT,
    ) -> Dict[str, SoquetT]:
        if is_symbolic(self.alt):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")
        # (5) Uniform(b) including b = B, flagged; the inner spin; and the uniform comparison
        # register.
        b, succ_b, phase_grad = bb.add(self.uniform, target=b, succ=succ_b, phase_grad=phase_grad)
        sigma_0 = bb.add(Hadamard(), q=sigma_0)
        m_b = bb.add(OnEach(self.b_k2, Hadamard()), q=m_b)
        # (6) the (x_o, b) load.
        out = self.add_qroam(bb, selection0=x_o, selection1=b)
        x_o, b = out.pop('selection0'), out.pop('selection1')
        alt, keep = out.pop('alt_b'), out.pop('keep_b')
        sgn, d_sgn = out.pop('sgn'), out.pop('d_sgn')
        # (7, 8) alias sampling proper: take alt whenever m_b >= keep.
        keep, m_b, less_than_b = bb.add(
            LinearDepthGreaterThan(self.b_k2), a=keep, b=m_b, target=less_than_b
        )
        less_than_b = bb.add(XGate(), q=less_than_b)
        less_than_b, alt, b = bb.add(CSwap(self.b_bitsize), ctrl=less_than_b, x=alt, y=b)
        return {
            'x_o': x_o, 'b': b, 'succ_b': succ_b, 'sigma_0': sigma_0, 'm_b': m_b,
            'less_than_b': less_than_b, 'alt_b': alt, 'keep_b': keep, 'sgn': sgn,
            'd_sgn': d_sgn, **out, 'phase_grad': phase_grad,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """``Uniform`` + flag + load + test + cswap (steps 5-8); the Hadamards are Clifford."""
        return {
            self.uniform: 1,                           # (5) and succ_b
            Hadamard(): 1,                             # (5) sigma_0
            OnEach(self.b_k2, Hadamard()): 1,            # (5) m_b
            self.qroam: 1,                             # (6)
            LinearDepthGreaterThan(self.b_k2): 1,        # (7) b_k2
            XGate(): 1,                                #     keep <= m
            CSwap(self.b_bitsize): 1,                  # (8) ceil lg(B+1)
        }

    @classmethod
    def from_coeffs(
        cls, w_sf, num_orb: int, b_k2: Optional[int] = None, log_block_size: Optional[int] = None,
        *, precision: Optional[float] = None, **kwargs,
    ) -> "InnerPrepare":
        r"""Build from the ``(R, C, B+1)`` signed SF coefficients ``w_b^(rc)``, to ``b_k2`` bits or
        to the bits a ``precision`` needs over ``B + 1`` values (:func:`keep_bits`).

        Row ``(r, c)`` is the inner LCU over ``b`` for ``O_SF,rc``; its ``b = B`` entry is the
        identity term. Only the weights matter for the amplitudes and only their signs for the
        phases, so the caller's normalization is irrelevant here (it is PREPARE's ``lambda``).
        """
        w = np.asarray(w_sf, float)
        if w.ndim != 3:
            raise ValueError(f"w_sf must be 3D (R, C, B+1); got shape {w.shape}")
        b_k2 = keep_bits(b_k2, precision, w.shape[2])
        rows = [signed_alias_table(row, b_k2) for row in w.reshape(-1, w.shape[-1])]
        alt, keep, sgn, d_sgn = (np.array(col) for col in zip(*rows))
        return cls(
            alt=alt, keep=keep, sgn=sgn, d_sgn=d_sgn,
            num_orb=num_orb, num_c=w.shape[1], b_k2=b_k2, log_block_size=log_block_size, **kwargs,
        )

    @classmethod
    def symbolic(
        cls, num_orb: 'SymbolicInt', num_r: 'SymbolicInt', num_b: 'SymbolicInt',
        num_c: 'SymbolicInt', b_k2: 'SymbolicInt', log_block_size: 'SymbolicInt',
        erase_log_block_size: 'SymbolicInt', s: 'SymbolicInt' = 7,
        phase_grad_bitsize: Optional['SymbolicInt'] = None,
    ) -> "InnerPrepare":
        r"""A costing-only placeholder at symbolic ``N``, ``R``, ``B``, ``C``, ``b_k2``, ``k_2``,
        ``k_e2``."""
        _check_symbolic_block_size(log_block_size)
        _check_symbolic_block_size(erase_log_block_size)
        shape = Shaped((num_r * num_c, num_b + 1))
        return cls(
            alt=shape, keep=shape, sgn=shape, d_sgn=shape, num_orb=num_orb, num_c=num_c, b_k2=b_k2,
            log_block_size=log_block_size, erase_log_block_size=erase_log_block_size, s=s,
            phase_grad_bitsize=s if phase_grad_bitsize is None else phase_grad_bitsize,
        )


# --- step 9: the inner signs --------------------------------------------------------


@frozen
class InnerSigns(Bloq):
    r"""Step 9: ``Z(sgn) CZ(less_than_b, d_sgn)`` gives each branch the sign of its ``w_b``."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(sgn=1, d_sgn=1, less_than_b=1)

    def build_composite_bloq(self, bb: BloqBuilder, sgn, d_sgn, less_than_b) -> Dict[str, SoquetT]:
        less_than_b, d_sgn = bb.add(CZ(), q1=less_than_b, q2=d_sgn)
        return {'sgn': bb.add(ZGate(), q=sgn), 'd_sgn': d_sgn, 'less_than_b': less_than_b}


# --- steps 1 and 5: the equal superposition ----------------------------------------


class v2(sympy.Function):
    r"""``v_2(L)``, the power of two in ``L``: ``L = 2^v_2(L) l``, ``l`` odd; evaluates on ints."""

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_Integer and n > 0:
            return sympy.Integer((int(n) & -int(n)).bit_length() - 1)


@attrs.frozen
class UniformViaPhaseGradient(PrepareUniformSuperposition):
    r"""Qualtran's ``PrepareUniformSuperposition`` with its two ``Rz`` on a phase gradient, and its
    success flag.

    For ``L = 2^eta l``, ``l`` odd, Hadamards give the ``2^eta`` and one round of amplitude
    amplification the ``l``, on the top ``ceil lg l`` qubits. Its two rotations are each a
    ``ZPowConstViaPhaseGradient`` on the top ``s`` qubits of ``phase_grad``, ``s - 2`` Toffolis.
    Their rounding moves a little amplitude to ``x >= L`` and leaves ``[0, L)`` exactly uniform;
    ``succ = [x < L]`` flags it, comparing just the top qubits with ``l``.

    Args:
        n: ``L``, the number of values.
        s: bits of the two rotations.
        phase_grad_bitsize: width of ``phase_grad``, at least ``s``; by default ``s``.

    Registers:
        target: the prepared register.
        succ: ``x < L``, XORed in.
        phase_grad: a phase-gradient state, only catalysed.
    """

    s: 'SymbolicInt' = 7
    phase_grad_bitsize: 'SymbolicInt' = field(
        default=attrs.Factory(lambda self: self.s, takes_self=True)
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.s, self.phase_grad_bitsize) and self.s > self.phase_grad_bitsize:
            raise ValueError(f"s={self.s} needs a phase gradient of at least s bits.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *Signature.build(ctrl=slen(self.cvs), target=bit_length(self.n - 1)),
                Register('succ', QBit()),
                Register('phase_grad', QFxp(self.phase_grad_bitsize, self.phase_grad_bitsize)),
            ]
        )

    def k_l_logL(self) -> Tuple['SymbolicInt', 'SymbolicInt', 'SymbolicInt']:
        """``(eta, l, logL)``; symbolically too, where Qualtran takes every ``L`` to be odd."""
        if not is_symbolic(self.n):
            return super().k_l_logL()
        eta = v2(self.n)
        l = self.n / 2**eta
        return eta, l, ceil(log2(l))

    def _rotation(self, l) -> ZPowConstViaPhaseGradient:
        """Qualtran's ``Rz(theta)``, as ``Z^(theta / pi)``: the same up to a global phase."""
        theta = acos(1 - (2 ** floor(log2(l))) / l)
        return ZPowConstViaPhaseGradient(theta / (sympy.pi if is_symbolic(l) else np.pi), self.s)

    def failure_probability(self) -> float:
        """``P(succ = 0)``: the amplitude amplification at the angle ``s`` bits give, in numpy."""
        _, l, logL = self.k_l_logL()
        if not logL:
            return 0.0
        theta = np.arccos(1 - 2 ** np.floor(np.log2(l)) / l)
        k = QFxp(self.s, self.s).to_fixed_width_int(theta / (2 * np.pi))  # what the rotation loads
        phase, n = np.exp(2j * np.pi * k / 2**self.s), 2**logL
        good, h = np.arange(n) < l, hadamard(n) / np.sqrt(n)
        psi = h @ (np.where(good, phase, 1) / np.sqrt(n))  # the phase on x < l, then H
        psi[0] *= phase  # the phase on 0
        return float(np.sum(np.abs(h @ psi)[~good] ** 2))

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        controls, target, succ = quregs.get('ctrl', ()), quregs['target'], quregs['succ']
        phase_grad = quregs['phase_grad'][: self.s]  # its top s qubits: an s-bit phase gradient
        k, l, logL = self.k_l_logL()
        logL_qubits = target[:logL]

        yield [
            op.controlled_by(*controls, control_values=self.concrete_cvs)
            for op in cirq.H.on_each(*target)
        ]
        if not logL:  # L = 2^eta: the Hadamards are exact, every x is in range
            yield cirq.X(*succ)
            return

        rotation = self._rotation(l)
        ancilla = context.qubit_manager.qalloc(1)
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=ancilla)
        yield rotation.on_registers(q=ancilla, phase_grad=phase_grad)
        less_than = LessThanConstant(logL, l).on_registers(x=logL_qubits, target=ancilla)
        yield less_than**-1  # type: ignore[operator]
        context.qubit_manager.qfree(ancilla)

        yield cirq.H.on_each(*logL_qubits)

        and_ancilla = context.qubit_manager.qalloc(len(self.concrete_cvs) + logL - 2)
        and_target = context.qubit_manager.qalloc(1)
        and_cv = (0,) * logL + self.concrete_cvs
        ctrl = np.asarray([*logL_qubits, *controls])[:, np.newaxis]
        junk = np.asarray(and_ancilla)[:, np.newaxis]
        if len(and_cv) <= 2:
            and_op = And(and_cv[0], and_cv[1]).on_registers(ctrl=ctrl, target=and_target)
        else:
            and_op = MultiAnd(cvs=and_cv).on_registers(ctrl=ctrl, junk=junk, target=and_target)

        yield and_op
        yield rotation.on_registers(q=and_target, phase_grad=phase_grad)
        yield cirq.inverse(and_op)

        yield cirq.H.on_each(*logL_qubits)
        context.qubit_manager.qfree([*and_target, *and_ancilla])
        # x < L exactly when its top logL qubits are below l
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=succ)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """Qualtran's symbolic count with the two rotations and the flag, and none of the
        amplitude amplification when ``L`` is a power of two."""
        if not is_symbolic(self.n, self.cvs):
            return super().build_call_graph(ssa)
        _, l, logL = self.k_l_logL()
        aa = sympy.Min(logL, 1)  # 0 for L = 2^eta, where the Hadamards are exact
        return {  # every term carries aa, so the total factors as aa * (4 logL + 2 s - 5)
            Hadamard(): 3 * logL,
            LessThanConstant(logL, l): 2 * aa,  # the amplification's, and succ
            LessThanConstant(logL, l).adjoint(): aa,
            self._rotation(l): 2 * aa,
            MultiAnd(HasLength(logL)): aa,
            MultiAnd(HasLength(logL)).adjoint(): aa,
        }
