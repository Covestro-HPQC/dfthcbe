r"""The DFTHC block encoding of ``H_SA / Lambda - I``, and its qubitization walk operator."""

from __future__ import annotations

from functools import cached_property
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, DecomposeTypeError, QBit, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import GlobalPhase, ZGate
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.symbolics import is_symbolic

from dfthcbe.prepare import alias_probabilities, InnerPrepare, InnerSigns, OuterPrepare
from dfthcbe.rot_data import vector_from_angles
from dfthcbe.select import InnerSelect
from dfthcbe.utils import walk_inputs

if TYPE_CHECKING:
    from qualtran import AddControlledT, CtrlSpec
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt

OUTER_ANCILLAS = ('x_o', 'm_x_o', 'sigma_1')  # a
INNER_ANCILLAS = ('b', 'm_b', 'sigma_0')      # B
# Computed from the ancillas by each PREPARE and uncomputed by its adjoint.
FLAGS = ('succ_x_o', 'less_than_x_o', 'succ_b', 'less_than_b')


@frozen
class DFTHCBlockEncoding(Bloq):
    r"""``BE[H_SA / Lambda - I] = U^dag REF_T2 U``: steps 1-27 of the ``README.md`` circuit::

        (1-4)    outer PREP
        (5-22)   inner PREP -> signs -> InnerSelect -> inner PREP^dag        BE[O]
        (23)     REF_T2, the reflection about |0>_B |1>_succ_x_o
        (5-22)   inner PREP -> InnerSelect^dag -> signs -> inner PREP^dag    BE[O]^dag
        (24-27)  outer PREP^dag

    The second ``BE[O]`` needs ``InnerSelect.adjoint()`` because ``Maj`` is not Hermitian.
    Controlled, only ``REF_T2`` takes the control: on ``ctrl = 0`` the rest is ``U^dag U = I``.

    Build it from a decomposed Hamiltonian with :meth:`from_factors`, from the three sub-bloqs, or
    as a :meth:`symbolic` placeholder.

    Args:
        outer: steps 1-4.
        inner: steps 5-8.
        select: steps 10-18.
        Lambda: ``Lambda`` of ``<0|BE|0> = H_SA / Lambda - I``: Eq. 33's over
            ``1 - P(succ_x_o = 0)``, as ``REF_T2`` is ``-I`` where ``Uniform(x_o)`` failed. The
            tables only hold ratios and unit vectors, so it is stored: :meth:`from_factors` sets it,
            else ``None``.
        is_controlled: whether a ``ctrl`` qubit is present. Set by ``.controlled()``.

    Registers:
        ctrl: the control, if ``is_controlled``.
        x_o, m_x_o, sigma_1: the outer ancillas ``a``.
        b, m_b, sigma_0: the inner ancillas ``B``.
        psi_up, psi_dn: the two ``N``-qubit system registers.
        phase_grad: the shared phase-gradient catalyst.
    """

    outer: OuterPrepare
    inner: InnerPrepare
    select: InnerSelect
    Lambda: Optional[float] = None
    is_controlled: bool = False

    def __attrs_post_init__(self):
        o, i, s = self.outer, self.inner, self.select
        sizes = {
            'N + RC': (o.num_x, i.num_x), 'N': (i.num_orb, s.num_orb), 'B': (i.num_b, s.num_b),
            'R': (i.num_r, s.num_r), 'C': (i.num_c, s.num_c),
            'b_rot, the outer phase gradient': (o.phase_grad_bitsize, s.b_rot),
            'b_rot, the inner phase gradient': (i.phase_grad_bitsize, s.b_rot),
        }
        if is_symbolic(*(v for pair in sizes.values() for v in pair)):
            return
        for name, (a, b) in sizes.items():
            if a != b:
                raise ValueError(f"the sub-bloqs disagree on {name}: {a} vs {b}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            ([Register('ctrl', QBit())] if self.is_controlled else [])
            + [self.outer.signature.get_left(name) for name in OUTER_ANCILLAS]
            + [self.inner.signature.get_left(name) for name in INNER_ANCILLAS]
            + [self.select.signature.get_left(name) for name in ('psi_up', 'psi_dn', 'phase_grad')]
        )

    @cached_property
    def ref_t2(self) -> Bloq:
        """(23) ``REF_T2``'s multi-controlled ``Z``: ``-1`` on ``|0>_B`` with ``succ_x_o = 1``
        (and ``ctrl = 1``), the gate B11 counts."""
        reflection = _reflection(self.signature, INNER_ANCILLAS, control_val=1, global_phase=1)
        return reflection.controlled() if self.is_controlled else reflection

    @cached_property
    def ref_t2_phase(self) -> Bloq:
        """(23) ``REF_T2``'s ``-1``, a global phase, a ``Z`` on ``ctrl`` when controlled."""
        return ZGate() if self.is_controlled else GlobalPhase(exponent=1)

    def _add_ref_t2(self, bb: BloqBuilder, soqs: Dict[str, SoquetT]) -> None:
        """(23) ``REF_T2``, the reflection about ``|0>_B |1>_succ_x_o``: ``2|0><0|_B - I`` where
        ``Uniform(x_o)`` worked, giving ``T_2(O^dag, O)``, and ``-I`` where it failed."""
        controls = ('ctrl', 'succ_x_o') if self.is_controlled else ('succ_x_o',)
        _add_reflection(bb, soqs, self.ref_t2, INNER_ANCILLAS, controls)
        if self.is_controlled:
            soqs['ctrl'] = bb.add(self.ref_t2_phase, q=soqs['ctrl'])
        else:
            bb.add(self.ref_t2_phase)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.select.angle_data):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")
        soqs |= {name: bb.allocate(1) for name in FLAGS}

        # (1-4) the outer sum over x_o; its alias garbage rides along to steps 24-26.
        _add(bb, soqs, self.outer)
        # (5-22) BE[O] = PREP_b^dag SEL D PREP_b, with D the signs of step 9.
        for bloq in (self.inner, InnerSigns(), self.select, self.inner.adjoint()):
            _add(bb, soqs, bloq)
        # (23) REF_T2: one step of oblivious amplitude amplification, giving T_2(O^dag, O).
        self._add_ref_t2(bb, soqs)
        # (5-22) BE[O]^dag = PREP_b^dag D SEL^dag PREP_b.
        for bloq in (self.inner, self.select.adjoint(), InnerSigns(), self.inner.adjoint()):
            _add(bb, soqs, bloq)
        # (24-27) the outer unprepare.
        _add(bb, soqs, self.outer.adjoint())
        for name in FLAGS:
            bb.free(soqs.pop(name))
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """Steps 1-27: the preps and their adjoints, the signs, both SELECTs and ``REF_T2``."""
        return {
            self.outer: 1,              # (1-4)
            self.inner: 2,              # (5-8), once per BE[O]
            InnerSigns(): 2,            # (9)
            self.select: 1,             # (10-18)
            self.inner.adjoint(): 2,    # (19-22)
            self.ref_t2: 1,             # (23)
            self.ref_t2_phase: 1,       #      its -1 (Clifford)
            self.select.adjoint(): 1,   # (10-18) of BE[O]^dag
            self.outer.adjoint(): 1,    # (24-27)
        }

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=attrs.evolve(self, is_controlled=True), ctrl_reg_name='ctrl',
        )

    @classmethod
    def from_factors(
        cls, U, W, w_B, w_plus, u_plus, w_minus, u_minus, *, b_rot: int,
        b_k1: Optional[int] = None, b_k2: Optional[int] = None,
        precision_outer: Optional[float] = None, precision_inner: Optional[float] = None,
        k_1: Optional[int] = None, k_2: Optional[int] = None, k_e1: Optional[int] = None,
        k_e2: Optional[int] = None, s: int = 7, exact_erase: bool = False,
    ) -> "DFTHCBlockEncoding":
        r"""Build from a decomposed Hamiltonian, the arrays of its ``factors.npz``.

        Each PREPARE takes its bits, ``b_k1`` and ``b_k2``, or a precision from which they follow
        (:func:`~dfthcbe.prepare.keep_bits`); ``k_1, k_2`` are the QROAM block sizes and
        ``k_e1, k_e2`` those of their erases, each by default the cheapest; ``s`` the bits of the
        ``Uniform`` rotations.
        """
        outer, inner = walk_inputs(W, w_B, w_plus, w_minus)
        uniform = dict(s=s, phase_grad_bitsize=b_rot, exact_erase=exact_erase)
        outer_prep = OuterPrepare.from_weights(
            outer, b_k1, k_1, precision=precision_outer, erase_log_block_size=k_e1, **uniform
        )
        return cls(
            outer=outer_prep,
            inner=InnerPrepare.from_coeffs(
                inner, np.shape(U)[2], b_k2, k_2, precision=precision_inner,
                erase_log_block_size=k_e2, **uniform,
            ),
            select=InnerSelect.from_vectors(
                U, u_plus, u_minus, num_c=np.shape(W)[2], b_rot=b_rot, exact_erase=exact_erase
            ),
            Lambda=float(outer.sum() / 2 / (1 - outer_prep.uniform.failure_probability())),
        )

    def encoded_factors(self, exact_prep: bool = False) -> Dict[str, np.ndarray]:
        r"""The decomposed Hamiltonian encoded here, its alias and angle rounding included.

        Keyed as ``factors.npz``, so :func:`~dfthcbe.utils.walk_integrals` and
        :func:`~dfthcbe.utils.walk_e_sos` of it give ``H_SA`` of ``<0|BE|0> = H_SA / Lambda - I``.
        With ``p``, ``q`` the prepared outer and inner probabilities, ``w_b = sign(w_b) q_b
        sqrt(4 Lambda p_rc)`` and ``w_pm = Lambda p``, with Eq. 33's ``Lambda``,
        ``self.Lambda (1 - P(succ_x_o = 0))``; the unit vectors come from the loaded angles. A
        failed ``Uniform(b)`` leaves an SF row's ``Maj`` off but not its sign: each ``q_b`` carries
        ``1 - P(succ_b = 0)``, and ``w_B`` gains ``P(succ_b = 0)`` times
        :meth:`~dfthcbe.prepare.InnerPrepare.failure_signs`.

        Args:
            exact_prep: leave the failed ``Uniform(b)`` out, as with exact equal superpositions;
                then only the rounding of the tables and angles is left.
        """
        if self.Lambda is None:
            raise ValueError("encoded_factors needs Lambda, which from_factors sets.")
        o, i, s = self.outer, self.inner, self.select
        num_sf, num_b, num_pos = s.num_sf, s.num_b, s.num_pos
        lam = self.Lambda * (1 - o.uniform.failure_probability())
        p = alias_probabilities(o.alt, o.keep, o.b_k1)
        q = np.array([alias_probabilities(alt, keep, i.b_k2) for alt, keep in zip(i.alt, i.keep)])
        fail = 0.0 if exact_prep else i.uniform.failure_probability()
        a = (1 - fail) * (1 - 2 * np.array(i.sgn)) * q
        if fail:
            a[:, num_b] += fail * i.failure_signs()
        w = a * np.sqrt(4 * lam * p[:num_sf])[:, None]
        w = w.reshape(s.num_r, s.num_c, num_b + 1)
        w_tail = lam * p[num_sf:]
        u_tail = np.array([vector_from_angles(row, s.b_rot) for row in s.tail_array])
        return {
            'U': np.array([[vector_from_angles(row, s.b_rot) for row in plane]
                           for plane in s.angle_array]),
            'W': w[..., :num_b].transpose(0, 2, 1),
            'w_B': w[..., num_b],
            'w_plus': w_tail[:num_pos], 'u_plus': u_tail[:num_pos],
            'w_minus': w_tail[num_pos:], 'u_minus': u_tail[num_pos:],
        }

    @classmethod
    def symbolic(
        cls, num_orb: 'SymbolicInt', num_r: 'SymbolicInt', num_b: 'SymbolicInt',
        num_c: 'SymbolicInt', b_rot: 'SymbolicInt', b_k1: 'SymbolicInt', b_k2: 'SymbolicInt',
        k_1: 'SymbolicInt', k_2: 'SymbolicInt', k_e1: 'SymbolicInt', k_e2: 'SymbolicInt',
        s: 'SymbolicInt' = 7,
    ) -> "DFTHCBlockEncoding":
        r"""A costing-only placeholder: ``b_k1, b_k2`` are the bits of the two ``keep``s;
        ``k_1, k_2`` the QROAM block sizes, ``k_e1, k_e2`` those of their erases; ``s`` the bits of
        the ``Uniform`` rotations."""
        uniform = dict(s=s, phase_grad_bitsize=b_rot)
        return cls(
            outer=OuterPrepare.symbolic(num_orb + num_r * num_c, b_k1, k_1, k_e1, **uniform),
            inner=InnerPrepare.symbolic(num_orb, num_r, num_b, num_c, b_k2, k_2, k_e2, **uniform),
            select=InnerSelect.symbolic(num_orb, num_r, num_b, num_c, b_rot),
        )


@frozen
class DFTHCWalk(Bloq):
    r"""The qubitization walk ``W = REF_aB . BE[H_SA / Lambda - I]``: steps 1-27, then ``REF_aB``
    on ``a u B`` (step 28).

    Controlled, both reflections take the control: on ``ctrl = 0`` the block encoding is the
    identity and ``REF_aB`` is skipped, so this is controlled-``W``.

    Args:
        block_encoding: steps 1-27.
        is_controlled: whether a ``ctrl`` qubit is present. Set by ``.controlled()``.

    Registers:
        Those of ``block_encoding``.
    """

    block_encoding: DFTHCBlockEncoding
    is_controlled: bool = False

    @cached_property
    def _block_encoding(self) -> DFTHCBlockEncoding:
        """Steps 1-27, controlled with the walk."""
        return attrs.evolve(self.block_encoding, is_controlled=self.is_controlled)

    @cached_property
    def signature(self) -> Signature:
        return self._block_encoding.signature

    @cached_property
    def ref_ab(self) -> ReflectionUsingPrepare:
        """(28) ``2|0><0| - I`` on ``a u B``, which makes ``W`` a qubitization walk."""
        control_val = 1 if self.is_controlled else None
        return _reflection(self.signature, OUTER_ANCILLAS + INNER_ANCILLAS, control_val)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.block_encoding.select.angle_data):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")
        _add(bb, soqs, self._block_encoding)
        controls = ('ctrl',) if self.is_controlled else ()
        _add_reflection(bb, soqs, self.ref_ab, OUTER_ANCILLAS + INNER_ANCILLAS, controls)
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self._block_encoding: 1, self.ref_ab: 1}  # (1-27), (28)

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=attrs.evolve(self, is_controlled=True), ctrl_reg_name='ctrl',
        )

    @classmethod
    def from_factors(cls, *args, **kwargs) -> "DFTHCWalk":
        """Build from factors, on :meth:`DFTHCBlockEncoding.from_factors`'s arguments."""
        return cls(DFTHCBlockEncoding.from_factors(*args, **kwargs))

    @classmethod
    def symbolic(cls, *args: 'SymbolicInt') -> "DFTHCWalk":
        """A costing-only placeholder, on :meth:`DFTHCBlockEncoding.symbolic`'s arguments."""
        return cls(DFTHCBlockEncoding.symbolic(*args))


def _reflection(
    signature: Signature, names: Sequence[str], control_val: Optional[int] = None,
    global_phase: int = -1,
) -> ReflectionUsingPrepare:
    """``2|0><0| - I`` on the named registers; with ``global_phase=1`` its multi-controlled ``Z``
    alone, ``I - 2|0><0|``."""
    bitsizes = tuple(signature.get_left(name).total_bits() for name in names)
    return ReflectionUsingPrepare.reflection_around_zero(
        bitsizes=bitsizes, control_val=control_val, global_phase=global_phase
    )


def _add(bb: BloqBuilder, soqs: Dict[str, SoquetT], bloq: Bloq) -> None:
    """Add ``bloq`` on the registers of the same name: the sub-bloqs share their names."""
    soqs |= bb.add_d(bloq, **{reg.name: soqs.pop(reg.name) for reg in bloq.signature.lefts()})


def _add_reflection(
    bb: BloqBuilder, soqs: Dict[str, SoquetT], reflection: Bloq, names: Sequence[str],
    controls: Sequence[str] = (),
) -> None:
    """Add a ``reflection_around_zero`` on the named registers (its ``reg{i}_``), controlled by
    the named qubits (its ``control``; an array once there are two)."""
    ins = {f'reg{i}_': soqs.pop(name) for i, name in enumerate(names)}
    if controls:
        ctrl = [soqs.pop(name) for name in controls]
        ins['control'] = ctrl[0] if len(ctrl) == 1 else np.array(ctrl)
    out = bb.add_d(reflection, **ins)
    if controls:
        ctrl = out.pop('control')
        soqs |= dict(zip(controls, [ctrl] if len(controls) == 1 else ctrl))
    soqs |= {name: out[f'reg{i}_'] for i, name in enumerate(names)}
