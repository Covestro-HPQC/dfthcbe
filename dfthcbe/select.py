r""":class:`InnerSelect` -- the inner SELECT of the DFTHC block encoding."""

from __future__ import annotations

from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
from attrs import field, frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    BQUInt,
    DecomposeTypeError,
    QAny,
    QBit,
    QFxp,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, XGate
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.symbolics import bit_length, is_symbolic, Shaped

from dfthcbe.majorana import Maj
from dfthcbe.rot_data import (
    angles_from_vector,
    discretize_angles,
    RotDataErase,
    sf_rotation_qrom,
    tail_rotation_qrom,
)
from dfthcbe.rotations import GivensRotationLadder
from dfthcbe.utils import freeze_table

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt


@frozen
class InnerSelect(Bloq):
    r"""The inner ``SELECT`` of the DFTHC block encoding: steps 10-18 of the ``README.md``
    circuit::

        (10) tail_qrom(x_o)           -> angles, G          D1/Q1, and the family selector
        (11) sf_qrom(b, r) ctrl G0=0  -> angles, b != B     SF
        (12) sg = sigma_{G0}                                the multiplexed spin
        (13) USwap(sg; psi_dn, psi_up)
        (14) ladder -> (15) Maj -> (16) ladder^dag   on psi_dn
        (17) undo the swap, unbuild sg
        (18) erase the angles, the flag and G

    Under SF-first ordering ``x_o < RC`` is SF and the final ``N`` values are the one-body
    D1/Q1 terms, so the D1/Q1 angles ride the *tail* of the ``x_o`` iteration. SF angles
    come from a ``(b, r)`` table.

    ``r`` is threaded through: inner PREPARE writes it on the SF branch and erases it at step
    21. ``G`` is written by inner SELECT.

    **The three spin wires.** ``sigma_0`` (inner) and ``sigma_1`` (outer) both arrive ``|+>``
    from inner/outer PREPARE and are *not* touched here; the swap control ``sg = sigma_{G0}``
    is a third, internal wire. ``sigma_0`` does double duty because the two
    jobs sit on opposite branches of ``G0``: SF sums the spin on it, D1/Q1 sums the spin
    outside (on ``sigma_1``) and uses it as :class:`Maj`'s flavour ancilla instead.

    The circuit is ``V^dag Maj V``, so its adjoint is the same circuit with ``Maj^dag``: still
    loading first and erasing last, where reversing it would run the measurement backwards.

    Args:
        angle_data: ``(R, B, N-1)`` SF angles keyed ``(r, b)`` (or a symbolic ``Shaped``). The
            ``b = B`` identity row is appended internally, so pass only the ``B`` loaded values.
        tail_data: ``(N, N-1)`` D1/Q1 angles, the ``N_+`` ``D1`` directions first.
        num_c: ``C`` -- copies per rank, so ``x_o`` spans ``N + RC`` values.
        num_pos: ``N_+`` -- how many tail rows are ``D1``; the rest are ``Q1``.
        b_rot: bits of precision per angle.
        exact_erase: run both loads backwards at step 18 instead of costing a
            measurement-based erase, keeping the bloq unitary and simulable.
        is_adjoint: apply ``Maj^dag``; set by :meth:`adjoint`.

    Registers:
        succ_x_o, succ_b: the two success flags gating ``Maj``.
        x_o: the outer index, ``N + RC`` values; addresses the tail load.
        b: the inner index, ``B+1`` values (``b = B`` is the identity term).
        r: the DFTHC rank, ``R`` values; written by inner PREPARE, read here.
        sigma_0, sigma_1: the inner/outer spin qubits, both arriving ``|+>``.
        psi_up, psi_dn: the two ``N``-qubit system registers.
        phase_grad: the shared phase-gradient catalyst.
    """

    angle_data: Union[Tuple[Tuple[Tuple[int, ...], ...], ...], Shaped] = field(
        converter=lambda d: freeze_table(d, 3, 'angle_data')
    )
    tail_data: Union[Tuple[Tuple[int, ...], ...], Shaped] = field(
        converter=lambda d: freeze_table(d, 2, 'tail_data')
    )
    num_c: 'SymbolicInt'
    num_pos: 'SymbolicInt'
    b_rot: 'SymbolicInt'
    exact_erase: bool = False
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        if is_symbolic(self.angle_data):
            return
        if self.num_r < 2 or self.num_b < 1:
            raise ValueError(f"need R>=2 and B>=1; got R={self.num_r}, B={self.num_b}.")
        if len(self.tail_data) != self.num_orb:
            raise ValueError(
                f"tail_data must have N={self.num_orb} rows; got {len(self.tail_data)}."
            )
        if self.tail_data and len(self.tail_data[0]) != self.num_orb - 1:
            raise ValueError(
                f"tail_data rows must have N-1={self.num_orb - 1} angles; "
                f"got {len(self.tail_data[0])}."
            )
        if not 0 <= self.num_pos <= self.num_orb:
            raise ValueError(f"num_pos must be in [0, N={self.num_orb}]; got {self.num_pos}.")
        max_sf = max((max(row) for plane in self.angle_data for row in plane), default=0)
        max_tail = max((max(row) for row in self.tail_data if row), default=0)
        if max(max_sf, max_tail) >= 2**self.b_rot:
            raise ValueError(
                f"angle values must fit in b_rot={self.b_rot} bits; "
                f"got max {max(max_sf, max_tail)}."
            )

    # --- shape helpers -----------------------------------------------------------

    @property
    def num_r(self) -> 'SymbolicInt':
        """``R`` -- iteration length of the ``r`` rank register."""
        return self.angle_data.shape[0] if is_symbolic(self.angle_data) else len(self.angle_data)

    @property
    def num_b(self) -> 'SymbolicInt':
        """``B`` -- number of loaded ``b`` values; the register spans ``B+1``."""
        return self.angle_data.shape[1] if is_symbolic(self.angle_data) else len(self.angle_data[0])

    @property
    def num_orb(self) -> 'SymbolicInt':
        """``N`` -- one more than the number of angles per rotation."""
        n_angles = self.angle_data.shape[2] if is_symbolic(self.angle_data) else len(self.angle_data[0][0])
        return n_angles + 1

    @property
    def num_sf(self) -> 'SymbolicInt':
        """``RC`` -- how many values of ``x_o`` are SF, i.e. precede the D1/Q1 tail."""
        return self.num_r * self.num_c

    @property
    def angle_array(self) -> np.ndarray:
        return np.array(self.angle_data, dtype=int)

    @property
    def tail_array(self) -> np.ndarray:
        """The ``(N, N-1)`` D1/Q1 angle table (``D1`` then ``Q1``)."""
        return np.array(self.tail_data, dtype=int)

    # --- the blocks --------------------------------------------------------------

    @cached_property
    def tail_qrom(self) -> QROM:
        """(10) The D1/Q1 angles + the family selector ``G``: ``00`` SF, ``10`` D1, ``11`` Q1."""
        return tail_rotation_qrom(self.tail_data, self.num_sf, self.num_pos, self.b_rot)

    @cached_property
    def sf_qrom(self) -> QROM:
        """(11) The SF angles + the ``b != B`` flag, addressed by ``(b, r)`` on ``G0 = 0``."""
        return sf_rotation_qrom(self.angle_data, self.b_rot)

    @cached_property
    def eraser(self) -> RotDataErase:
        """(18) The measurement-based erase of the angle registers, ``R + B + N + s(RC) - 5``."""
        return RotDataErase(
            num_r=self.num_r, num_b=self.num_b, num_orb=self.num_orb, num_sf=self.num_sf,
            b_rot=self.b_rot,
        )

    @cached_property
    def ladder(self) -> GivensRotationLadder:
        return GivensRotationLadder(num_orb=self.num_orb, b_rot=self.b_rot)

    @cached_property
    def maj(self) -> Bloq:
        """(15) The Majorana gate, gated on ``succ_x_o and (G0 or succ_b)`` (5 Toffolis)."""
        return Maj().adjoint() if self.is_adjoint else Maj()

    @cached_property
    def uswap(self) -> CSwap:
        """(13, 17) ``USwap``: swap the two ``N``-qubit registers on ``sg``."""
        return CSwap(bitsize=self.num_orb)

    @property
    def _angle_names(self) -> Tuple[str, ...]:
        return tuple(f'target{k}_' for k in range(self.num_orb - 1))

    @property
    def _flag_name(self) -> str:
        """The SF load's last target column, holding ``b != B``."""
        return f'target{self.num_orb - 1}_'

    @property
    def _family_names(self) -> Tuple[str, str]:
        """The tail load's last two target columns, holding ``G0`` and ``G1``."""
        return f'target{self.num_orb - 1}_', f'target{self.num_orb}_'

    @cached_property
    def signature(self) -> Signature:
        n = self.num_orb
        return Signature(
            [
                Register('succ_x_o', QBit()),
                Register('succ_b', QBit()),
                Register('x_o', BQUInt(bit_length(self.num_sf + n - 1), self.num_sf + n)),
                Register('b', BQUInt(bit_length(self.num_b), self.num_b + 1)),
                Register('r', BQUInt(bit_length(self.num_r - 1), self.num_r)),
                Register('sigma_0', QBit()),
                Register('sigma_1', QBit()),
                Register('psi_up', QAny(n)),
                Register('psi_dn', QAny(n)),
                Register('phase_grad', QFxp(self.b_rot, self.b_rot, signed=False)),
            ]
        )

    def adjoint(self) -> 'InnerSelect':
        """The same circuit with ``Maj^dag``."""
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def build_composite_bloq(
        self, bb: BloqBuilder, succ_x_o: SoquetT, succ_b: SoquetT, x_o: SoquetT, b: SoquetT,
        r: SoquetT, sigma_0: SoquetT, sigma_1: SoquetT, psi_up: SoquetT, psi_dn: SoquetT,
        phase_grad: SoquetT,
    ) -> Dict[str, SoquetT]:
        if is_symbolic(self.angle_data):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")
        names = self._angle_names
        angles = {name: bb.allocate(self.b_rot) for name in names}

        # (10) Load the D1/Q1 angles and the family selector G.
        g0_name, g1_name = self._family_names
        G0, G1 = bb.allocate(1, QBit()), bb.allocate(1, QBit())
        out = bb.add_d(self.tail_qrom, selection=x_o, **angles, **{g0_name: G0, g1_name: G1})
        x_o, G0, G1 = out.pop('selection'), out.pop(g0_name), out.pop(g1_name)
        angles = {name: out[name] for name in names}

        # (11) Load the SF angles, also write b != B flag.
        b_neq_B = bb.allocate(1, QBit())
        G0, b, r, b_neq_B, angles = self._load_sf(bb, G0, b, r, b_neq_B, angles)

        # (12) Build the multiplexed spin sg = sigma_{G0}
        sg = bb.allocate(1, QBit())
        G0, sigma_0, sigma_1, sg = self._build_sigma(bb, G0, sigma_0, sigma_1, sg)

        # (13) USwap: route the active spin register into psi_dn.
        sg, psi_dn, psi_up = bb.add(self.uswap, ctrl=sg, x=psi_dn, y=psi_up)

        # (14) Rotate
        loaded = np.array([angles[name] for name in names])
        psi_dn, loaded, phase_grad = bb.add(
            self.ladder, system=psi_dn, angles=loaded, phase_grad=phase_grad
        )
        # (15) Majorana on one spin-orbital, gated on succ_x_o and (G0 or succ_b).
        qubits = list(bb.split(psi_dn))
        succ_x_o, succ_b, G0, G1, b_neq_B, sigma_0, qubits[0] = bb.add_t(
            self.maj, succ_x_o=succ_x_o, succ_b=succ_b, G0=G0, G1=G1, b_neq_B=b_neq_B,
            flavour=sigma_0, target=qubits[0],
        )
        psi_dn = bb.join(np.array(qubits), dtype=QAny(self.num_orb))
        # (16) UNRotate.
        psi_dn, loaded, phase_grad = bb.add(
            self.ladder.adjoint(), system=psi_dn, angles=loaded, phase_grad=phase_grad
        )

        # (17) Undo the swap, then unbuild sg
        sg, psi_dn, psi_up = bb.add(self.uswap, ctrl=sg, x=psi_dn, y=psi_up)
        G0, sigma_0, sigma_1, sg = self._build_sigma(bb, G0, sigma_0, sigma_1, sg)
        bb.free(sg)

        # (18) Erase the angles, the flag and G.
        x_o, b, r = self._erase(bb, x_o, b, r, G0, G1, dict(zip(names, loaded)), b_neq_B)
        return {
            'succ_x_o': succ_x_o, 'succ_b': succ_b, 'x_o': x_o, 'b': b, 'r': r,
            'sigma_0': sigma_0, 'sigma_1': sigma_1, 'psi_up': psi_up, 'psi_dn': psi_dn,
            'phase_grad': phase_grad,
        }

    def _build_sigma(self, bb: BloqBuilder, G0, sigma_0, sigma_1, sg):
        """``sg ^= sigma_{G0}`` -- two Toffolis, and its own inverse."""
        ctrls, sg = bb.add(
            MultiControlX(cvs=(0, 1)), controls=np.array([G0, sigma_0]), target=sg
        )
        G0, sigma_0 = ctrls
        ctrls, sg = bb.add(
            MultiControlX(cvs=(1, 1)), controls=np.array([G0, sigma_1]), target=sg
        )
        G0, sigma_1 = ctrls
        return G0, sigma_0, sigma_1, sg

    def _load_sf(self, bb: BloqBuilder, G0, b, r, b_neq_B, angles):
        """One SF read. QROM controls fire on 1 only, hence the two (Clifford) X gates."""
        G0 = bb.add(XGate(), q=G0)
        out = bb.add_d(
            self.sf_qrom, control=G0, selection0=b, selection1=r, **angles,
            **{self._flag_name: b_neq_B},
        )
        G0 = bb.add(XGate(), q=out.pop('control'))
        b, r = out.pop('selection0'), out.pop('selection1')
        b_neq_B = out.pop(self._flag_name)
        return G0, b, r, b_neq_B, {name: out[name] for name in self._angle_names}

    def _erase(self, bb: BloqBuilder, x_o, b, r, G0, G1, angles, b_neq_B):
        """Undo the angle load.

        By default this is the measurement-based erase: one X-measurement frees the registers,
        then the two phase fixups of :class:`~dfthcbe.rot_data.RotDataErase`. With
        ``exact_erase`` both loads instead run backwards -- a QROM is its own
        inverse -- which keeps everything unitary at the price of a second full read. The SF
        read goes first, as it is fired on ``G0``.
        """
        if not self.exact_erase:
            out = bb.add_d(
                self.eraser, x_o=x_o, b=b, r=r,
                angles=np.array([angles[name] for name in self._angle_names]), b_neq_B=b_neq_B,
                G0=G0, G1=G1,
            )
            return out['x_o'], out['b'], out['r']
        G0, b, r, b_neq_B, angles = self._load_sf(bb, G0, b, r, b_neq_B, angles)
        g0_name, g1_name = self._family_names
        out = bb.add_d(self.tail_qrom, selection=x_o, **angles, **{g0_name: G0, g1_name: G1})
        bb.free(b_neq_B)
        for name in self._angle_names + self._family_names:
            bb.free(out[name])
        return out['selection'], b, r

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        """Two loads + the spin plumbing + 2 ladders + Maj + the erase (steps 10-18)."""
        counts = {
            self.tail_qrom: 1,      # (10) D1/Q1 angles + G, cost ~N
            self.sf_qrom: 1,        # (11) SF angles + flag, cost ~RB
            XGate(): 2,             #      firing that load on G0 = 0 (Clifford)
            MultiControlX(cvs=(0, 1)): 2,  # (12, 17) sg = sigma_{G0}, build + unbuild:
            MultiControlX(cvs=(1, 1)): 2,  #      4 Toffolis
            self.uswap: 2,          # (13, 17) 2N Toffolis
            self.ladder: 1,           # (14) the Givens ladder
            self.maj: 1,            # (15) Maj and its gating, 5
            self.ladder.adjoint(): 1,  # (16)
        }
        if not self.exact_erase:
            counts[self.eraser] = 1                  # (18) R + B + N + s(RC) - 5
        else:
            counts[self.sf_qrom] += 1                        # (18) a second full read
            counts[XGate()] += 2
            counts[self.tail_qrom] += 1
        return counts

    # --- convenience constructors --------------------------------------------------

    @classmethod
    def from_angles(
        cls, angles, tail_angles, num_c: int, num_pos: int, b_rot: int, **kwargs
    ) -> "InnerSelect":
        r"""Build from angle tables in **radians**: SF ``(R, B, N-1)``, tail ``(N, N-1)``."""
        return cls(
            angle_data=discretize_angles(angles, b_rot),
            tail_data=discretize_angles(tail_angles, b_rot),
            num_c=num_c,
            num_pos=num_pos,
            b_rot=b_rot,
            **kwargs,
        )

    @classmethod
    def from_vectors(
        cls, u_vecs, u_plus, u_minus, num_c: int, b_rot: int, **kwargs
    ) -> "InnerSelect":
        r"""Build from Majorana directions: SF ``u_vecs`` ``(R, B, N)``, and the
        ``D1``/``Q1`` one-body eigenvectors ``u_plus``/``u_minus`` (``(N_+, N)``/``(N_-, N)``).

        Only the *directions* matter here -- each vector is normalised, its magnitude being a
        PREPARE coefficient -- so raw DFTHC factors can be passed straight in.
        """
        u = np.asarray(u_vecs, float)
        if u.ndim != 3:
            raise ValueError(f"u_vecs must be 3D (R, B, N); got shape {u.shape}")
        tail = np.vstack([np.atleast_2d(u_plus), np.atleast_2d(u_minus)])
        if tail.shape[0] != u.shape[2]:
            raise ValueError(
                f"u_plus and u_minus must have N={u.shape[2]} rows in total; got {tail.shape[0]}"
            )
        unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
        angles = np.array([[angles_from_vector(unit(u[i, j])) for j in range(u.shape[1])]
                           for i in range(u.shape[0])])
        tail_angles = np.array([angles_from_vector(unit(v)) for v in tail])
        return cls.from_angles(
            angles, tail_angles, num_c=num_c, num_pos=len(np.atleast_2d(u_plus)), b_rot=b_rot,
            **kwargs,
        )

    @classmethod
    def symbolic(
        cls, num_orb: 'SymbolicInt', num_r: 'SymbolicInt', num_b: 'SymbolicInt',
        num_c: 'SymbolicInt', b_rot: 'SymbolicInt', **kwargs
    ) -> "InnerSelect":
        r"""A costing-only placeholder at symbolic ``N``, ``R``, ``B``, ``C``, ``b_rot``."""
        return cls(
            angle_data=Shaped((num_r, num_b, num_orb - 1)),
            tail_data=Shaped((num_orb, num_orb - 1)),
            num_c=num_c,
            num_pos=num_orb,
            b_rot=b_rot,
            **kwargs,
        )
