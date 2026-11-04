r"""The alias data of the two PREPAREs: its QROAM read and erase, and exact pruned read costs.

Steps 2 and 6 of the ``README.md`` circuit read the alias tables with :class:`QROAMCleanHetero`;
steps 26 and 21 erase them, by X-measurement.

A read with ``k`` controls whose address walk visits ``l`` leaves costs ``l + k - 2`` ANDs. Qualtran
prunes every aligned block over which the table is constant to one leaf, so a constant run
``[a, b)`` costs :func:`~dfthcbe.utils.num_blocks` leaves, ``s(D)`` for a prefix ``[0, D)``. A
symbolic table has no entries to prune, so :class:`PrunedQROM` and :class:`QROAMCleanHetero` are
told the leaf count.
"""
from __future__ import annotations

import warnings
from functools import cached_property
from typing import cast, Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import Adjoint, Bloq, BloqBuilder, Register, Side, SoquetT
from qualtran.bloqs.basic_gates import CSwap, Toffoli
from qualtran.bloqs.data_loading.qroam_clean import (
    QROAMClean,
    QROAMCleanAdjoint,
    QROAMCleanAdjointWrapper,
)
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.swap_network import SwapWithZero
from qualtran.symbolics import ceil, is_symbolic, prod

from dfthcbe.utils import aligned_blocks, num_blocks

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.symbolics import SymbolicInt


# --- exact symbolic costs of pruned reads -------------------------------------------


@attrs.frozen
class PrunedQROM(QROM):
    r"""A ``QROM`` whose symbolic cost counts the ``num_leaves`` leaves of its pruned walk.

    Qualtran's symbolic count, ``prod(data_shape) + k - 2`` ANDs, makes every entry a leaf. With
    data, Qualtran walks the real table and ``num_leaves`` is ignored.
    """

    num_leaves: Optional['SymbolicInt'] = attrs.field(default=None, kw_only=True)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        counts = super().build_call_graph(ssa)
        if self.has_data() or self.num_leaves is None:
            return counts
        n_and = self.num_leaves + self.num_controls - 2
        return {**counts, And(): n_and, And().adjoint(): n_and}


# --- the alias read and its erase ---------------------------------------------------


def optimal_erase_log_block_size(qroam: 'QROAMCleanHetero') -> int:
    """The ``k`` with the cheapest measurement-based erase of ``qroam``."""
    rest = (0,) * (len(qroam.data_shape) - 1)
    erase = lambda k: QROAMCleanHeteroAdjointWrapper(qroam, log_block_sizes=(k, *rest))
    return min(range(int(qroam.data_shape[0]).bit_length()), key=lambda k: erase(k).num_toffoli)


@attrs.frozen
class ExactSwapWithZero(SwapWithZero):
    """``SwapWithZero`` with exact ``CSwap``s, at the same Toffoli count.

    Qualtran's ``CSwapApprox`` swaps only up to a sign set by the swapped values. Its own
    ``SelectSwapQROM`` swaps back, which cancels it; after a measurement-based erase it would stay
    on the address, as a wrong sign in the prepared state.
    """

    @cached_property
    def cswap_n(self) -> CSwap:
        return CSwap(self.target_bitsize)


@attrs.frozen
class QROAMCleanHetero(QROAMClean):
    r"""``QROAMClean`` with per-target block shapes and caller-owned (THRU) targets.

    A target in ``flat_target_idxs`` gets block shape ``(1, ..., 1)``: the shared address walk
    writes it with CNOTs, it skips the ``SwapWithZero`` and has no junk register. Correct only if
    its data is constant along every axis read in blocks. The targets and the junk are THRU,
    allocated by the caller in ``|0>``. The ``SwapWithZero`` wiring follows the data axes, so
    ``log_block_sizes`` with a 0, e.g. ``(0, k_2)``, decomposes, which Qualtran's ``QROAMClean``
    does not. The swaps are exact (:class:`ExactSwapWithZero`), so the read leaves no sign on the
    address.

    Args (beyond ``QROAMClean``'s):
        flat_target_idxs: the targets that skip the swap.
        num_leaves: the pruned walk's leaf count, for a symbolic table (:class:`PrunedQROM`).
        num_live_rows: the leading rows of the first axis that hold data, the rest being zero;
            ``None``: all of them. The erase stops its walk there.
        erase_log_block_sizes: ``(k, 0, ...)``, ``k`` the bits of the first address in the
            erase's one-hot; ``None`` lets Qualtran pick.
        exact_adjoint: ``.adjoint()`` is the exact unitary inverse, a second read, instead of the
            measurement-based erase; for simulation only, so it warns.
    """

    flat_target_idxs: Tuple[int, ...] = attrs.field(default=(), converter=tuple)
    num_leaves: Optional['SymbolicInt'] = attrs.field(default=None, kw_only=True)
    num_live_rows: Optional['SymbolicInt'] = attrs.field(default=None, kw_only=True)
    erase_log_block_sizes: Optional[Tuple['SymbolicInt', ...]] = attrs.field(
        default=None, kw_only=True
    )
    exact_adjoint: bool = attrs.field(default=False, kw_only=True)

    def with_flat_target_idxs(self, flat_target_idxs: Tuple[int, ...]) -> 'QROAMCleanHetero':
        return attrs.evolve(self, flat_target_idxs=tuple(flat_target_idxs))

    @cached_property
    def _target_reg_side(self) -> Side:
        return Side.THRU

    def _block_shape_for_target(self, i: int) -> Tuple['SymbolicInt', ...]:
        flat = i in self.flat_target_idxs
        return (1,) * len(self.block_sizes) if flat else tuple(self.block_sizes)

    def _active_axes(self) -> list:
        """Data axes read in blocks of more than one entry."""
        return [a for a, k in enumerate(self.log_block_sizes) if is_symbolic(k) or k > 0]

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        """One THRU junk register per non-flat target, holding its ``2**k - 1`` other copies."""
        junk_regs = []
        for i, reg in enumerate(self.target_registers):
            block_size = prod(self._block_shape_for_target(i))
            if is_symbolic(block_size) or block_size > 1:
                junk_regs.append(
                    attrs.evolve(reg, name='junk_' + reg.name, shape=(block_size - 1,))
                )
        return tuple(junk_regs)

    @cached_property
    def batched_data(self) -> list:
        """Per target, its block of each batch; a flat target takes the block's first entry."""
        out = []
        for i, data in enumerate(self.padded_data):
            block = self._block_shape_for_target(i)
            arr = np.zeros(tuple(self.batched_qrom_shape) + tuple(block), dtype=int)
            block_slices = tuple(slice(0, k) for k in block)
            for batch_idx in np.ndindex(cast(Tuple[int, ...], tuple(self.batched_qrom_shape))):
                data_idx = tuple(
                    slice(x * bs, x * bs + k)
                    for x, bs, k in zip(batch_idx, self.block_sizes, block)
                )
                arr[(*batch_idx, *block_slices)] = data[data_idx]
            out.append(arr)
        return out

    @cached_property
    def _qrom_axes(self) -> list:
        """Batch axes the QROM walks: an axis read in one block is one batch, no address bits."""
        shape = self.batched_qrom_shape
        return [a for a, n in enumerate(shape) if is_symbolic(n) or n > 1] or [0]

    @cached_property
    def qrom_bloq(self) -> QROM:
        # Qualtran's QROM drops a one-batch axis's selection register but would still index the
        # data along it, so the axis is squeezed out of the data as well.
        keep, bits = self._qrom_axes, self.batched_qrom_selection_bitsizes
        shapes = tuple(self._block_shape_for_target(i) for i in range(len(self.target_bitsizes)))
        qrom = PrunedQROM.build_from_bitsize(
            tuple(self.batched_qrom_shape[a] for a in keep), self.target_bitsizes,
            target_shapes=shapes, selection_bitsizes=tuple(bits[a] for a in keep),
            num_controls=self.num_controls,
        )
        qrom = attrs.evolve(qrom, num_leaves=self.num_leaves)
        if is_symbolic(self):
            return qrom
        drop = tuple(a for a in range(len(self.batched_qrom_shape)) if a not in keep)
        return qrom.with_data(*(np.squeeze(d, axis=drop) for d in self.batched_data))

    def _add_qrom_bloq(self, bb: 'BloqBuilder', ctrls, sel_l, targets, uncompute: bool = False):
        # The (0-bit) address soquet of a one-batch axis goes around the QROM, which has no
        # register for it.
        keep = self._qrom_axes
        ctrls, walked, targets = super()._add_qrom_bloq(
            bb, ctrls, [sel_l[a] for a in keep], targets, uncompute
        )
        sel_l = list(sel_l)
        for a, soq in zip(keep, walked):
            sel_l[a] = soq
        return ctrls, sel_l, targets

    @cached_property
    def swap_with_zero_bloqs(self) -> list:
        """Per target, its ``SwapWithZero`` over the axes read in blocks only; ``None`` if flat."""
        active = self._active_axes()
        return [
            None if i in self.flat_target_idxs or not active else ExactSwapWithZero(
                tuple(self.log_block_sizes[a] for a in active), target_bitsize=b,
                n_target_registers=tuple(self.block_sizes[a] for a in active),
            )
            for i, b in enumerate(self.target_bitsizes)
        ]

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: Dict[Bloq, 'SymbolicInt'] = {self.qrom_bloq: 1}
        for swz in self.swap_with_zero_bloqs:
            if swz is not None and any(is_symbolic(s) or s > 0 for s in swz.selection_bitsizes):
                ret[swz] = ret.get(swz, 0) + 1
        return ret

    @cached_property
    def batched_data_permuted(self) -> list:
        """The data in the order the erase sees it, after each target's swap."""
        if is_symbolic(*self.block_sizes):
            raise ValueError(f"Cannot decompose with symbolic block sizes {self.block_sizes}.")
        active = self._active_axes()
        lead_shape = tuple(n * k for n, k in zip(self.batched_qrom_shape, self.block_sizes))
        ret = []
        for i, data in enumerate(self.batched_data):
            block = cast(Tuple[int, ...], self._block_shape_for_target(i))
            swz = self.swap_with_zero_bloqs[i]
            if swz is None:  # flat: no permutation
                ret.append(np.asarray(self.padded_data[i]).reshape(lead_shape + (1,) * len(block)))
                continue
            active_dims = tuple(self.block_sizes[a] for a in active)
            permuted = np.zeros(data.shape + block, dtype=data.dtype)
            for sel_l in np.ndindex(cast(Tuple[int, ...], tuple(self.batched_qrom_shape))):
                for sel_k in np.ndindex(block):
                    sel = {reg.name: sel_k[a] for reg, a in zip(swz.selection_registers, active)}
                    squeezed = np.copy(data[sel_l]).reshape(active_dims)
                    permuted[(*sel_l, *sel_k)][:] = swz.call_classically(
                        **sel, targets=squeezed
                    )[-1].reshape(block)
            n = len(block)
            axes = [x for a in range(n) for x in [a, a + n]] + [a + 2 * n for a in range(n)]
            ret.append(permuted.transpose(axes).reshape(lead_shape + block))
        return ret

    def _add_swap_with_zero_bloq(
        self, bb: 'BloqBuilder', selection: list, targets: list, uncompute: bool = False
    ):
        # One selection soquet per data axis (bitsize 0 on those of block size 1, passed through);
        # each non-flat target's swap acts on the active-axis view of its block.
        active = self._active_axes()
        sel_k, out_targets = list(selection), list(targets)
        for i, swz in enumerate(self.swap_with_zero_bloqs):
            if swz is None:
                continue
            full = out_targets[i]
            in_soqs = {reg.name: sel_k[a] for reg, a in zip(swz.selection_registers, active)}
            in_soqs['targets'] = full.reshape(tuple(self.block_sizes[a] for a in active))
            out = bb.add_d(swz.adjoint() if uncompute else swz, **in_soqs)
            for reg, a in zip(swz.selection_registers, active):
                sel_k[a] = out[reg.name]
            out_targets[i] = np.asarray(out['targets'], dtype=object).reshape(full.shape)
        return sel_k, out_targets

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        ctrl = [soqs.pop(reg.name) for reg in self.control_registers]
        selection = [soqs.pop(reg.name) for reg in self.selection_registers]
        if is_symbolic(*self.block_sizes):
            raise ValueError(f"Cannot decompose with symbolic block sizes {self.block_sizes}.")
        qrom_targets = []
        for i, reg in enumerate(self.target_registers):
            qrom_target = np.empty(self._block_shape_for_target(i), dtype=object)
            qrom_target.flat[0] = soqs.pop(reg.name)
            if qrom_target.size > 1:
                junk = np.asarray(soqs.pop('junk_' + reg.name), dtype=object).reshape(-1)
                qrom_target.flat[1:] = junk
            qrom_targets.append(qrom_target)
        assert not soqs, f"All registers must have been used by now. Found: {soqs}"
        if any(b > 1 for b in self.block_sizes):
            ctrl, selection, qrom_targets = self._build_composite_bloq_with_swz_clean(
                bb, ctrl, selection, qrom_targets
            )
        else:
            ctrl, selection, qrom_targets = self._add_qrom_bloq(bb, ctrl, selection, qrom_targets)
        out = {reg.name: soq for reg, soq in zip(self.control_registers, ctrl)}
        out |= {reg.name: soq for reg, soq in zip(self.selection_registers, selection)}
        for reg, target in zip(self.target_registers, qrom_targets):
            out[reg.name] = target.flat[0]
            if target.size > 1:
                out['junk_' + reg.name] = target.flat[1:]
        return out

    def on_classical_vals(self, **vals):
        # Qualtran's QROAMClean version zips every target's permuted data against the junk
        # registers, which misaligns once a flat target has none.
        out = SelectSwapQROM.on_classical_vals(self, **vals)
        selection = tuple(vals[reg.name] for reg in self.selection_registers)
        for i, reg in enumerate(self.target_registers):
            if prod(self._block_shape_for_target(i)) > 1:
                out['junk_' + reg.name] = self.batched_data_permuted[i][selection].flat[1:]
        return out

    def adjoint(self) -> Bloq:
        if self.exact_adjoint:
            warnings.warn(
                "QROAMCleanHetero: exact unitary adjoint instead of the measurement-based erase; "
                "for simulation, it costs a second read",
                stacklevel=2,
            )
            return Adjoint(self)
        blocks = {} if self.erase_log_block_sizes is None else {
            'log_block_sizes': self.erase_log_block_sizes
        }
        return QROAMCleanHeteroAdjointWrapper(self, **blocks)


@attrs.frozen
class QROAMCleanHeteroAdjointWrapper(QROAMCleanAdjointWrapper):
    """The measurement-based erase of :class:`QROAMCleanHetero`: flat targets have no junk, and
    the THRU targets come back as fresh ``|0>`` registers.

    The phase fixup is a one-hot over the low ``k`` bits of the first address and all of the
    others, and a walk over the rest of the first, up to the last live row and past the zero rows
    in aligned blocks (Eq. B50). The inner ``QROAMCleanAdjoint`` only carries the registers.
    """

    @cached_property
    def num_toffoli(self) -> 'SymbolicInt':
        """The walk's ``leaves - 2`` plus the one-hot's ``2**k M - 2``, ``M`` the other values."""
        qc, k = self.qroam_clean, self.log_block_sizes[0]
        leaves = ceil(qc.data_shape[0] / 2**k)
        if qc.num_live_rows is not None:
            if qc.has_data():
                assert not any(np.any(d[qc.num_live_rows:]) for d in qc.data), "dead rows not zero"
            live = ceil(qc.num_live_rows / 2**k)
            symbolic = is_symbolic(live, leaves)
            dead = aligned_blocks(live, leaves) if symbolic else num_blocks(live, leaves)
            leaves = live + dead
        n = leaves + 2**k * prod(qc.data_shape[1:]) - 4 + qc.num_controls
        return n if is_symbolic(n) else max(0, n)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {Toffoli(): self.num_toffoli}

    @cached_property
    def qroam_clean_adjoint_bloq(self) -> QROAMCleanAdjoint:
        qc = self.qroam_clean
        shapes = tuple(qc._block_shape_for_target(i) for i in range(len(qc.target_bitsizes)))
        if qc.has_data():
            return QROAMCleanAdjoint.build_from_data(
                *qc.batched_data_permuted, target_bitsizes=qc.target_bitsizes,
                target_shapes=shapes, log_block_sizes=self.log_block_sizes,
                num_controls=qc.num_controls,
            )
        return QROAMCleanAdjoint.build_from_bitsize(
            qc.data_shape, target_bitsizes=qc.target_bitsizes, target_shapes=shapes,
            log_block_sizes=self.log_block_sizes, num_controls=qc.num_controls,
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        qc, adj = self.qroam_clean, self.qroam_clean_adjoint_bloq
        for i, (target, adj_target) in enumerate(zip(qc.target_registers, adj.target_registers)):
            junk = soqs.pop('junk_' + target.name, np.array([]))
            soqs[adj_target.name] = np.array(
                [soqs.pop(target.name), *np.asarray(junk, dtype=object).reshape(-1)]
            ).reshape(qc._block_shape_for_target(i))
        out: Dict[str, 'SoquetT'] = dict(bb.add_d(adj, **soqs))  # the selection only
        for i, target in enumerate(qc.target_registers):
            out[target.name] = bb.allocate(dtype=target.dtype)
            n = prod(qc._block_shape_for_target(i))
            if not is_symbolic(n) and n > 1:
                out['junk_' + target.name] = np.array(
                    [bb.allocate(dtype=target.dtype) for _ in range(n - 1)], dtype=object
                )
        return out

    def adjoint(self) -> QROAMCleanHetero:
        return self.qroam_clean
