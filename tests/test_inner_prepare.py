import pytest 

from dfthcbe.prepare import InnerPrepare
from qualtran.drawing import show_bloq
from qualtran.linalg.lcu_util import sub_bit_prec_from_epsilon, preprocess_probabilities_for_reversible_sampling
import numpy as np
from qualtran import BloqBuilder
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
import cirq

@pytest.mark.intensive
@pytest.mark.parametrize(
    "N, R, C, B, b_coeffs, precision",
    [
        (3, 2, 1, 3, np.array([[0.4, 0.2, 0.2, 0.2], [0.1, 0.2, 0.1, 0.6]]),  1e-1),
        (2, 3, 1, 3, np.array([[0.4, 0.2, 0.2, 0.2], [0.1, 0.2, 0.1, 0.6], [0.5, 0.3, 0.1, 0.1]]),  1e-1),
        (1, 2, 1, 2, np.array([[0.4, 0.6, 0.2], [0.3, 0.2, 0.5]]),  1e-1),

    ]
)
def test_inner_prepare(N, R, C, B, b_coeffs, precision):
    b_k2 = sub_bit_prec_from_epsilon(B+1, precision)
    alt_table, keep_table = np.zeros((R*C, B+1)), np.zeros((R*C, B+1))
    sgn, d_sgn = np.zeros((R*C, B+1), int), np.zeros((R*C, B+1), int)
    for i in range(R*C):
        alt_vec, keep_vec, _ = preprocess_probabilities_for_reversible_sampling(
            unnormalized_probabilities=b_coeffs[i], epsilon=precision
        )
        alt_table[i] = alt_vec
        keep_table[i] = keep_vec
        row_sgn = (b_coeffs[i] < 0).astype(int)
        sgn[i] = row_sgn
        d_sgn[i] = row_sgn ^ row_sgn[alt_vec]

    bloq = InnerPrepare(
        alt=alt_table, keep=keep_table, b_k2=b_k2, num_orb=N, num_c=C, sgn=sgn, d_sgn=d_sgn, log_block_size=1,
        s=1,  # 1 qubit of phase gradient; reading succ_b = 1 only is exact at any s
    )
    # phase_grad holds a phase-gradient state, for the two rotations of Uniform
    bb = BloqBuilder()
    soqs = {r.name: bb.add_register(r) for r in bloq.signature.lefts() if r.name != 'phase_grad'}
    soqs['phase_grad'] = bb.add(PhaseGradientState(bloq.phase_grad_bitsize))
    out = bb.add_from(bloq.decompose_bloq(), **soqs)  # its own boxes: Uniform stays one box
    cbloq = bb.finalize(**dict(zip([r.name for r in bloq.signature.rights()], out)))

    import re
    circuit = cbloq.to_cirq_circuit()

    all_qubits = sorted(circuit.all_qubits(), key=str)

    def _reg_qubits(pattern):
        return sorted(
            (q for q in all_qubits if isinstance(q, cirq.NamedQubit) and re.fullmatch(pattern, q.name)),
            key=lambda q: q.name,
        )

    x_o_qubits = _reg_qubits(r"x_o\d*")
    b_qubits = _reg_qubits(r"b\d*")
    r_qubits = _reg_qubits(r"r\d*")

    starting_state = np.zeros((2 ** len(x_o_qubits),))
    starting_state[:R*C] = 1 / np.sqrt(R*C)
    x_o_list = [_ for _ in x_o_qubits]
    circuit = [cirq.StatePreparationChannel(starting_state).on(*x_o_list)] + [circuit]

    circuit = cirq.Circuit(circuit)
    
    if cirq.__version__ >= "1.6.1":
        # Split boxes over 13 qubits: cirq >= 1.6.1 would build their full unitary. Reuse the
        # temporary qubits of their parts, or the state vector outgrows memory.
        reuse = cirq.DecompositionContext(cirq.GreedyQubitManager('anc', maximize_reuse=True))
        circuit = cirq.Circuit(
            cirq.decompose(circuit, keep=lambda op: len(op.qubits) <= 13, context=reuse)
        )
    
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    sv = result.final_state_vector

    joint_qubits = x_o_qubits + b_qubits + r_qubits

    joint_indices = [result.qubit_map[q] for q in joint_qubits]
    n = len(result.qubit_map)

    succ = result.qubit_map[cirq.NamedQubit("succ_b")]
    p_fail = np.sum(np.abs(sv.reshape([2] * n).take(0, axis=succ)) ** 2)
    assert np.isclose(p_fail, bloq.uniform.failure_probability())

    joint_probs = {}
    for idx, amp in enumerate(sv):
        bits = format(idx, f"0{n}b")
        if abs(amp) < 1e-5 or bits[succ] == "0":
            continue
        joint_bits = "".join(bits[i] for i in joint_indices)
        joint_val = int(joint_bits, 2)
        joint_probs[joint_val] = joint_probs.get(joint_val, 0) + abs(amp) ** 2

    stapeled_state = np.zeros((R*C*(B+1)))
    for i, (val, prob) in enumerate(sorted(joint_probs.items())):
        stapeled_state[i] = R*C * prob / (1 - p_fail)
    b_probs = (b_coeffs / b_coeffs.sum(axis=1, keepdims=True)).flatten()
    print(np.max(np.abs(stapeled_state - b_probs)))
    assert np.allclose(stapeled_state, b_probs, rtol=0., atol=precision)
