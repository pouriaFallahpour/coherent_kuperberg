"""
Module 1 of the modified Kuperberg pipeline.

Yoder-Low-Chuang fixed-point amplitude amplification
   T. J. Yoder, G. H. Low, I. L. Chuang,
   "Fixed-point quantum search with an optimal number of queries",
   Phys. Rev. Lett. 113, 210501 (2014).

Setup
-----
We have an n-qubit state |Phi> = g|Good> + b|Bad>  (g^2 + b^2 = 1, real WLOG)
and a known lower bound  g^2 >= lam.  Access:

    A          : n-qubit unitary,  A|0...0> = |Phi>
    O_Good     : (n+1)-qubit oracle that XORs a Good-indicator into a 1-qubit
                 ancilla:
                     O_Good |Good>|b> = |Good>|b XOR 1>
                     O_Good |Bad> |b> = |Bad> |b>

Reflections (user-specified sign conventions):
    S_s(alpha) = I - (1 - e^{-i alpha}) |Phi><Phi|
    S_t(beta)  = I - (1 - e^{+i beta}) |Good><Good|

Grover-like iterate:
    G(alpha, beta) = - S_s(alpha) . S_t(beta)        (the leading - is global)

Iteration count and schedule:
    L = smallest odd integer strictly greater than  log(2/delta) / sqrt(lam)
    l = (L - 1) / 2
    1/gamma = cosh( arccosh(1/delta) / L )           (analytic continuation
                                                      of  cos(arccos(.)/L) )
    For j = 1, ..., l:
        alpha_j = 2 * arccot( tan(2 pi j / L) * sqrt(1 - gamma^2) )
        beta_j  = - alpha_{l - j + 1}

Output S_L of l iterates, applied to the prepared state:
    S_L = G(alpha_l, beta_l) ... G(alpha_1, beta_1)
    |Psi> = S_L . (A tensor I_anc) . |0...0>_R |0>_anc
"""

from __future__ import annotations
import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit.qasm2 import dumps    # OpenQASM 2 (most readable, like the old .qasm())


# ---------------------------------------------------------------------------
# Phase schedule  (l alpha's and l beta's, indexed 0..l-1 here = 1..l in spec)
# ---------------------------------------------------------------------------
def _ylc_phase_schedule(L: int, delta: float) -> tuple[list[float], list[float]]:
    """
    Compute (alphas, betas), each of length l = (L-1)//2.

        1/gamma = cosh( arccosh(1/delta) / L )
        alpha_j = 2 * arccot( tan(2 pi j / L) * sqrt(1 - gamma^2) )   j = 1..l
        beta_j  = - alpha_{l - j + 1}                                 j = 1..l

    arccot is the principal branch:  arccot(x) = pi/2 - arctan(x), in (0, pi).
    """
    l = (L - 1) // 2
    gamma_inv = math.cosh(math.acosh(1.0 / delta) / L)
    s = math.sqrt(max(0.0, 1.0 - (1.0 / gamma_inv) ** 2))   # sqrt(1 - gamma^2)

    alphas: list[float] = []
    for j in range(1, l + 1):
        x = math.tan(2.0 * math.pi * j / L) * s
        alpha_j = 2.0 * (math.pi / 2.0 - math.atan(x))     # = 2 * arccot(x)
        alphas.append(alpha_j)
    betas = [-alphas[l - j] for j in range(1, l + 1)]      # beta_j = -alpha_{l-j+1}
    return alphas, betas


# ---------------------------------------------------------------------------
# Multi-qubit reflection-with-phase about |0...0>, in Qiskit's e^{+i theta} form
#     S_0(theta) = I + (e^{+i theta} - 1) |0...0><0...0|
# Used inside S_s, NOT as a separate exported subroutine.
# ---------------------------------------------------------------------------
def _apply_S0(qc: QuantumCircuit, theta: float, qubits: list[int]) -> None:
    """In-place: append S_0(theta) on the given qubits of qc."""
    qc.x(qubits)
    if len(qubits) == 1:
        qc.p(theta, qubits[0])
    else:
        qc.mcp(theta, qubits[:-1], qubits[-1])
    qc.x(qubits)


# ---------------------------------------------------------------------------
# S_s(alpha) on the n-qubit state register:
#     S_s(alpha) = A . S_0(-alpha) . A^dagger
# because Qiskit's S_0 carries e^{+i theta} but the spec wants e^{-i alpha}
# on |Phi>, so we pass -alpha.
# ---------------------------------------------------------------------------
def _apply_Ss(qc: QuantumCircuit, alpha: float,
              R: list[int],
              A_gate, A_inv_gate) -> None:
    qc.append(A_inv_gate, R)
    _apply_S0(qc, -alpha, R)
    qc.append(A_gate, R)


# ---------------------------------------------------------------------------
# S_t(beta) on (R, ancilla):  e^{+i beta} phase on Good states.
#     O_Good  .  p(+beta) on ancilla  .  O_Good
# ---------------------------------------------------------------------------
def _apply_St(qc: QuantumCircuit, beta: float,
              R: list[int], a: int,
              O_Good_gate) -> None:
    qc.append(O_Good_gate, R + [a])
    qc.p(beta, a)
    qc.append(O_Good_gate, R + [a])


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def YLC_amplifier(
    state_prep: tuple[QuantumCircuit, QuantumCircuit],
    O_Good: QuantumCircuit,
    lam: float,
    delta: float = 1e-2,
) -> tuple[Statevector, QuantumCircuit, QuantumCircuit]:
    """
    Yoder-Low-Chuang fixed-point amplitude amplification.

    Parameters
    ----------
    state_prep : (A, A_inverse)
        A is an n-qubit QuantumCircuit with  A|0...0> = |Phi> = g|Good> + b|Bad>.
        A_inverse must equal A^dagger.
    O_Good : QuantumCircuit on n+1 qubits
        Convention: qubits 0..n-1 are the state register R, qubit n is the
        1-qubit Good-indicator ancilla a.  Acts as
            |Good>_R |b>_a -> |Good>_R |b XOR 1>_a
            |Bad>_R  |b>_a -> |Bad>_R  |b>_a
    lam : float in (0, 1]
        A known lower bound on the success probability g^2.
    delta : float in (0, 1)
        Target failure amplitude on |Bad> after amplification.

    Returns
    -------
    psi : Statevector on n+1 qubits
        The amplified joint state  S_L . (A tensor I_anc) . |0...0>_R |0>_a.
        With ancilla in |0>, the Good-component on R has amplitude  >= sqrt(1-delta^2).
    S_L : QuantumCircuit on n+1 qubits
        The l-fold YLC iterate  G(alpha_l, beta_l) ... G(alpha_1, beta_1)
        as a standalone circuit (does NOT include the initial A).
    S_L_inv : QuantumCircuit on n+1 qubits
        The Hermitian conjugate of S_L, ready for use in downstream modules
        (e.g. amplitude estimation, phase estimation, uncomputation steps).
    """
    A, A_inv = state_prep
    n = A.num_qubits

    if A_inv.num_qubits != n:
        raise ValueError("A and A_inverse must act on the same number of qubits.")
    if O_Good.num_qubits != n + 1:
        raise ValueError(f"O_Good must act on n+1 = {n+1} qubits, got {O_Good.num_qubits}.")
    if not (0.0 < lam <= 1.0):
        raise ValueError("lam (lower bound on g^2) must lie in (0, 1].")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")

    # ------------------------------------------------------------------
    # Iteration count:  smallest odd L > log(2/delta) / sqrt(lam).
    # ------------------------------------------------------------------
    L_real = math.log(2.0 / delta) / math.sqrt(lam)
    L = math.floor(L_real) + 1                            # smallest integer > L_real
    if L % 2 == 0:
        L += 1                                            # promote to odd
    l = (L - 1) // 2

    # ------------------------------------------------------------------
    # Phase schedule.
    # ------------------------------------------------------------------
    alphas, betas = _ylc_phase_schedule(L, delta)

    # ------------------------------------------------------------------
    # Build S_L = G(alpha_l, beta_l) ... G(alpha_1, beta_1)
    #
    # Qubit layout in S_L:
    #     qubits 0 .. n-1  : state register R
    #     qubit  n         : Good-indicator ancilla a
    # ------------------------------------------------------------------
    R = list(range(n))
    a = n

    A_gate      = A.to_gate(label="A")
    A_inv_gate  = A_inv.to_gate(label="A†")
    O_Good_gate = O_Good.to_gate(label="O_G")

    S_L = QuantumCircuit(n + 1, name=f"S_L(l={l})")
    for j in range(l):                          # j = 0..l-1  ↔  spec's 1..l
        # G_j = - S_s(alpha_j) . S_t(beta_j)    (leading - is a global phase)
        # Right-to-left in the operator product = first-to-last in gate order:
        _apply_St(S_L, betas[j], R, a, O_Good_gate)
        _apply_Ss(S_L, alphas[j], R, A_gate, A_inv_gate)

    # ------------------------------------------------------------------
    # Hermitian conjugate as a separate circuit.
    # ------------------------------------------------------------------
    S_L_inv = S_L.inverse()
    S_L_inv.name = f"S_L_inv(l={l})"

    # ------------------------------------------------------------------
    # Compute the amplified state by simulating  S_L . A|0>_R |0>_a.
    # ------------------------------------------------------------------
    prep_circ = QuantumCircuit(n + 1, name="prep")
    prep_circ.append(A_gate, R)
    full = prep_circ.compose(S_L)

    psi = Statevector.from_label("0" * (n + 1)).evolve(full)
    return psi, S_L, S_L_inv


# ---------------------------------------------------------------------------
# Self-test
#   n = 1; A = R_y(2 theta).  Good = |1>, so O_Good = CNOT(R -> ancilla).
#   g = sin(theta).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    theta = 0.30                                     # g = sin(0.30) ~= 0.2955
    g_true = math.sin(theta)
    lam = 0.05                                       # we only know g^2 >= 0.05
    delta = 0.05

    A     = QuantumCircuit(1, name="A");  A.ry(2.0 * theta, 0)
    A_inv = QuantumCircuit(1, name="A†"); A_inv.ry(-2.0 * theta, 0)

    O_Good = QuantumCircuit(2, name="O_Good")
    O_Good.cx(0, 1)                                  # ancilla XOR= (R == |1>)

    psi, S_L, S_L_inv = YLC_amplifier((A, A_inv), O_Good, lam, delta)

    # |Good>_R |0>_a  =  |q1=0, q0=1>  in Qiskit little-endian → index 1
    p_good = abs(psi.data[1]) ** 2
    print(f"true g^2 (initial P(Good)): {g_true ** 2:.6f}")
    print(f"after amplification:        {p_good:.6f}")
    print(f"target 1 - delta^2:         {1.0 - delta ** 2:.6f}")
    print(f"S_L depth (decomposed):     {S_L.decompose().depth()}")
    print("OK" if p_good >= 1.0 - delta ** 2 else "BELOW TARGET")
    # print(dumps(S_L.decompose()))