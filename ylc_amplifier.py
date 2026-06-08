"""
Module 1 of the modified Kuperberg pipeline.

Yoder-Low-Chuang fixed-point amplitude amplification
   T. J. Yoder, G. H. Low, I. L. Chuang,
   "Fixed-point quantum search with an optimal number of queries",
   Phys. Rev. Lett. 113, 210501 (2014).

Given a pure state |Phi> = alpha|Good> + beta|Bad> and a unitary A|0...0> = |Phi>
with |alpha| >= lambda_lower, this module amplifies |Good> to amplitude >= sqrt(1 - delta^2) 
using L = O(log(1/delta) / lambda_lower) queries.  
Unlike standard Grover, the success probability is monotone in L
for *any* true alpha >= lambda_lower, so the protocol never overshoots.

Define two reflections: S_A(phi) = I - (1 - e^{i phi}) A|0><0|A^dagger,
and S_0(phi) = I - (1-e^{i phi}) |0...0><0...0|. The algorithm proceeds 
by successively applying two reflections S_A(beta_j) S_0(alpha_j) for a list of
beta_j's and alpha_j's where j iterates over [0,...,L-1].
The expensive oracle is S_A(phi).
"""

from __future__ import annotations
import math
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------------
# Helper: reflection-with-phase about |0...0>.
# ---------------------------------------------------------------------------
def _S0_reflection(n_qubits: int, phi: float) -> QuantumCircuit:
    """
    Build the unitary  S_0(phi) = I - (1 - e^{i phi}) |0...0><0...0|.

    Strategy: bracket a multi-controlled phase gate (which fires only on
    |11...1>) with X gates on every wire, mapping the all-zero state to
    the all-one state and back.
    """
    qc = QuantumCircuit(n_qubits, name=f"S_0({phi:.3f})")
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.p(phi, 0)
    else:
        qc.mcp(phi, list(range(n_qubits - 1)), n_qubits - 1)
    qc.x(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# Helper: YLC phase schedule
# ---------------------------------------------------------------------------
def _ylc_phase_schedule(L: int, delta: float) -> tuple[list[float], list[float]]:
    """
    Compute (alphas, betas), each of length L, for the YLC algorithm.

        gamma^{-1} = cosh( (1/L) * arccosh(1/delta) )            [Chebyshev]
        alpha_j    = 2 * arccot( tan(2*pi*j/L) * sqrt(1 - gamma^2) )    [j=1..L]
        beta_j     = -alpha_{L-j+1}

    arccot is taken on the principal branch (0, pi):  arccot(x) = pi/2 - arctan(x).
    """
    gamma_inv = math.cosh(math.acosh(1.0 / delta) / L)
    one_minus_gamma_sq = 1.0 - (1.0 / gamma_inv) ** 2
    s = math.sqrt(max(0.0, one_minus_gamma_sq))

    alphas: list[float] = []
    for j in range(1, L + 1):
        x = math.tan(2.0 * math.pi * j / L) * s
        alpha_j = 2.0 * (math.pi / 2.0 - math.atan(x))    # 2 * arccot(x)
        alphas.append(alpha_j)
    # beta_j = -alpha_{L-j+1}; in 0-indexed terms that's -alphas[L-j]
    betas = [-alphas[L - j] for j in range(1, L + 1)]
    return alphas, betas


# ---------------------------------------------------------------------------
# Main: YLC fixed-point amplitude amplifier
# ---------------------------------------------------------------------------
def YLC_amplifier(
    initial_state: Statevector,
    lambda_lower: float,
    state_prep: tuple[QuantumCircuit, QuantumCircuit],
    SA_reflection: Callable[[float], QuantumCircuit],
    delta: float = 1e-2,
    sanity_check: bool = True,
) -> tuple[Statevector, QuantumCircuit]:
    """
    Yoder-Low-Chuang fixed-point amplitude amplification.

    Inputs
    ------
    initial_state : Statevector
        The pre-amplification state |Phi> = alpha|Good> + beta|Bad>.
        Only used for an (optional) sanity check that A|0> == |Phi>; the
        amplification itself re-derives the state from `state_prep`.
    lambda_lower  : float in (0, 1]
        A lower bound on |alpha|.  The amplifier guarantees success for
        every true alpha with |alpha| >= lambda_lower.
    state_prep    : (A, A_inverse)
        Two QuantumCircuits on the same n qubits, with  A|0...0> = |Phi>
        and  A_inverse = A^dagger.  Only `A` is used in this version (for
        the one-time initial state preparation); `A_inverse` is accepted
        for interface compatibility with downstream modules.
    SA_reflection  : phi -> QuantumCircuit
        A *function* that, given a phase angle phi, returns a circuit
        implementing the reflection-with-phase about the prepared state,
            S_A(phi) = I - (1 - e^{i phi}) * A|0><0|A^dagger,
        on the same n qubits as A. 
    delta         : float in (0, 1), default 1e-2
        Target failure amplitude.  After amplification,
            |<Bad | amplified_state>| <= delta.

    Returns
    -------
    amplified_state : Statevector
        The state after applying the full YLC circuit to |0...0>.
    circuit         : QuantumCircuit
        The circuit itself; circuit . |0...0> == amplified_state.
    """
    A, A_inv = state_prep
    n_qubits = A.num_qubits

    if A_inv.num_qubits != n_qubits:
        raise ValueError("A and A_inverse must act on the same number of qubits.")
    if not (0.0 < lambda_lower <= 1.0):
        raise ValueError("lambda_lower must lie in (0, 1].")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")

    if sanity_check:
        prepared = Statevector.from_label("0" * n_qubits).evolve(A)
        if not prepared.equiv(initial_state):
            raise ValueError("state_prep[0] applied to |0...0> does not match initial_state.")

    # ------------------------------------------------------------------
    # 1. Pick L: smallest odd integer with  L >= log(2/delta) / (2 arcsin(lambda)).
    # ------------------------------------------------------------------
    L_real = math.log(2.0 / delta) / (2.0 * math.asin(lambda_lower))
    L = max(1, math.ceil(L_real))
    if L % 2 == 0:
        L += 1

    # ------------------------------------------------------------------
    # 2. Phase schedule.
    # ------------------------------------------------------------------
    alphas, betas = _ylc_phase_schedule(L, delta)

    # ------------------------------------------------------------------
    # 3. Build the circuit:  G_L ... G_1 . A . |0...0>
    #    with  G_j = - S_A(beta_j) . S_0(alpha_j).
    #
    #    Compared to textbook YLC, the outer reflection A . S_0(beta_j) . A^dagger
    #    is replaced by the user-provided S_A(beta_j); we no longer apply A
    #    and A^dagger inside the loop, so A_inv is unused here. 
    # ------------------------------------------------------------------
    A_gate = A.to_gate(label="A")
    _ = A_inv                                    # retained in the API; unused here

    circ = QuantumCircuit(n_qubits, name=f"YLC(L={L})")
    circ.append(A_gate, range(n_qubits))         # initial state prep: |Phi> = A|0>

    for j in range(L):
        s0 = _S0_reflection(n_qubits, alphas[j]).to_gate(label=f"S_0(alpha_{j+1})")
        sa = SA_reflection(betas[j]).to_gate(label=f"S_A(beta_{j+1})")
        circ.append(s0, range(n_qubits))         # inner: S_0 
        circ.append(sa, range(n_qubits))         # outer: user-supplied S_A

    # ------------------------------------------------------------------
    # 4. Apply circ to |0...0> to get the final state.
    # ------------------------------------------------------------------
    amplified_state = Statevector.from_label("0" * n_qubits).evolve(circ)
    return amplified_state, circ


# ---------------------------------------------------------------------------
# Self-test: amplify a tiny single-qubit example.
#   |Phi> = sin(theta)|1> + cos(theta)|0>.
#   A = R_y(2*theta); A^dagger = R_y(-2*theta).
#   S_A(phi) is built directly as A . S_0(phi) . A^dagger so the test mimics
#   "the user has an efficient S_A oracle" without secretly relying on YLC's
#   internal construction.
#
# Note: with |Good> identified as |0...0>, we have alpha = <0|Phi> = cos(theta),
# so the initial success amplitude is cos(theta), not sin(theta).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    theta = 1.30                                 # cos(1.30) ~ 0.2675 = true alpha
    lam   = 0.25                                 # known lower bound
    delta = 1e-3

    A = QuantumCircuit(1, name="A");      A.ry(2 * theta, 0)
    A_inv = QuantumCircuit(1, name="A^dagger"); A_inv.ry(-2 * theta, 0)

    def SA_reflection(phi: float) -> QuantumCircuit:
        """User-provided S_A(phi) = A . S_0(phi) . A^dagger on 1 qubit."""
        qc = QuantumCircuit(1, name=f"S_A({phi:.3f})")
        qc.compose(A_inv, qubits=[0], inplace=True)
        qc.x(0); qc.p(phi, 0); qc.x(0)           # this is S_0(phi) on 1 qubit
        qc.compose(A, qubits=[0], inplace=True)
        return qc

    phi0 = Statevector.from_label("0").evolve(A)
    final, circ = YLC_amplifier(phi0, lam, (A, A_inv), SA_reflection, delta=delta)

    p_good = abs(final.data[0]) ** 2             
    print(f"depth of YLC circuit:   {circ.decompose().depth()}")
    print(f"initial P(Good):        {math.cos(theta) ** 2:.6f}")
    print(f"amplified P(Good):      {p_good:.6f}")
    print(f"target 1 - delta^2:     {1 - delta ** 2:.6f}")
    print("OK" if p_good >= 1 - delta ** 2 else "BELOW TARGET")