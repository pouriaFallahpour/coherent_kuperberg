"""
Module 2 of the modified Kuperberg pipeline.

LMR (Lloyd-Mohseni-Rebentrost) sample-based implementation of S_s(alpha).

   S. Lloyd, M. Mohseni, P. Rebentrost,
   "Quantum principal component analysis",
   Nat. Phys. 10, 631-633 (2014).

Given kappa copies of |Phi> (on separate n-qubit registers) and the working
n-qubit register R in some state rho_R, this module applies

        S_s(alpha) = I - (1 - e^{-i alpha}) |Phi><Phi|
                   = exp(-i alpha |Phi><Phi|)         (since |Phi><Phi|^2 = |Phi><Phi|)

to rho_R without needing the state-prep unitary A.

Mechanism.  Per-step LMR identity:

    tr_C [ exp(-i S_{RC} dt) (rho_R \otimes |Phi><Phi|_C) exp(+i S_{RC} dt) ]
        = rho_R - i dt [|Phi><Phi|, rho_R] + O(dt^2)

i.e. one infinitesimal step of the desired evolution.  Repeating kappa times
with dt = alpha / kappa accumulates total time alpha; the residual error is
O(alpha^2 / kappa) in trace distance.

Output of this module is a QuantumCircuit; tracing out the copies and the
ancilla is the *caller's* responsibility (the circuit itself is unitary).

Qubit layout in the returned circuit (n + n*kappa + 1 qubits total):
    indices [0, n)                         : working register R
    indices [n + k*n, n + (k+1)*n)         : copy k    (k = 0..kappa-1)
    index   n*(1+kappa)                    : 1-qubit ancilla
"""

from __future__ import annotations

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Single partial register SWAP:  exp(-i S_{R,C} dt)
# ---------------------------------------------------------------------------
def _swap_hamiltonian(
    qc: QuantumCircuit,
    R_qubits: list[int],
    C_qubits: list[int],
    ancilla: int,
    dt: float,
) -> None:
    """
    Append exp(-i S_{R,C} dt) to qc, where S_{R,C} is the register-SWAP that
    maps |x>_R |y>_C to |y>_R |x>_C.

    Implementation.  S_{R,C}^2 = I, so S has eigenvalues +-1 correspond to 
    two eigenspaces V^+ and V^-. Let P^+ and P^- be the projection to these
    eigenspaces. We have that P^+ = (I+S)/2 and P^- = (I-S)/2. Then we have 
    S = P^+ + (-1) P^-. Therefore, we have e^{-iS dt} = e^{-i dt} P^+ + e^{i dt} P^
    We bin the joint state |psi>_RC into the two eigenspaces by entangling them with a Hadamard
    ancilla through a controlled register-SWAP; apply R_x(2 dt) on the ancilla
    so the V^+ eigenspace picks up e^{-i dt} and the V^- eigenspace picks up
    e^{+i dt}; then uncompute the entanglement, leaving the ancilla in |0>.

    Cost: 2 Hadamards + 2n CSWAPs + 1 R_x per.
    """
    if len(R_qubits) != len(C_qubits):
        raise ValueError("R and C must hold the same number of qubits.")

    qc.h(ancilla)
    for r, c in zip(R_qubits, C_qubits):
        qc.cswap(ancilla, r, c)
    qc.rx(2.0 * dt, ancilla)                 # +1 eigenspace -> e^{-i dt}, -1 -> e^{+i dt}
    for r, c in zip(R_qubits, C_qubits):
        qc.cswap(ancilla, r, c)
    qc.h(ancilla)


# ---------------------------------------------------------------------------
# Coherent LMR simulation
# ---------------------------------------------------------------------------
def LMR_simulation(alpha: float, kappa: int, n: int) -> QuantumCircuit:
    """
    Build the LMR approximation of S_s(alpha) = exp(-i alpha |Phi><Phi|) using
    kappa copies of |Phi> as a resource.

    Parameters
    ----------
    alpha : float
        The desired phase angle.
    kappa : int >= 1
        Number of LMR (Trotter) steps; one copy of |Phi> is consumed per step.
        Trace-distance error scales as O(alpha^2 / kappa).
    n : int >= 1
        Qubits per register (so |Phi> is an n-qubit state).

    Returns
    -------
    QuantumCircuit on n*(1+kappa) + 1 qubits.
    Pre-conditions on the caller:
        - qubits [0, n)                       hold rho_R, the working state;
        - qubits [n, n*(1+kappa))             hold kappa independent copies of |Phi>,
                                              one per n-qubit block;
        - qubit  n*(1+kappa)                  is in |0>.
    Post-conditions after running:
        - the ancilla is restored to |0>;
        - the copies are in some state entangled with R; the caller must trace
          them out (e.g. via partial_trace, or by measurement-and-discard).
    """
    if kappa < 1:
        raise ValueError("kappa must be a positive integer.")
    if n < 1:
        raise ValueError("n must be a positive integer.")

    total = n * (1 + kappa) + 1
    qc = QuantumCircuit(total, name=f"LMR(alpha={alpha:.4f}, kappa={kappa})")

    R = list(range(n))
    ancilla = n * (1 + kappa)
    dt = alpha / kappa

    for k in range(kappa):
        C = list(range(n + k * n, n + (k + 1) * n))
        _swap_hamiltonian(qc, R, C, ancilla, dt)

    return qc


# ---------------------------------------------------------------------------
# Self-test: trace-distance convergence  d_tr(rho_LMR, rho_exact) ~ 1/kappa.
#
# We simulate the LMR channel step-by-step (tracing out the copy after every
# step), which is what the algorithm physically does, and avoids exponential
# blow-up in the simulation cost as kappa grows.
# Note that direct use of LMR_simulation would result in exponential-size 
# matrix manipulation but by tracing out used copies at each step, we avoid
# such issues.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"TODO")
    