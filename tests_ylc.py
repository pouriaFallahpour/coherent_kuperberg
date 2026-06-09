"""
Tests for YLC_amplifier with complex-amplitude states on 2 and 3 qubits.

Each test fixes a target state |Phi> with non-trivial phases on every basis
amplitude, builds A = StatePreparation(|Phi>) and its inverse, marks "Good"
as the subspace where state-register qubit 0 is |1> (so O_Good is a single
CNOT), and runs the amplifier.  We check four things:

  1. P(Good) after amplification is at least 1 - delta^2  (the YLC guarantee).
  2. The ancilla is restored to |0> in the joint final state (each S_t cleans up).
  3. The amplified joint state is properly normalised.
  4. S_L_inv really inverts S_L on a random test vector.
"""

from __future__ import annotations
import math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from ylc_amplifier import YLC_amplifier


# ---------------------------------------------------------------------------
# Index helpers (Qiskit little-endian: bit k of index i is (i >> k) & 1)
# ---------------------------------------------------------------------------
def good_indices_state(n: int) -> list[int]:
    """Indices on the n-qubit state register where state-q0 = 1."""
    return [i for i in range(2 ** n) if (i & 1) == 1]


def good_indices_joint(n: int) -> list[int]:
    """Joint-system indices (n state + 1 ancilla) for 'state-q0 = 1, ancilla = 0'."""
    return [i for i in range(2 ** (n + 1))
            if (i & 1) == 1 and ((i >> n) & 1) == 0]


def ancilla_zero_indices(n: int) -> list[int]:
    """Joint indices with the ancilla bit (qubit n) = 0."""
    return [i for i in range(2 ** (n + 1)) if ((i >> n) & 1) == 0]


# ---------------------------------------------------------------------------
# One test
# ---------------------------------------------------------------------------
def run_test(name: str, phi_vec, lam: float, delta: float,
             rng_seed: int = 42) -> bool:
    phi_vec = np.asarray(phi_vec, dtype=complex)
    phi_vec = phi_vec / np.linalg.norm(phi_vec)
    n = int(round(math.log2(len(phi_vec))))
    assert 2 ** n == len(phi_vec), "len(phi_vec) must be a power of 2"

    g_sq_true = sum(abs(phi_vec[i]) ** 2 for i in good_indices_state(n))
    assert g_sq_true + 1e-12 >= lam, (
        f"lam={lam} is not a valid lower bound on g^2={g_sq_true:.6f}"
    )

    # State prep:  A|0...0> = |Phi>
    A = QuantumCircuit(n, name="A")
    A.append(StatePreparation(phi_vec), range(n))
    A_inv = A.inverse()
    A_inv.name = "A†"

    # Good oracle:  flip ancilla iff state-q0 = 1.
    # In our convention the ancilla is qubit index n of the (n+1)-qubit register.
    O_Good = QuantumCircuit(n + 1, name="O_Good")
    O_Good.cx(0, n)

    # Run YLC.
    psi, S_L, S_L_inv = YLC_amplifier((A, A_inv), O_Good, lam, delta)

    # ---------- check 1: P(Good) after amplification ----------
    p_good_after = sum(abs(psi.data[i]) ** 2 for i in good_indices_joint(n))
    target       = 1.0 - delta ** 2

    # ---------- check 2: ancilla returned to |0> ----------
    p_anc_zero = sum(abs(psi.data[i]) ** 2 for i in ancilla_zero_indices(n))

    # ---------- check 3: normalisation ----------
    p_total = float(np.sum(np.abs(psi.data) ** 2))

    # ---------- check 4: S_L_inv really inverts S_L ----------
    rng = np.random.default_rng(rng_seed)
    rv  = rng.standard_normal(2 ** (n + 1)) + 1j * rng.standard_normal(2 ** (n + 1))
    rv /= np.linalg.norm(rv)
    test_sv  = Statevector(rv)
    roundtrip = test_sv.evolve(S_L).evolve(S_L_inv)
    inv_err  = float(np.max(np.abs(roundtrip.data - test_sv.data)))

    ok_amp   = p_good_after >= target - 1e-9
    ok_anc   = abs(p_anc_zero - 1.0) < 1e-9
    ok_norm  = abs(p_total    - 1.0) < 1e-9
    ok_inv   = inv_err < 1e-8
    passed   = ok_amp and ok_anc and ok_norm and ok_inv

    print(f"\n=== {name} ===")
    print(f"  n = {n} state qubits + 1 ancilla")
    print(f"  lam = {lam},  delta = {delta}")
    print(f"  true g^2                : {g_sq_true:.6f}")
    print(f"  P(Good) before YLC      : {g_sq_true:.6f}")
    print(f"  P(Good) after YLC       : {p_good_after:.6f}     "
          f"{'OK' if ok_amp else 'FAIL'}  (target {target:.6f})")
    print(f"  P(ancilla = 0) after    : {p_anc_zero:.6f}     "
          f"{'OK' if ok_anc else 'FAIL'}")
    print(f"  total norm              : {p_total:.6f}     "
          f"{'OK' if ok_norm else 'FAIL'}")
    print(f"  ||S_L_inv S_L v - v||_∞ : {inv_err:.2e}    "
          f"{'OK' if ok_inv else 'FAIL'}")
    print(f"  S_L size / depth        : {S_L.size()} / {S_L.depth()}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = []

    # -----------------------------------------------------------------------
    # Test 1 — 2-qubit state, non-trivial phase on every component.
    #   |Phi> = sqrt(0.55)|00>            (|amp|^2 = 0.55)   Bad
    #         + sqrt(0.15) e^{i pi/3}|01> (|amp|^2 = 0.15)   Good
    #         + 0.5 e^{i pi/4}|10>        (|amp|^2 = 0.25)   Bad
    #         + sqrt(0.05) e^{-i pi/2}|11>(|amp|^2 = 0.05)   Good
    #   g^2 = 0.20.  We tell the amplifier only lam = 0.15.
    # -----------------------------------------------------------------------
    phi_2q = [
        np.sqrt(0.55),
        np.sqrt(0.15) * np.exp(1j * np.pi / 3),
        0.5            * np.exp(1j * np.pi / 4),
        np.sqrt(0.05) * np.exp(-1j * np.pi / 2),
    ]
    results.append(run_test(
        "Test 1: 2-qubit state, complex phases (g^2 = 0.20)",
        phi_2q, lam=0.15, delta=0.05,
    ))

    # -----------------------------------------------------------------------
    # Test 2 — 3-qubit state, irrational phases on every basis amplitude.
    # Magnitudes are picked so g^2 ~ 0.18 after normalisation;
    # we feed the amplifier a looser bound lam = 0.10.
    # -----------------------------------------------------------------------
    phi_3q = [
        0.40 * np.exp(1j * 0.70),    # |000>  Bad
        0.15 * np.exp(-1j * 1.20),   # |001>  Good
        0.30 * np.exp(1j * 2.10),    # |010>  Bad
        0.20 * np.exp(-1j * 0.50),   # |011>  Good
        0.40 * np.exp(1j * 1.50),    # |100>  Bad
        0.15 * np.exp(-1j * 2.80),   # |101>  Good
        0.50 * np.exp(1j * 0.30),    # |110>  Bad
        0.25 * np.exp(-1j * 1.70),   # |111>  Good
    ]
    results.append(run_test(
        "Test 2: 3-qubit state, irrational complex phases (g^2 ~ 0.18)",
        phi_3q, lam=0.10, delta=0.05,
    ))

    # -----------------------------------------------------------------------
    # Test 3 — 3-qubit state, very loose lower bound (lam << g^2)
    # to confirm YLC's fixed-point property: amplification still works,
    # and the algorithm does NOT over-rotate.
    # -----------------------------------------------------------------------
    phi_3q_b = [
        0.20 * np.exp(1j * 1.10),    # |000>  Bad
        0.45 * np.exp(-1j * 0.30),   # |001>  Good
        0.10 * np.exp(1j * 2.40),    # |010>  Bad
        0.40 * np.exp(-1j * 1.85),   # |011>  Good
        0.15 * np.exp(1j * 0.20),    # |100>  Bad
        0.30 * np.exp(-1j * 2.15),   # |101>  Good
        0.25 * np.exp(1j * 0.95),    # |110>  Bad
        0.45 * np.exp(-1j * 1.30),   # |111>  Good
    ]
    results.append(run_test(
        "Test 3: 3-qubit, loose bound (lam = 0.05, true g^2 ~ 0.60)",
        phi_3q_b, lam=0.05, delta=0.01,
    ))

    print()
    print("=" * 60)
    print("OVERALL:", "ALL PASS" if all(results) else "SOME FAILED",
          f"({sum(results)}/{len(results)} passed)")