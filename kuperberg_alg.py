"""
Kuperberg's sieve for the Dihedral Hidden Subgroup Problem on D_{2^n}.

This is a *simulation* of the algorithm: the hidden shift `s` is fixed at the
top of the script, an oracle circuit is built that hides it, and Kuperberg's
sieve recovers it.  We use Statevector objects to represent each labeled qubit
in the pool, because the sieve maintains many coherent single-qubit states at
once and a single monolithic circuit would be unwieldy.

Tested mentally against Qiskit 1.x / 2.x APIs.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Statevector, partial_trace


# ---------------------------------------------------------------------------
# Data structure: a labeled qubit (k, |phi_k>) where |phi_k> = (|0>+w^{ks}|1>)/sqrt2
# ---------------------------------------------------------------------------
@dataclass
class LabeledQubit:
    k: int               # classical label in Z_N (known)
    state: Statevector   # single-qubit pure state carrying the phase w^{ks}


# ---------------------------------------------------------------------------
# MODULE 1 -- Oracle and coset-state preparation
# ---------------------------------------------------------------------------
def hiding_oracle(n: int, s: int) -> QuantumCircuit:
    """
    Build a circuit acting on (A_reg of n qubits, parity qubit, B_reg of n qubits)
    that maps |x>|b>|0> -> |x>|b>|f(x,b)>  where
        f(x, 0) = x          mod N
        f(x, 1) = x + s      mod N
    so f is constant on left cosets {(x,0),(x-s,1)} of the hidden subgroup.

    In a real attack the oracle is given; here we synthesize it because we
    know s and we want a self-contained demo.
    """
    N = 1 << n
    A = QuantumRegister(n, "x")     # element of Z_N
    P = QuantumRegister(1, "b")     # parity bit (Z_2 part)
    B = QuantumRegister(n, "y")     # output register
    qc = QuantumCircuit(A, P, B, name="O_f")

    # Copy x into y:  y <- x   (CNOT pairwise)
    for i in range(n):
        qc.cx(A[i], B[i])

    # If parity bit set, add s modulo N to y:  y <- y + s  (controlled by P[0])
    # Two interchangeable implementations are available below:
    #   - _controlled_add_const            : QFT-based Draper adder, scales well
    #   - _controlled_add_const_permutation: dense permutation matrix, simpler
    qc.append(_controlled_add_const(n, s).to_gate(label=f"+{s}"), [P[0], *B])

    return qc


def _controlled_add_const(n: int, c: int) -> QuantumCircuit:
    """
    Circuit on (ctrl, n target qubits) that adds the constant c mod 2^n to the
    target register when ctrl = 1.  Implemented as QFT . controlled-phase . QFT^{-1}.
    """
    N = 1 << n
    c = c % N
    ctrl = QuantumRegister(1, "c")
    tgt  = QuantumRegister(n, "t")
    qc = QuantumCircuit(ctrl, tgt)

    qc.append(QFT(n, do_swaps=False).to_gate(), tgt)
    for j in range(n):
        # phase on qubit j is 2*pi * c * 2^j / 2^n, but the bit ordering of QFT(do_swaps=False)
        # places the highest-weight qubit at index n-1; this matches Draper's convention.
        angle = 2 * math.pi * c * (1 << j) / N
        qc.cp(angle, ctrl[0], tgt[j])
    qc.append(QFT(n, do_swaps=False, inverse=True).to_gate(), tgt)
    return qc


def _controlled_add_const_permutation(n: int, c: int) -> QuantumCircuit:
    """
    Simpler alternative to `_controlled_add_const`: since c is a *classical*
    constant, the map  |y> -> |y + c mod 2^n>  is just a fixed permutation of
    the 2^n basis states.  We build that permutation matrix directly and wrap
    it as a controlled unitary.

    Pros: reads in five lines; no Fourier-basis reasoning needed.
    Cons: materialises a dense N x N matrix, so it doesn't scale past n ~ 12.
          For real hardware or large n, use the QFT-based Draper adder above
          (which has an O(n^2) gate decomposition).

    Both functions are interchangeable for the purposes of this demo.
    """
    N = 1 << n
    c = c % N

    perm = np.zeros((N, N), dtype=complex)
    for y in range(N):
        perm[(y + c) % N, y] = 1.0
    add_gate = UnitaryGate(perm, label=f"+{c}").control(1)

    ctrl = QuantumRegister(1, "c")
    tgt  = QuantumRegister(n, "t")
    qc = QuantumCircuit(ctrl, tgt)
    qc.append(add_gate, [ctrl[0], *tgt])
    return qc


def generate_labeled_qubit(n: int, s: int, rng: random.Random) -> LabeledQubit:
    """
    Prepare a coset state, apply QFT_{Z_N} to the x-register, measure it, and
    return the resulting (label k, single-qubit state on the parity register).

    Rather than running the full oracle + measurement (which works but is slow
    in simulation), we exploit the fact that, given s, the distribution of
    labels k is uniform on Z_N and the post-measurement state is exactly
        |phi_k> = (|0> + e^{2 pi i k s / N} |1>) / sqrt(2).
    We therefore *construct* this state directly.  The commented-out block
    below shows the equivalent circuit-level simulation for the curious reader.
    """
    N = 1 << n
    k = rng.randrange(N)
    phase = np.exp(2j * np.pi * k * s / N)
    vec = np.array([1.0, phase], dtype=complex) / np.sqrt(2)
    return LabeledQubit(k=k, state=Statevector(vec))

    # ----- Equivalent full-circuit version (slow but pedagogical) -----
    # A = QuantumRegister(n, "x"); P = QuantumRegister(1, "b"); B = QuantumRegister(n, "y")
    # cA = ClassicalRegister(n, "cA"); cB = ClassicalRegister(n, "cB")
    # qc = QuantumCircuit(A, P, B, cA, cB)
    # qc.h(A); qc.h(P[0])                      # uniform over D_N
    # qc.append(hiding_oracle(n, s), [*A, P[0], *B])
    # qc.measure(B, cB)                        # collapse to a coset
    # qc.append(QFT(n).to_gate(), A)
    # qc.measure(A, cA)
    # ... run on AerSimulator, read k from cA, extract the parity qubit state ...


# ---------------------------------------------------------------------------
# MODULE 2 -- Combine two labeled qubits via CNOT + measurement
# ---------------------------------------------------------------------------
def combine(q1: LabeledQubit, q2: LabeledQubit, N: int) -> tuple[LabeledQubit, int]:
    """
    Apply CNOT(control=q1, target=q2) and measure q2 in the computational basis.
    Returns (new labeled qubit on the q1 wire, measurement bit b).
        b = 0  ->  label becomes k1 + k2 (mod N)   ('+' branch)
        b = 1  ->  label becomes k1 - k2 (mod N)   ('-' branch)
    """
    # Build the joint 2-qubit state with q1 on qubit 0, q2 on qubit 1.
    # Statevector.expand(other) returns  self ⊗ other  with `other` on the
    # low-index qubit, so q2.state.expand(q1.state) gives q2 ⊗ q1 with
    # q1 at qubit 0 and q2 at qubit 1, which is what we want.
    joint = q2.state.expand(q1.state)

    circ = QuantumCircuit(2)
    circ.cx(0, 1)                       # control = qubit 0 (q1), target = qubit 1 (q2)
    joint = joint.evolve(circ)

    # Measure qubit 1 (the target, which holds q2)
    outcome, post = joint.measure([1])
    b = int(outcome[0])

    # Trace out qubit 1 to recover the single-qubit state on qubit 0
    rho = partial_trace(post, [1])
    # rho should be pure; extract its statevector
    new_state = _purify(rho)

    new_k = (q1.k + q2.k) % N if b == 0 else (q1.k - q2.k) % N
    return LabeledQubit(k=new_k, state=new_state), b


def _purify(rho) -> Statevector:
    """Given a (numerically) pure 1-qubit density matrix, return its statevector."""
    mat = rho.data
    # Take the eigenvector with largest eigenvalue.
    evals, evecs = np.linalg.eigh(mat)
    vec = evecs[:, np.argmax(evals)]
    # Fix global phase so the first nonzero amplitude is real and positive.
    for amp in vec:
        if abs(amp) > 1e-12:
            vec = vec * np.conj(amp) / abs(amp)
            break
    return Statevector(vec)


# ---------------------------------------------------------------------------
# MODULE 3 -- Read the LSB of s from a target-labeled qubit
# ---------------------------------------------------------------------------
def measure_in_hadamard_basis(q: LabeledQubit) -> int:
    """For a qubit with label k = N/2, |phi_k> = (|0> + (-1)^{s_0}|1>)/sqrt2.
       Apply H and measure: outcome 0 -> s_0 = 0, outcome 1 -> s_0 = 1."""
    circ = QuantumCircuit(1)
    circ.h(0)
    post = q.state.evolve(circ)
    outcome, _ = post.measure([0])
    return int(outcome[0])


# ---------------------------------------------------------------------------
# MODULE 4 -- One sieve stage: bucket by low `width` bits, pair, keep '-' branch
# ---------------------------------------------------------------------------
def sieve_stage(pool: list[LabeledQubit], width: int, N: int) -> list[LabeledQubit]:
    buckets: dict[int, list[LabeledQubit]] = {}
    mask = (1 << width) - 1
    for q in pool:
        buckets.setdefault(q.k & mask, []).append(q)

    new_pool: list[LabeledQubit] = []
    for bucket in buckets.values():
        random.shuffle(bucket)
        while len(bucket) >= 2:
            q1 = bucket.pop()
            q2 = bucket.pop()
            q_new, b = combine(q1, q2, N)
            if b == 1:                       # the difference branch zeroes the low bits
                new_pool.append(q_new)
            # else: '+' branch -- low bits doubled, not useful, discard
    return new_pool


# ---------------------------------------------------------------------------
# MODULE 4/5 -- Recover the LSB of the current shift
# ---------------------------------------------------------------------------
def recover_low_bit(n: int, s: int, rng: random.Random,
                    pool_multiplier: float = 8.0,
                    max_attempts: int = 4) -> int:
    """Run the sieve on Z_{2^n} hiding shift s and return s mod 2."""
    N = 1 << n
    if n == 1:
        # Trivial base case: just generate one labeled qubit with k=1 and measure
        # in Hadamard basis.  k is random in {0,1}; resample until k=1.
        for _ in range(64):
            q = generate_labeled_qubit(1, s, rng)
            if q.k == 1:
                return measure_in_hadamard_basis(q)
        raise RuntimeError("Failed to draw k=1 in base case")

    stage_width = max(1, int(round(math.sqrt(n))))
    num_stages  = math.ceil((n - 1) / stage_width)
    target      = N >> 1                          # = 2^{n-1}

    for attempt in range(max_attempts):
        pool_size = int(pool_multiplier * num_stages * (1 << stage_width))
        pool = [generate_labeled_qubit(n, s, rng) for _ in range(pool_size)]

        for stage in range(1, num_stages + 1):
            width = min(stage * stage_width, n - 1)
            pool = sieve_stage(pool, width, N)
            if not pool:
                break

        # Survivors have k in {0, target}.  Find any with k == target.
        survivors = [q for q in pool if q.k == target]
        if survivors:
            return measure_in_hadamard_basis(survivors[0])

        # Otherwise retry with a larger pool
        pool_multiplier *= 2

    raise RuntimeError(f"Sieve failed to produce label {target} after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# MODULE 6 -- Top-level recursion that recovers s bit by bit
# ---------------------------------------------------------------------------
def kuperberg(n: int, s_secret: int, seed: int = 0) -> int:
    """
    Recover the hidden shift s in D_{2^n} bit by bit.
    `s_secret` is passed only because our `generate_labeled_qubit` simulates
    the oracle directly; in a real run we'd carry a *function pointer* that
    halves at every recursive step.  We mimic that halving classically.
    """
    rng = random.Random(seed)
    recovered = 0
    current_s = s_secret
    for bit in range(n):
        m = n - bit
        s_bit = recover_low_bit(m, current_s, rng)
        recovered |= (s_bit << bit)
        # "Reduce": the new shift hidden by the next-level oracle is floor(s/2)
        current_s = current_s >> 1
    return recovered


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n = 5                       # group is D_{16}; keep small so simulation is cheap
    N = 1 << n
    s_true = 0b1011             # = 11
    assert 0 <= s_true < N

    s_found = kuperberg(n, s_true, seed=42)
    print(f"N = {N}")
    print(f"true     secret s = {s_true:0{n}b} ({s_true})")
    print(f"recovered  secret = {s_found:0{n}b} ({s_found})")
    print("SUCCESS" if s_found == s_true else "FAILURE")