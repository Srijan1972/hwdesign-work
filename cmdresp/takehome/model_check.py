"""model_check.py -- checks ntt_eval against things that are not ntt_eval.

    python model_check.py            # writes results/model_check.txt, exits 1 on a failure

1. **Hand-computed examples** (prompt.md, Verification): impulses whose transforms follow from
   zeta^256 = -1 alone, worked out by hand, not by running any code.
2. **A direct reference**: the defining formula
   ``w_hat[i] = sum_j w[j] * zeta^((2*brv8(i)+1)*j) mod q``, evaluated with Python integers --
   no butterflies, no twiddle table -- on random polynomials.  Must match exactly.
3. **The convolution property**: NTT(a * b mod (X^256 + 1)) = NTT(a) .* NTT(b) mod q, with the
   negacyclic product computed directly (schoolbook, Python integers).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from ntt import N, Q, ZETA, brv8, ntt_eval
except ModuleNotFoundError:  # imported from elsewhere
    from .ntt import N, Q, ZETA, brv8, ntt_eval  # type: ignore[no-redef]

RESULTS = Path(__file__).resolve().parent / "results" / "model_check.txt"


def delta(k: int, v: int = 1) -> list[int]:
    x = [0] * N
    x[k] = v
    return x


def ntt_reference(w: list[int]) -> list[int]:
    """The defining formula, term by term, with Python integers."""
    return [sum(int(w[j]) * pow(ZETA, ((2 * brv8(i) + 1) * j) % 512, Q) for j in range(N)) % Q
            for i in range(N)]


def negacyclic(a: list[int], b: list[int]) -> list[int]:
    """a * b mod (X^256 + 1, q), schoolbook: X^256 = -1 wraps with a sign flip."""
    c = [0] * N
    for i in range(N):
        for j in range(N):
            k = i + j
            if k < N:
                c[k] += a[i] * b[j]
            else:
                c[k - N] -= a[i] * b[j]
    return [v % Q for v in c]


def hand_examples() -> list[tuple[str, list[int], list[int]]]:
    """(name, input, expected), each expected value worked out by hand.

    zeta = 1753 is a primitive 512th root of unity, so zeta^256 = -1 = q - 1, and
    brv8(2i+1) = brv8(2i) + 128 (the low bit of 2i+1 becomes bit 7).
    """
    z128 = 4808194   # zeta^128 mod q, from FIPS 204's table (zetas[1])
    d1_even_odd = [1753, Q - 1753]   # i = 0: zeta^1;  i = 1: zeta^(2*128+1) = -zeta
    return [
        # x = delta_0: every evaluation of the constant polynomial 1 is 1.
        ("delta_0", delta(0), [1] * N),
        # x = (q-1) delta_0: the constant q-1 evaluates to q-1 everywhere.
        ("(q-1)*delta_0", delta(0, Q - 1), [Q - 1] * N),
        # the zero polynomial
        ("zero", [0] * N, [0] * N),
        # x = delta_1 (the polynomial X): w_hat[i] = zeta^(2*brv8(i)+1).  Check i = 0, 1 below,
        # and w_hat[2i+1] = -w_hat[2i] for every i, separately.
        ("delta_1[0:2]", delta(1), d1_even_odd),
        # x = delta_128: w_hat[i] = zeta^(128*(2*brv8(i)+1)) = zeta^128 * (-1)^brv8(i), and
        # brv8(i) is even exactly when i < 128.
        ("delta_128", delta(128), [z128] * 128 + [Q - z128] * 128),
    ]


def main() -> int:
    lines: list[str] = []
    fails = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    lines.append("1. Hand-computed examples (ntt_eval)")
    for name, x, want in hand_examples():
        got = [int(v) for v in ntt_eval(x)]
        if name.endswith("[0:2]"):
            got = got[:2]
        report(name, got == want, f"got[0:4]={got[:4]} want[0:4]={want[:4]}")
    y = [int(v) for v in ntt_eval(delta(1))]
    report("delta_1: w_hat[2i+1] == q - w_hat[2i] for all i",
           all(y[2 * i + 1] == Q - y[2 * i] for i in range(N // 2)))

    lines.append("")
    lines.append("2. Direct reference: the defining formula with Python integers")
    rng = random.Random(2026)
    cases = [("random seed 2026 #%d" % k, [rng.randrange(Q) for _ in range(N)]) for k in range(4)]
    cases += [("all q-1", [Q - 1] * N), ("alternating 0, q-1", [0, Q - 1] * (N // 2)),
              ("delta_255", delta(255))]
    for name, x in cases:
        got = [int(v) for v in ntt_eval(x)]
        want = ntt_reference(x)
        bad = next((i for i in range(N) if got[i] != want[i]), None)
        report(name, bad is None, "" if bad is None else f"first difference at {bad}")

    lines.append("")
    lines.append("3. Convolution property: NTT(a*b mod X^256+1) == NTT(a) .* NTT(b) mod q")
    for k in range(3):
        a = [rng.randrange(Q) for _ in range(N)]
        b = [rng.randrange(Q) for _ in range(N)]
        lhs = [int(v) for v in ntt_eval(negacyclic(a, b))]
        fa, fb = ntt_eval(a).astype(object), ntt_eval(b).astype(object)
        rhs = [int(u) * int(v) % Q for u, v in zip(fa, fb)]
        report(f"random pair #{k}", lhs == rhs)

    lines.append("")
    lines.append(f"{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    text = "\n".join(lines) + "\n"
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if fails == 0 else 1


def test_model_check() -> None:
    """pytest entry point."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
