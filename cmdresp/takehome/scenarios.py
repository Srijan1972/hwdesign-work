"""scenarios.py -- the test scenarios, their expected results, and the checker.

Each scenario is one kernel run (one ``ap_start``), written as a short list of **intents**:
"an NTT command with these polynomials", "one whose third polynomial's burst ends early", "then
END".  From the intents this module writes

- the stimulus, ``data/<scenario>/in`` -- a burst bundle the Python model and the C++
  testbench both read; and
- the expected response, ``data/<scenario>/expected``, plus ``expected_status.json``: the
  status registers, and how many input words the kernel must leave unread.

The expected response is computed from the intent, not by parsing the stimulus, so a model or
kernel that misreads the protocol cannot agree with it by sharing the mistake.  The arithmetic
is :func:`ntt.ntt_eval`, which ``model_check.py`` pins down against hand-computed examples and
the defining formula.

Every error scenario keeps sending after the bad word, as a host with commands in flight would.
The kernel must leave all of it unread (its ``unread_words`` is checked); the host's reset,
which discards it, ends the scenario.  The activation after each error scenario is a fresh one
that must pass: the testbench gives every activation new, empty streams.

:func:`check` compares any run -- the model, csim, cosim -- against the expected results, word
for word and TLAST for TLAST, plus the status registers and the unread-word count.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from ntt import (MAX_BATCH, N, Q, WORD_BW, NttCmd, NttCmdHdr, NttError, NttRespHdr, Uint32,
                     ntt_eval)
except ModuleNotFoundError:  # imported from elsewhere
    from .ntt import (MAX_BATCH, N, Q, WORD_BW, NttCmd, NttCmdHdr, NttError,  # type: ignore
                      NttRespHdr, Uint32, ntt_eval)
from waveflow.hw.arrayutils import write_array
from waveflow.utils.burst_io import StreamBurst, read_bursts, write_bursts


@dataclass(frozen=True)
class Poly:
    """One polynomial's input burst, as the host sends it.

    ``coeffs`` are the 256 words the host means to send (a word >= q is BAD_COEFF).  ``sent``
    is how many it actually sends, and ``tlast`` whether the last word sent carries TLAST:
    ``sent < 256`` with ``tlast`` is an early TLAST, ``tlast=False`` a missing one.
    """

    coeffs: np.ndarray
    sent: int = N
    tlast: bool = True


@dataclass(frozen=True)
class Cmd:
    """One command header and the bursts the host sends behind it.

    ``nbatch`` is the header's field (default: the number of polynomials sent).  For a command
    the kernel rejects, ``polys`` is whatever the host queued anyway.
    """

    tx_id: int
    polys: tuple[Poly, ...]
    cmd_type: int = int(NttCmd.NTT)
    nbatch: int | None = None
    hdr_tlast: bool = True           # header TLAST is not checked (prompt v2): either is valid

    @property
    def hdr_nbatch(self) -> int:
        return len(self.polys) if self.nbatch is None else int(self.nbatch)


@dataclass(frozen=True)
class Scenario:
    """One kernel run: its commands, whether END follows, and its role."""

    cmds: list[Cmd]
    end: bool = True
    end_nbatch: int = 0              # END's nbatch is ignored (prompt v2), so any value is valid
    end_tlast: bool = True
    purpose: str = ""
    target: str | None = None        # the throughput target this activation measures
    after_error: str | None = None   # the error scenario this fresh activation recovers from


def _hdr(cmd_type: int, tx_id: int = 0, nbatch: int = 0) -> np.ndarray:
    h = NttCmdHdr()
    h.cmd_type, h.tx_id, h.nbatch = cmd_type, tx_id, nbatch
    return h.serialize(word_bw=WORD_BW)


def _words(x: np.ndarray) -> np.ndarray:
    return write_array(np.asarray(x, dtype=np.uint64), elem_type=Uint32, word_bw=WORD_BW)


def stimulus(sc: Scenario) -> list[StreamBurst]:
    """The input stream: each command's header and its polynomials' bursts, then END."""
    bursts: list[StreamBurst] = []
    for c in sc.cmds:
        bursts.append(StreamBurst(_hdr(c.cmd_type, c.tx_id, c.hdr_nbatch), c.hdr_tlast))
        for p in c.polys:
            bursts.append(StreamBurst(_words(p.coeffs[:p.sent]), p.tlast))
    if sc.end:
        bursts.append(StreamBurst(_hdr(int(NttCmd.END), 0, sc.end_nbatch), sc.end_tlast))
    return bursts


def expected(sc: Scenario) -> tuple[list[StreamBurst], dict]:
    """The response the prompt promises, from the intent alone.

    Per NTT command: a header error (BAD_CMD, then BAD_PARAM) writes nothing.  Otherwise the
    response header, then per polynomial, in order: if a word sent is >= q, BAD_COEFF at the
    first such word; else if the burst ends early with TLAST, TLAST_EARLY; either way nothing is
    output for that polynomial.  Otherwise its transform, as one closed 256-word burst, and if
    its last word lacked TLAST, TLAST_MISSING after it.  Nothing after an error is answered or
    read.  ``unread_words`` is everything sent minus what the kernel had to read.
    """
    out: list[StreamBurst] = []
    status = {"halted": 0, "error": int(NttError.NO_ERROR), "tx_id": 0}
    read = 0

    def halt(err: NttError, tx: int) -> None:
        status.update(halted=1, error=int(err), tx_id=tx)

    for c in sc.cmds:
        read += 1                                            # the header word
        if c.cmd_type not in (NttCmd.NTT, NttCmd.END):
            halt(NttError.BAD_CMD, c.tx_id)
            break
        if c.cmd_type == NttCmd.END:
            raise ValueError("write END with Scenario.end, not as a Cmd")
        if not 1 <= c.hdr_nbatch <= MAX_BATCH:
            halt(NttError.BAD_PARAM, c.tx_id)
            break
        if len(c.polys) != c.hdr_nbatch:
            raise ValueError(f"tx {c.tx_id}: a valid command sends nbatch polynomials")
        resp = NttRespHdr()
        resp.tx_id, resp.nbatch = c.tx_id, c.hdr_nbatch
        out.append(StreamBurst(resp.serialize(word_bw=WORD_BW)))
        for p in c.polys:
            bad = next((k for k in range(p.sent) if int(p.coeffs[k]) >= Q), None)
            if bad is not None:
                read += bad + 1
                halt(NttError.BAD_COEFF, c.tx_id)
                break
            if p.sent < N:
                if not p.tlast:
                    raise ValueError("a short burst must end in TLAST (or the kernel blocks)")
                read += p.sent
                halt(NttError.TLAST_EARLY, c.tx_id)
                break
            read += N
            out.append(StreamBurst(_words(ntt_eval(p.coeffs))))
            if not p.tlast:
                halt(NttError.TLAST_MISSING, c.tx_id)
                break
        if status["halted"]:
            break
    else:
        if not sc.end:
            raise ValueError("a scenario without END must end in an error, or the kernel blocks")
        read += 1                                            # the END header
    sent = sum(len(b.words) for b in stimulus(sc))
    return out, {**status, "unread_words": sent - read}


def scenarios() -> dict[str, Scenario]:
    """Every scenario, in run order, from fixed seeds."""
    rng = np.random.default_rng(6463)

    def rand() -> np.ndarray:
        return rng.integers(0, Q, N, dtype=np.int64).astype(np.uint64)

    def polys(n: int) -> tuple[Poly, ...]:
        return tuple(Poly(rand()) for _ in range(n))

    def with_word(x: np.ndarray, k: int, v: int) -> np.ndarray:
        y = x.copy()
        y[k] = v
        return y

    def delta(k: int, v: int = 1) -> np.ndarray:
        return with_word(np.zeros(N, np.uint64), k, v)

    tx = iter(range(200, 1000))

    def recover(err_name: str) -> Scenario:
        return Scenario([Cmd(next(tx), polys(2))], purpose=f"fresh activation after {err_name}",
                        after_error=err_name)

    s: dict[str, Scenario] = {}
    # --- well-formed activations -------------------------------------------------------------
    s["single"] = Scenario([Cmd(10, polys(1))], target="nbatch1",
                           purpose="nbatch = 1; the 1-polynomial throughput target")
    s["batch16"] = Scenario([Cmd(20, polys(16))], target="nbatch16",
                            purpose="nbatch = 16 (the limit); the 16-polynomial throughput target")
    s["param_change"] = Scenario([Cmd(31, polys(3)), Cmd(32, polys(1)), Cmd(33, polys(16)),
                                  Cmd(34, polys(2))],
                                 purpose="nbatch 3, 1, 16, 2 in one activation")
    edge = (np.full(N, Q - 1, np.uint64), np.zeros(N, np.uint64),
            np.tile(np.array([0, Q - 1], np.uint64), N // 2), delta(0), delta(0, Q - 1),
            delta(1), delta(128), delta(255))
    s["edge_values"] = Scenario([Cmd(40, tuple(Poly(x) for x in edge))],
                                purpose="all q-1, all 0, alternating 0/q-1, and the impulses "
                                        "of the hand-computed examples, in one batch of 8")
    s["lenient_headers"] = Scenario([Cmd(45, polys(2), hdr_tlast=False),
                                     Cmd(46, polys(1), hdr_tlast=False)],
                                    end_nbatch=200, end_tlast=False,
                                    purpose="command headers without TLAST, and END with "
                                            "nbatch = 200: not errors")
    s["end_only"] = Scenario([], purpose="END alone: no output, clean status")

    # --- one activation per error, each followed by a fresh activation -------------------------
    queued = Cmd(999, polys(1))     # a command the host queued behind the bad one: never read
    errors: dict[str, Scenario] = {
        "bad_cmd": Scenario([Cmd(50, polys(1), cmd_type=0x7F), queued],
                            purpose="cmd_type 0x7F: BAD_CMD, nothing written or read after it"),
        "bad_cmd_zero": Scenario([Cmd(51, polys(1), cmd_type=0), queued],
                                 purpose="cmd_type 0 (an all-zero opcode) is BAD_CMD, not END"),
        "bad_param_0": Scenario([Cmd(60, (), nbatch=0), queued],
                                purpose="nbatch = 0: BAD_PARAM"),
        "bad_param_17": Scenario([Cmd(61, polys(17))],
                                 purpose="nbatch = 17: BAD_PARAM; its 17 bursts stay unread"),
        "bad_coeff_q": Scenario([Cmd(70, (Poly(rand()), Poly(with_word(rand(), 100, Q)),
                                          Poly(rand())))],
                                purpose="word 100 of polynomial 1 is q: BAD_COEFF"),
        "bad_coeff_ffff": Scenario([Cmd(71, (Poly(rand()), Poly(with_word(rand(), 255,
                                                                          0xFFFFFFFF))))],
                                   purpose="word 255 of polynomial 1 is 0xFFFFFFFF: BAD_COEFF"),
        "tlast_early": Scenario([Cmd(80, (Poly(rand()), Poly(rand()), Poly(rand(), sent=100)))],
                                purpose="TLAST on word 99 of polynomial 2 of 3: TLAST_EARLY"),
        "tlast_missing": Scenario([Cmd(90, (Poly(rand()), Poly(rand(), tlast=False)))],
                                  purpose="no TLAST on word 255 of polynomial 1 of 2"),
        "tlast_missing_first": Scenario([Cmd(91, (Poly(rand(), tlast=False), Poly(rand())))],
                                        purpose="no TLAST on polynomial 0 of 2: polynomial 1 "
                                                "is never read"),
        "coeff_and_early": Scenario([Cmd(100, (Poly(with_word(rand(), 49, Q), sent=50),
                                               Poly(rand())))],
                                    purpose="word 49 is q and carries an early TLAST: "
                                            "BAD_COEFF wins"),
        "coeff_and_missing": Scenario([Cmd(101, (Poly(with_word(rand(), 255, Q + 5),
                                                      tlast=False),))],
                                      purpose="word 255 is >= q and has no TLAST: BAD_COEFF "
                                              "wins, nothing output"),
        "second_cmd_error": Scenario([Cmd(110, polys(2)),
                                      Cmd(111, (Poly(rand()), Poly(rand(), sent=10)))],
                                     purpose="the second command fails (TLAST_EARLY) after the "
                                             "first succeeded"),
    }
    for name, sc in errors.items():
        s[name] = sc
        s[f"{name}_recover"] = recover(name)
    return s


def write_scenarios(data_dir: Path) -> list[str]:
    """Write every scenario's stimulus and expected response under ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, sc in scenarios().items():
        d = data_dir / name
        write_bursts(stimulus(sc), d / "in")
        exp, status = expected(sc)
        write_bursts(exp, d / "expected")
        (d / "expected_status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")
        meta = {"purpose": sc.purpose, "target": sc.target, "after_error": sc.after_error,
                "n_cmds": len(sc.cmds), "end": sc.end}
        (d / "scenario.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
        names.append(name)
    (data_dir / "scenarios.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def check(data_dir: Path, stage: str, names: list[str] | None = None) -> dict[str, list[str]]:
    """Compare ``<scenario>/<stage>`` against ``<scenario>/expected``, for each scenario.

    Returns ``{scenario: [problems]}``; an empty list is a pass.  Words, burst boundaries, TLAST
    flags, the status registers and the unread-word count must all match exactly.
    """
    names = names or (data_dir / "scenarios.txt").read_text(encoding="utf-8").split()
    report: dict[str, list[str]] = {}
    for name in names:
        d = data_dir / name
        problems: list[str] = []
        got_dir = d / stage
        if not (got_dir / "words.bin").exists():
            report[name] = [f"no {stage} output in {got_dir}"]
            continue
        want, got = read_bursts(d / "expected"), read_bursts(got_dir)
        want_st = json.loads((d / "expected_status.json").read_text(encoding="utf-8"))
        if want_st["halted"] and got and not got[-1].tlast:
            problems.append("the output burst in progress at the error was not closed with TLAST")
        if len(want) != len(got):
            problems.append(f"{len(got)} bursts, expected {len(want)}")
        for k, (w, g) in enumerate(zip(want, got)):
            if w.tlast != g.tlast:
                problems.append(f"burst {k}: tlast={g.tlast}, expected {w.tlast}")
            if not np.array_equal(np.asarray(w.words, np.uint64), np.asarray(g.words, np.uint64)):
                n = min(len(w.words), len(g.words))
                bad = next((i for i in range(n) if int(w.words[i]) != int(g.words[i])), n)
                problems.append(f"burst {k}: {len(g.words)} words, expected {len(w.words)}; "
                                f"first difference at word {bad}")
        st_path = got_dir / "status.json"
        if st_path.exists():
            got_st = json.loads(st_path.read_text(encoding="utf-8"))
            for key in ("halted", "error", "tx_id", "unread_words"):
                if got_st.get(key) != want_st[key]:
                    problems.append(f"{key} = {got_st.get(key)}, expected {want_st[key]}")
        else:
            problems.append("no status.json")
        report[name] = problems
    return report
