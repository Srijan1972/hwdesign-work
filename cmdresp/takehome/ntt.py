"""ntt.py -- the batched NTT-256 accelerator (ML-DSA, q = 8380417), written hook-first.

Modeled on Waveflow's ``examples/stream_inband`` (the streaming polynomial).  In reading order:

1. **Schemas** -- the command header, the response header and the error codes.  Waveflow
   generates their C++ headers and serializers; nothing packs words by hand.
2. **The bit-exact model** -- pure functions with no simulator in them: :func:`ntt_eval` (the
   FIPS 204 Algorithm 41 butterflies on one polynomial) and :func:`ntt_stream_model` (the whole
   protocol over a stream of bursts, TLAST rules and errors included).  The C++ kernel is checked
   against what scenarios.py expects, and so is this model.
3. **The module** -- :class:`NttAccel` declares the ports and the status-only register map and
   names its kernel body (``cpp_body = "body"``).  Waveflow generates the kernel boundary
   (``gen/ntt.hpp``, ``gen/ntt.cpp``); the whole kernel is the hand-written ``ntt_body_impl.tpp``.

The scenarios, their independently computed expected outputs and the checker are in
``scenarios.py``; the build that runs everything is ``ntt_build.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from waveflow.hw.arrayutils import read_array, write_array
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataList, EnumField, IntField
from waveflow.hw.hw_hostactivated import HostActivated
from waveflow.hw.hw_module import HwParam
from waveflow.hw.interface import StreamIFMaster, StreamIFSlave
from waveflow.hw.regmap import Bit, RegAccess, RegField, VitisRegMap, VitisRegMapMMIFSlave
from waveflow.simulation.simobj import ProcessGen
from waveflow.utils.burst_io import StreamBurst

# ---------------------------------------------------------------------------
# The transform's constants
# ---------------------------------------------------------------------------

#: The ML-DSA modulus, 2^23 - 2^13 + 1, and its primitive 512th root of unity (FIPS 204).
Q = 8380417
ZETA = 1753
#: Coefficients per polynomial, and the largest batch one command may carry.
N = 256
MAX_BATCH = 16


def brv8(k: int) -> int:
    """Reverse the 8 bits of ``k``."""
    return int(f"{k:08b}"[::-1], 2)


#: ``ZETAS[m] = zeta^brv8(m) mod q``: the twiddle factor of butterfly group m (m = 1..255).
#: ``ZETAS[0] = 1`` is never used.  ntt_build.py writes the same table into ntt_zetas.h.
ZETAS = np.array([pow(ZETA, brv8(m), Q) for m in range(N)], dtype=np.int64)

# ---------------------------------------------------------------------------
# 1. Schemas
# ---------------------------------------------------------------------------

INCLUDE_DIR = "include"
WORD_BW = 32
WORD_BW_SUPPORTED = [WORD_BW]

#: cmd_type is a plain 8-bit integer, not an EnumField: a Waveflow enum cannot carry a value
#: outside its members, and an unknown opcode is an error the scenarios must be able to send.
CmdTypeField = IntField.specialize(bitwidth=8, signed=False)
TxIdField = IntField.specialize(bitwidth=16, signed=False)
NbatchField = IntField.specialize(bitwidth=8, signed=False)
#: One coefficient per 32-bit word.  The whole word is the value on input (>= q is BAD_COEFF).
Uint32 = IntField.specialize(bitwidth=32, signed=False, include_dir=INCLUDE_DIR)


class NttCmd(IntEnum):
    """The opcodes.  0 is deliberately not a command, so an all-zero word is BAD_CMD, not END."""
    NTT = 1
    END = 2


class NttError(IntEnum):
    """What the kernel reports in its ``error`` register when it halts."""
    NO_ERROR = 0
    TLAST_EARLY = 1     # TLAST before word 255 of a polynomial's input burst
    TLAST_MISSING = 2   # word 255 of a polynomial's input burst has no TLAST
    BAD_PARAM = 3       # nbatch = 0 or nbatch > 16
    BAD_CMD = 4         # cmd_type is neither NTT nor END
    BAD_COEFF = 5       # an input word >= q

NttErrorField = EnumField.specialize(enum_type=NttError, bitwidth=8)


class NttCmdHdr(DataList):
    """Command header: one 32-bit word.  ``END`` sends nbatch = 0, ignored."""
    elements = {
        "cmd_type": {"schema": CmdTypeField, "description": "NTT = 1 or END = 2"},
        "tx_id":    {"schema": TxIdField,    "description": "Command ID: echoed, or reported on error"},
        "nbatch":   {"schema": NbatchField,  "description": "Polynomials in the batch, 1 to 16"},
    }


class NttRespHdr(DataList):
    """Response header: the NTT command's tx_id and nbatch, echoed.  One 32-bit word."""
    elements = {
        "tx_id":  {"schema": TxIdField,   "description": "Echo of the transaction ID"},
        "nbatch": {"schema": NbatchField, "description": "Echo of nbatch: output bursts to follow"},
    }


SCHEMA_CLASSES = [NttErrorField, NttCmdHdr, NttRespHdr]

# ---------------------------------------------------------------------------
# 2. The bit-exact model (pure functions)
# ---------------------------------------------------------------------------


def ntt_eval(w: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """The forward NTT of one polynomial: FIPS 204 Algorithm 41, in place, output in its order.

    ``w`` is 256 coefficients in [0, q).  For len = 128, 64, ..., 1 and m counting 1..255
    across the groups: ``t = zetas[m] * w[j+len] mod q``, ``w[j+len] = w[j] - t mod q``,
    ``w[j] = w[j] + t mod q``.  Every value is the canonical residue in [0, q), so the result is
    the same whatever reduction the hardware uses.  int64 holds every product (< 2^46).
    """
    a = np.asarray(w, dtype=np.int64).copy()
    if a.shape != (N,) or a.min() < 0 or a.max() >= Q:
        raise ValueError("ntt_eval takes 256 coefficients in [0, q)")
    m = 0
    ln = N // 2
    while ln >= 1:
        for start in range(0, N, 2 * ln):
            m += 1
            lo = a[start:start + ln].copy()
            t = (ZETAS[m] * a[start + ln:start + 2 * ln]) % Q
            a[start + ln:start + 2 * ln] = (lo - t) % Q
            a[start:start + ln] = (lo + t) % Q
        ln //= 2
    return a.astype(np.uint32)


@dataclass
class NttStreamResult:
    """What the kernel does to one stream: its output bursts, its status, and the words it read."""

    out: list[StreamBurst]
    halted: int = 0
    error: int = int(NttError.NO_ERROR)
    tx_id: int = 0
    words_read: int = 0

    def status(self) -> dict[str, int]:
        return {"halted": self.halted, "error": self.error, "tx_id": self.tx_id}


def ntt_stream_model(bursts: list[StreamBurst]) -> NttStreamResult:
    """One activation of the kernel, as a pure function of its input stream.

    Word by word, what ``ntt_body_impl.tpp`` does:

    - the status starts clear;
    - read a command header (one word; its TLAST is not inspected).  ``END``: return.  An
      opcode that is neither: ``BAD_CMD``.  ``nbatch`` outside 1..16: ``BAD_PARAM``.  Either
      error is reported before anything is written for the command or any data is read;
    - write the response header, TLAST on it;
    - per polynomial, read up to 256 words.  A word >= q is ``BAD_COEFF`` and a TLAST before
      word 255 is ``TLAST_EARLY`` (``BAD_COEFF`` wins on the same word): either stops at that
      word, with nothing output for this polynomial.  Otherwise the polynomial is transformed and
      written as one 256-word burst with TLAST on its last word, and if word 255 had no TLAST
      the kernel then halts with ``TLAST_MISSING``.

    On any error the kernel sets halted / error / tx_id and reads nothing more.  The output is
    split into bursts at TLAST, the only boundary on the wire.
    """
    words = [(int(w), i + 1 == len(b.words) and b.tlast)
             for b in bursts for i, w in enumerate(np.asarray(b.words, dtype=np.uint64))]
    out: list[tuple[int, bool]] = []
    res = NttStreamResult(out=[])
    pos = 0

    def halt(err: NttError, tx: int) -> None:
        res.halted, res.error, res.tx_id = 1, int(err), int(tx)

    while res.halted == 0:
        if pos >= len(words):
            raise ValueError("the stimulus ends before an END command: the kernel would block")
        hdr = NttCmdHdr().deserialize([words[pos][0]], word_bw=WORD_BW)
        pos += 1
        if hdr.cmd_type == NttCmd.END:
            break
        if hdr.cmd_type != NttCmd.NTT:
            halt(NttError.BAD_CMD, hdr.tx_id)
            break
        if not 1 <= hdr.nbatch <= MAX_BATCH:
            halt(NttError.BAD_PARAM, hdr.tx_id)
            break
        resp = NttRespHdr()
        resp.tx_id, resp.nbatch = hdr.tx_id, hdr.nbatch
        out += [(int(w), True) for w in resp.serialize(word_bw=WORD_BW)]

        for _ in range(int(hdr.nbatch)):
            poly = np.zeros(N, dtype=np.int64)
            err = NttError.NO_ERROR
            for k in range(N):
                if pos >= len(words):
                    raise ValueError("the stimulus ends inside a polynomial: the kernel would block")
                w, last = words[pos]
                pos += 1
                v = int(read_array([w], elem_type=Uint32, word_bw=WORD_BW, shape=1).val[0])
                if v >= Q:
                    err = NttError.BAD_COEFF
                    break
                poly[k] = v
                if last and k < N - 1:
                    err = NttError.TLAST_EARLY
                    break
                if k == N - 1 and not last:
                    err = NttError.TLAST_MISSING
            if err in (NttError.BAD_COEFF, NttError.TLAST_EARLY):
                halt(err, hdr.tx_id)
                break
            y = write_array(ntt_eval(poly), elem_type=Uint32, word_bw=WORD_BW)
            out += [(int(v), i == N - 1) for i, v in enumerate(y)]
            if err == NttError.TLAST_MISSING:
                halt(err, hdr.tx_id)
                break

    res.words_read = pos
    cur: list[int] = []
    for w, last in out:
        cur.append(w)
        if last:
            res.out.append(StreamBurst(np.array(cur, dtype=np.uint64), True))
            cur = []
    if cur:
        res.out.append(StreamBurst(np.array(cur, dtype=np.uint64), False))
    return res


# ---------------------------------------------------------------------------
# 3. The module: ports, register map, and the kernel body
# ---------------------------------------------------------------------------


@dataclass
class NttAccel(HostActivated):
    """The NTT accelerator: a body-only kernel.

    Waveflow generates ``gen/ntt.hpp`` / ``gen/ntt.cpp`` -- the prototype, every interface
    pragma, and one call ``ntt_impl::body(s_in, m_out, halted, error, tx_id)``.  The whole
    kernel is the hand-written ``ntt_body_impl.tpp``.  The register map holds only status;
    everything the kernel computes with arrives on ``s_in``.
    """

    cpp_kernel_name: ClassVar[str | None] = "ntt"
    # Must differ from the kernel function name, or `::ntt(...)` in the testbench is ambiguous.
    cpp_namespace:   ClassVar[str | None] = "ntt_impl"
    cpp_body:        ClassVar[str | None] = "body"

    in_bw:    HwParam[int] = WORD_BW
    out_bw:   HwParam[int] = WORD_BW
    aximm_bw: HwParam[int] = 32
    clk:      Clock = field(default_factory=lambda: Clock(freq=100e6))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in  = StreamIFSlave( name=f'{self.name}_s_in',  sim=self.sim, bitwidth=self.in_bw)
        self.m_out = StreamIFMaster(name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.out_bw)
        self.regmap = VitisRegMap({
            "halted": RegField(Bit,           RegAccess.R, description="1 = halted on error"),
            "error":  RegField(NttErrorField, RegAccess.R, description="Error code (NttError)"),
            "tx_id":  RegField(TxIdField,     RegAccess.R, description="tx_id of the failed command"),
        }, bitwidth=self.aximm_bw)
        self.s_lite = VitisRegMapMMIFSlave(
            name=f'{self.name}_s_lite', sim=self.sim, bitwidth=self.aximm_bw,
            regmap=self.regmap, on_start=self.on_start,
        )
        for ep in (self.s_in, self.m_out, self.s_lite):
            self.add_endpoint(ep)

    def body(self) -> ProcessGen[None]:
        """The kernel hook.  Its C++ is ``ntt_body_impl.tpp``.

        This design is verified through the pure model, C simulation and co-simulation, with no
        pysim run (the course flow needs no Python timing model), so the Python body is not used.
        """
        raise NotImplementedError("NttAccel is verified with ntt_stream_model, csim and cosim; "
                                  "it has no pysim body")
        yield  # pragma: no cover  (makes this a generator, as the hook is)
