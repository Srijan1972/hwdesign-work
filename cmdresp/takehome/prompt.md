# NTT-256: a batched number-theoretic transform for ML-DSA

Build a Vitis HLS kernel in this folder following the Waveflow stream_inband
example (https://sdrangan.github.io/waveflow/docs/examples/stream_inband/) and
the command-response pattern.

## Operation
The kernel computes the forward NTT of polynomials of 256 coefficients over
Z_q, with the fixed modulus q = 8380417 = 2^23 - 2^13 + 1. This is exactly
the NTT of FIPS 204 (ML-DSA), Algorithm 41, with zeta = 1753, a primitive
512th root of unity mod q.

For an input w[0..255], each coefficient in [0, q), the output is

    w_hat[i] = sum_{j=0}^{255} w[j] * zeta^((2*brv8(i) + 1) * j)  mod q,   i = 0..255

where brv8(i) reverses the 8 bits of i. The output is in FIPS 204 order (the
order the in-place butterflies leave it in): the kernel does not reorder it.
Every output is the canonical residue in [0, q). The butterflies are those of
Algorithm 41: for len = 128, 64, ..., 1, and m counting 1..255 across the
groups, t = zetas[m] * w[j+len] mod q, w[j+len] = w[j] - t mod q,
w[j] = w[j] + t mod q, with zetas[m] = zeta^brv8(m) mod q. Because q is fixed,
the 255 twiddle factors are constants inside the kernel; they are not sent on
the stream.

The arithmetic is exact modular arithmetic, so there is no rounding to
specify: any reduction method (Barrett, Montgomery, shift-and-add on the form
of q) is acceptable as long as every output is the canonical residue.

A command carries a batch of nbatch polynomials. Each is transformed on its
own; nothing carries over between the polynomials of a batch, between
commands, or between activations.

## Requirements
- Coefficients are unsigned integers, one per 32-bit stream word. On input
  the whole 32-bit word is the value, and it must be less than q. On output
  the value is in the low 23 bits and the upper 9 bits are zero.
- nbatch is 1 to 16. The kernel must handle a batch of 16 polynomials.
- The stream is 32 bits wide; no other width is needed.
- Throughput: each of the 8 butterfly stages processes one butterfly per
  clock (128 butterflies in about 128 cycles), so the butterflies of one
  polynomial take about 1024 cycles. With 256 cycles to read a polynomial and
  256 to write it, the target is at most 1,700 cycles per polynomial:
  - an activation with one NTT command of nbatch = 16 and END completes in at
    most 27,200 cycles (co-simulation latency of that activation);
  - an activation with one NTT command of nbatch = 1 and END completes in at
    most 1,700 cycles.
  The latency of an activation is Vitis's co-simulation latency of that
  kernel call, ap_start to ap_done (result.transaction.rpt), including the
  END header. "One butterfly per clock" means the butterfly loop is at II = 1
  in the synthesis report. Polynomials may be processed one at a time (read,
  transform, write); overlapping them is not required.
- The butterfly loop must reach II = 1. Each butterfly reads two words and
  writes two, so give each stage its own RAM ports for both reads and both
  writes (for example, the input buffer read and the output buffer written
  directly by each stage's loop, not through one function shared by every
  stage). This is part of the design, not timing closure.

## Commands and parameters
- `NTT`: the header carries cmd_type, tx_id and nbatch. It is followed by
  nbatch input bursts, one per polynomial, each 256 words with TLAST on its
  256th word. The response is a response header echoing tx_id, then nbatch
  output bursts, one per polynomial in input order, each 256 words with TLAST
  on its 256th word.
- `END`: a header alone. The kernel returns with a clear status. END's tx_id
  and nbatch are ignored: an END with any nbatch is not an error.

The host must end every activation with an END header. Input that stops
before END, or before all 256 words of a polynomial have arrived (with no
early TLAST), is outside the host contract: it is not an error the kernel
reports, the kernel waits for the missing words indefinitely, and the host
recovers by resetting the accelerator and its stream path. No scenario sends
such input.

nbatch = 0 and nbatch > 16 are invalid (BAD_PARAM). A cmd_type that is neither
NTT nor END is invalid (BAD_CMD); the value 0 is not a command, so an
all-zero header word is BAD_CMD, not END. Both are checked from the header,
before any data is read and before anything is written for the command.

The command header is a fixed-length message: its TLAST is not checked, and
a header with or without TLAST is accepted. Only the polynomial bursts are
framed by TLAST. The response header is written as soon as the command
header passes its checks, before any of the command's data is read, so a
data error in polynomial 0 leaves the response header alone on the output.

## Errors
Errors go in AXI-Lite status registers (halted, error code, tx_id).  On an
error the kernel closes any output burst it started with TLAST and returns
without reading more input.  Nothing persists between activations: the status
registers are cleared at the start of each one.

The errors, and what the kernel has sent when it reports each one:
- BAD_CMD, BAD_PARAM: detected from the header. Nothing is written for the
  command and its data is never read. BAD_CMD wins over BAD_PARAM.
- BAD_COEFF: an input word of polynomial p has a value >= q. Polynomials
  0..p-1 of the command have been output in full; nothing is output for
  polynomial p. The kernel reads nothing after the offending word.
- TLAST_EARLY: TLAST on word k < 255 of polynomial p's input burst. As for
  BAD_COEFF: polynomials 0..p-1 output in full, nothing for p, nothing read
  after word k.
- TLAST_MISSING: word 255 of polynomial p has no TLAST. Polynomial p is
  complete, so it is transformed and output in full (TLAST on its last
  word); then the kernel halts.
- If one word is both >= q and carries an early TLAST, or is word 255 with
  no TLAST, BAD_COEFF wins.
- tx_id is the tx_id of the command that failed (from its header, for
  BAD_CMD too).

## Verification
Check the Python model independently of itself:
- Hand-computed examples: x = delta_0 (x[0] = 1, rest 0) gives all ones;
  x = (q-1) * delta_0 gives all q-1; x = delta_1 gives w_hat[0] = 1753 and
  w_hat[1] = q - 1753 = 8378664, and in general w_hat[2i+1] = q - w_hat[2i];
  x = delta_128 gives 4808194 (= zeta^128 mod q) for i < 128 and
  q - 4808194 = 3572223 for i >= 128; the zero polynomial gives zeros.
- A reference: the formula above evaluated directly with Python integers, on
  random polynomials, must match exactly. Also check the convolution
  property: NTT(a * b mod (X^256 + 1)) = NTT(a) .* NTT(b) mod q, with the
  negacyclic product computed directly.

The scenarios:
- nbatch = 1, and nbatch = 16 (the limit) with random coefficients.
- nbatch changing between commands in one activation: 3, then 1, then 16,
  then 2.
- edge values: all coefficients q-1, all 0, alternating 0 and q-1, and the
  hand-computed impulses above, inside one batch.
- an activation that is END alone.
- one activation per error: BAD_CMD with an unknown opcode and with opcode
  0; BAD_PARAM with nbatch = 0 and with nbatch = 17; BAD_COEFF with value q,
  and with 0xFFFFFFFF, in a later polynomial of a batch; TLAST_EARLY in
  polynomial 2 of a batch of 3; TLAST_MISSING in polynomial 1 of a batch of
  2, and in polynomial 0 of a batch of 2 (polynomial 1 must stay unread); a
  word that is both >= q and early-TLAST, and word 255 >= q with no TLAST;
  an error in the second command of an activation, after a first command
  succeeded. Each error activation is followed by a fresh activation that
  must pass.
- an END header with a nonzero nbatch, and command headers sent without
  TLAST, in an activation that must pass.
- the two timing activations of the throughput target.

Compare the model, C simulation and co-simulation bit for bit.  Run
co-simulation once: one testbench run that walks through every scenario in
order, calling the kernel once per activation with fresh streams, including
each error activation and the fresh activation after it.

## results.md
Write results.md with these sections:
- Interface: the command and response layouts, how values pack into words,
  the status registers and the error codes.
- Design: the loop structure, where state is kept, and how the throughput
  target is reached.
- Targets: each target above with the required value, the measured value,
  PASS/FAIL and the file it came from (co-simulation for timing; synthesis
  for clock, II and resources).  Always report the latency and throughput
  from co-simulation and the estimated clock and resources from synthesis,
  writing "no target" where this prompt sets none.
- Verification: the model's check against the hand-computed examples and the
  reference; then one row per scenario saying whether the model, C simulation
  and co-simulation agree, and for each error scenario the status it set and
  whether the fresh activation after it passed. Name the source file of each.
- Assumptions: every place this prompt was silent or ambiguous and what you
  chose, apart from interface details already given under Interface.
- Open Items: questions you could not settle, and anything that failed or was
  not done, or "None".
- Run History: one row per run, with the prompt version, what was built or
  changed, and what passed, so that a later run can continue from here.

Every number and PASS/FAIL in Targets and Verification names the file it came
from (a log, a comparison output or a synthesis report), and that file stays in
this folder to be committed: a reviewer opens them to check results.md.

## Revision history
| Version | Change | Why |
| --- | --- | --- |
| 1 | First version | |
| 2 | The butterfly loop must reach II = 1, with each stage given its own RAM ports for its two reads and two writes | Run 1 missed both throughput targets (2,662 and 42,457 cycles) at II = 2: its Open Item 1 traced it to one write port shared by both butterfly outputs, and named the fix |
| 2 | END's tx_id and nbatch are ignored; an END with any nbatch is not an error | Run 1 had to choose (Assumption 1); I kept its choice |
| 2 | Command-header TLAST is not checked; only polynomial bursts are framed | Run 1 had to choose which TLASTs are errors (Assumption 2); I kept its choice |
| 2 | The response header goes out right after the header checks, before any data | Run 1 had to choose what is on the output before a data error in polynomial 0 (Assumption 3); I kept its choice |
| 2 | Latency is Vitis's ap_start-to-ap_done cosim latency, including END; "one butterfly per clock" is II = 1 in synthesis; polynomials may be processed one at a time | Run 1 had to interpret how the throughput targets are measured (Assumptions 4, 5, 7); I kept its reading |
| 2 | Opcode 0 is not a command (BAD_CMD); the stream is 32 bits only | Run 1 chose both (Assumptions 6, 8); I kept them |
| 2 | Added the scenarios run 1 added on its own, and one with END nbatch != 0 and headers without TLAST | Run 1's Assumption 9 added three error scenarios; the new rules above need a passing scenario of their own |
| 3 | The host must end every activation with END; input that stops before END or inside a polynomial is outside the host contract, and the kernel waits for it | Run 2 had to assume the host always sends END and that truncated input blocks the kernel (its Assumption 5); I kept its choice and made it a host requirement |
