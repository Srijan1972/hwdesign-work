"""ntt_build.py -- the build pipeline for the batched NTT-256 accelerator.

    python ntt_build.py --list-steps
    python ntt_build.py --through check_model       # Python only, no Vitis
    python ntt_build.py --through summary           # everything, Vitis included

Modeled on Waveflow's examples/stream_inband/poly_build.py, at one stream width (32 bits) and
with no pysim timing model (the course flow measures timing in co-simulation only).

    scenarios        write each scenario's stimulus and expected response (scenarios.py)
    model_check      ntt_eval against hand-computed examples, the defining formula and the
                     convolution property (model_check.py) -> results/model_check.txt
    checker_selftest the checker rejects deliberately wrong outputs -> results/checker_selftest.txt
    py_model         run the pure bit-exact protocol model on every scenario
    check_model      ...and compare it with the expected response
    layout           the word layout of every header, by serializing instances -> results/layout.md
    gen_include      schema headers, serializers and the stream/testbench helpers (include/)
    zetas            the twiddle-factor ROM, from ntt.ZETAS -> ntt_zetas.h
    gen_kernel       the kernel boundary (gen/ntt.hpp, gen/ntt.cpp); the body is ntt_body_impl.tpp
    csim             Vitis C simulation, every scenario -> results/csim_output.txt
    check_csim       ...compare
    synth_cosim      C synthesis, then ONE RTL co-simulation run through every scenario
                     -> reports/ (csynth.rpt, ntt_cosim.rpt, result.transaction.rpt)
    check_cosim      ...compare
    inspect_synth    loops, II and resources from the synthesis report -> results/resources.json
    cosim_latency    each activation's latency from the cosim transaction report, and the
                     throughput targets -> results/cosim_latency.json
    verification_table  a row per scenario: expected status, each stage's verdict, the cosim
                     status and cycles -> results/verification.md
    summary         every check, the targets and the resources -> results/summary.json

Every check compares against the same expected response, computed from each scenario's intent.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveflow.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from waveflow.build.cli import run_dag_cli
from waveflow.build.hwcodegen_steps import HlsCodegenStep
from waveflow.build.resource_steps import InspectSynthStep
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.hw.arrayutils import ArrayUtilsStep
from waveflow.hw.dataschema import DataSchemaStep
from waveflow.toolchain import toolchain
from waveflow.utils.burst_io import StreamBurst, read_bursts, write_bursts

import scenarios as S
from ntt import (MAX_BATCH, N, Q, SCHEMA_CLASSES, WORD_BW, WORD_BW_SUPPORTED, ZETAS, NttAccel,
                 NttCmd, NttCmdHdr, NttRespHdr, Uint32, ntt_eval, ntt_stream_model)

_SOURCE_DIR = Path(__file__).resolve().parent
TOP = "ntt"
PROJ = "prj_ntt"          # one directory deep, and ignored by the repository's .gitignore

#: The throughput targets of prompt.md: the activation's co-simulation latency, in cycles.
TARGETS = {"nbatch1": 1700, "nbatch16": 27200}


def data_dir(root: Path) -> Path:
    return root / "data"


def _names(root: Path) -> list[str]:
    return (data_dir(root) / "scenarios.txt").read_text(encoding="utf-8").split()


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Python: scenarios, the model and its checks
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ScenariosStep(BuildStep):
    description = "Write every scenario's stimulus and expected response."
    consumes = ["ntt_source", "scenarios_source"]
    produces = {"scenario_list": Path("data/scenarios.txt")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        S.write_scenarios(data_dir(config.root_dir))
        return {"scenario_list": data_dir(config.root_dir) / "scenarios.txt"}


@dataclass(kw_only=True)
class ModelCheckStep(BuildStep):
    description = "Check ntt_eval against hand-computed values, the formula and convolution."
    consumes = ["ntt_source", "model_check_source"]
    produces = {"model_check": Path("results/model_check.txt")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        import model_check
        if model_check.main() != 0:
            raise RuntimeError("model_check failed: see results/model_check.txt")
        return {"model_check": config.root_dir / "results" / "model_check.txt"}


@dataclass(kw_only=True)
class CheckerSelfTestStep(BuildStep):
    """Show that the checker rejects wrong outputs: a checker that passes everything proves nothing.

    Each case takes a scenario's expected response, breaks it the way a plausible wrong kernel
    would, and requires :func:`scenarios.check` to report a problem.
    """
    description = "Show the checker rejecting deliberately wrong outputs."
    consumes = ["scenario_list"]
    produces = {"checker_selftest": Path("results/checker_selftest.txt")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        root = data_dir(config.root_dir)
        sc = S.scenarios()

        def natural_order(name: str) -> tuple[list[StreamBurst], dict]:
            # The right transform, written in natural order (w_hat[brv8(i)] at position i).
            out, st = S.expected(sc[name])
            perm = [int(f"{i:08b}"[::-1], 2) for i in range(N)]
            fixed = [out[0]] + [StreamBurst(np.asarray(b.words)[perm], b.tlast) for b in out[1:]]
            return fixed, st

        def no_halt(name: str) -> tuple[list[StreamBurst], dict]:
            # An error path that keeps going: status left clear.
            out, st = S.expected(sc[name])
            return out, {**st, "halted": 0, "error": 0, "tx_id": 0}

        def open_burst(name: str) -> tuple[list[StreamBurst], dict]:
            # A partial polynomial written and left open at the error.
            out, st = S.expected(sc[name])
            return out + [StreamBurst(np.zeros(99, np.uint64), False)], st

        def drains(name: str) -> tuple[list[StreamBurst], dict]:
            # A kernel that drains its input after the error.
            out, st = S.expected(sc[name])
            return out, {**st, "unread_words": 0}

        def no_mod(name: str) -> tuple[list[StreamBurst], dict]:
            # A reduction that leaves the last butterfly's sum unreduced when it is >= q.
            out, st = S.expected(sc[name])
            b = np.asarray(out[1].words, np.uint64).copy()
            b[0] += Q
            return [out[0], StreamBurst(b, True)] + out[2:], st

        cases = [("natural-order output", "single", natural_order),
                 ("error path does not halt", "tlast_early", no_halt),
                 ("partial polynomial left open after an early TLAST", "tlast_early", open_burst),
                 ("kernel drains its input after BAD_PARAM", "bad_param_17", drains),
                 ("one output not reduced mod q", "edge_values", no_mod)]
        lines, ok = [], True
        for label, name, make in cases:
            out, st = make(name)
            stage = "selftest"
            d = root / name / stage
            write_bursts(out, d)
            (d / "status.json").write_text(json.dumps(st) + "\n", encoding="utf-8")
            problems = S.check(root, stage, [name])[name]
            shutil.rmtree(d)
            rejected = bool(problems)
            ok &= rejected
            lines.append(f"{'REJECTED' if rejected else 'ACCEPTED (bad!)'}  {label} "
                         f"[{name}]: {problems}")
        text = "\n".join(lines) + f"\n\n{'ALL REJECTED' if ok else 'A WRONG OUTPUT PASSED'}\n"
        out_path = config.root_dir / "results" / "checker_selftest.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(text, end="")
        if not ok:
            raise RuntimeError("the checker accepted a wrong output")
        return {"checker_selftest": out_path}


@dataclass(kw_only=True)
class ModelStep(BuildStep):
    description = "Run the pure bit-exact protocol model (ntt_stream_model) on every scenario."
    consumes = ["scenario_list"]
    produces = {"model_done": Path("results/model_done.txt")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        root = config.root_dir
        for name in _names(root):
            d = data_dir(root) / name
            bursts = read_bursts(d / "in")
            res = ntt_stream_model(bursts)
            write_bursts(res.out, d / "model")
            sent = sum(len(b.words) for b in bursts)
            _write_json(d / "model" / "status.json",
                        {**res.status(), "unread_words": sent - res.words_read})
        done = root / "results" / "model_done.txt"
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("done\n", encoding="utf-8")
        return {"model_done": done}


@dataclass(kw_only=True)
class CheckStep(BuildStep):
    """Compare one stage's recorded responses with the expected ones (scenarios.check)."""
    stage: str
    done_artifacts: tuple[str, ...]
    description = "Compare a stage's responses with the expected responses."
    params = {}

    @property
    def consumes(self) -> list:  # type: ignore[override]
        return [*self.done_artifacts, "scenario_list"]

    @property
    def produces(self) -> dict:  # type: ignore[override]
        return {f"check_{self.stage}": Path(f"results/check_{self.stage}.json")}

    def run(self, config: BuildConfig, **_) -> dict:
        root = config.root_dir
        report = S.check(data_dir(root), self.stage, _names(root))
        out = _write_json(root / "results" / f"check_{self.stage}.json",
                          {n: (p or "PASS") for n, p in report.items()})
        failed = {k: v for k, v in report.items() if v}
        for name, problems in report.items():
            print(f"    {self.stage:6s} {name:28s} {'PASS' if not problems else 'FAIL'}")
            for p in problems:
                print(f"        {p}")
        if failed:
            raise RuntimeError(f"{self.stage}: {len(failed)} scenario(s) differ from the "
                               f"expected response: {sorted(failed)}")
        return {f"check_{self.stage}": out}


@dataclass(kw_only=True)
class LayoutStep(BuildStep):
    """Every header's word layout, from serializing instances -- not from reading the schema."""
    description = "Write the word layout of every header by serializing instances."
    consumes = ["ntt_source"]
    produces = {"layout": Path("results/layout.md")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        def fields(cls, **vals):
            h = cls()
            for k, v in vals.items():
                setattr(h, k, v)
            return h.serialize(word_bw=WORD_BW)

        rows = []
        for label, cls, vals, probe in [
            ("CmdHdr (NTT)", NttCmdHdr, {"cmd_type": int(NttCmd.NTT), "tx_id": 0x1234,
                                         "nbatch": 16}, ["cmd_type", "tx_id", "nbatch"]),
            ("CmdHdr (END)", NttCmdHdr, {"cmd_type": int(NttCmd.END), "tx_id": 0, "nbatch": 0},
             []),
            ("RespHdr", NttRespHdr, {"tx_id": 0x1234, "nbatch": 16}, ["tx_id", "nbatch"]),
        ]:
            w = fields(cls, **vals)
            # Each field's bit range: set it to all ones, alone, and see which bits move.
            spans = []
            for f in probe:
                ones = (1 << cls.elements[f]["schema"].bitwidth) - 1
                one = fields(cls, **{f: ones})
                bits = [b for b in range(32) if (int(one[0]) >> b) & 1]
                spans.append(f"`{f}` bits {max(bits)}:{min(bits)}")
            rows.append(f"| {label} | {len(w)} | {', '.join(f'0x{int(x):08X}' for x in w)} | "
                        f"{', '.join(spans) or 'cmd_type = 2 in bits 7:0'} |")
        coeff = S._words(np.array([0, 1, Q - 1, 0xFFFFFFFF], np.uint64))
        text = "\n".join([
            "# Word layouts (32-bit stream words), from serializing instances",
            "",
            "| Message | Words | Example | Fields |",
            "| --- | --- | --- | --- |",
            *rows,
            f"| Coefficient burst | 256 per polynomial | 0, 1, q-1, 0xFFFFFFFF -> "
            f"{', '.join(f'0x{int(x):08X}' for x in coeff)} | one coefficient per word, the "
            f"whole word (uint32); TLAST on word 255 |",
            "",
            "Example values: NTT header with tx_id = 0x1234, nbatch = 16; response header for it.",
            "",
        ])
        out = config.root_dir / "results" / "layout.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return {"layout": out}


# ---------------------------------------------------------------------------
# Vitis: headers, the twiddle ROM, the kernel boundary, csim, csynth + cosim
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HlsGenIncludeStep(BuildStep):
    description = "Generate the schema headers, serializers, and stream/testbench helpers."
    consumes = ["ntt_source"]
    produces = {"include_dir": Path("include")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        inner = BuildDag()
        inner.add(StreamUtilsStep(output_dir="include"))
        for cls in SCHEMA_CLASSES:
            inner.add(DataSchemaStep(cls, word_bw_supported=WORD_BW_SUPPORTED,
                                     include_dir="include"))
        inner.add(ArrayUtilsStep(Uint32, WORD_BW_SUPPORTED))
        failed = [n for n, r in inner.run(config).items() if not r.success]
        if failed:
            raise RuntimeError(f"header generation failed: {failed}")
        return {"include_dir": config.root_dir / "include"}


@dataclass(kw_only=True)
class ZetasStep(BuildStep):
    """The twiddle-factor ROM, written from ``ntt.ZETAS`` so the C++ and Python share one table."""
    description = "Write ntt_zetas.h, the twiddle-factor ROM, from ntt.ZETAS."
    consumes = ["ntt_source"]
    produces = {"zetas_h": Path("ntt_zetas.h")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        vals = [int(v) for v in ZETAS]
        rows = [", ".join(f"{v:7d}" for v in vals[i:i + 8]) for i in range(0, N, 8)]
        text = "\n".join([
            "// ntt_zetas.h -- written by ntt_build.py (step zetas) from ntt.ZETAS.  Do not edit.",
            "// ZETAS[m] = 1753^brv8(m) mod 8380417 (FIPS 204); ZETAS[0] = 1 is never used.",
            "#pragma once",
            "",
            "namespace ntt_impl {",
            f"static const unsigned int ZETAS[{N}] = {{",
            *[f"    {r}," for r in rows],
            "};",
            "}  // namespace ntt_impl",
            "",
        ])
        out = config.root_dir / "ntt_zetas.h"
        out.write_text(text, encoding="utf-8")
        return {"zetas_h": out}


@dataclass(kw_only=True)
class SourcesStep(BuildStep):
    description = "The hand-written body, testbench and Tcl, in place before generation."
    consumes = ["ntt_source", "zetas_h"]
    produces = {"kernel_sources": Path("ntt_body_impl.tpp")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        for f in ("ntt_body_impl.tpp", "ntt_tb.cpp", "run.tcl"):
            if not (config.root_dir / f).exists():
                raise RuntimeError(f"missing hand-written source {f}")
        return {"kernel_sources": config.root_dir / "ntt_body_impl.tpp"}


def _run_vitis(config: BuildConfig, stage: str, log_name: str) -> str:
    """Run run.tcl at ``stage``; keep the console output as evidence in results/<log_name>."""
    env = {"WAVEFLOW_NTT_STAGE": stage}
    log = config.root_dir / "results" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = toolchain.run_vitis_hls(config.root_dir / "run.tcl", work_dir=config.root_dir,
                                         capture_output=True, env=env)
        text = result.stdout or ""
    except subprocess.CalledProcessError as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        log.write_text(text, encoding="utf-8")
        raise RuntimeError(f"Vitis {stage} failed; see {log}\n{text[-3000:]}") from exc
    log.write_text(text, encoding="utf-8")
    return text


@dataclass(kw_only=True)
class CSimStep(BuildStep):
    description = "Vitis C simulation of every scenario, with ntt_tb.cpp."
    consumes = ["ntt_cpp", "ntt_hpp", "ntt_body_impl", "include_dir", "zetas_h", "scenario_list"]
    produces = {"csim_done": Path("results/csim_output.txt")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        root = config.root_dir
        for name in _names(root):
            (data_dir(root) / name / "csim").mkdir(parents=True, exist_ok=True)
        text = _run_vitis(config, "csim", "csim_output.txt")
        if "WAVEFLOW_SUCCESS" not in text:
            raise RuntimeError("csim did not report success; see results/csim_output.txt")
        return {"csim_done": root / "results" / "csim_output.txt"}


@dataclass(kw_only=True)
class SynthCosimStep(BuildStep):
    """C synthesis, then one RTL co-simulation run through every scenario.

    The cited reports are copied out of the Vitis project (which is not committed) into reports/.
    """
    description = "Vitis C synthesis, then one co-simulation run of every scenario."
    consumes = ["ntt_cpp", "ntt_hpp", "ntt_body_impl", "include_dir", "zetas_h", "check_csim"]
    produces = {"report_dir": Path(f"{PROJ}/solution1"),
                "cosim_done": Path("results/cosim_output.txt")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        root = config.root_dir
        for name in _names(root):
            (data_dir(root) / name / "cosim").mkdir(parents=True, exist_ok=True)
        text = _run_vitis(config, "synth", "cosim_output.txt")
        if "WAVEFLOW_SUCCESS" not in text:
            raise RuntimeError("synthesis/cosim did not report success; see "
                               "results/cosim_output.txt")
        sol = root / PROJ / "solution1"
        rep = root / "reports"
        rep.mkdir(exist_ok=True)
        for src in (sol / "syn" / "report" / f"{TOP}_csynth.rpt",
                    sol / "syn" / "report" / "csynth.rpt",
                    sol / "sim" / "report" / f"{TOP}_cosim.rpt",
                    sol / "sim" / "report" / "verilog" / "result.transaction.rpt"):
            if src.exists():
                shutil.copy(src, rep / src.name)
        return {"report_dir": sol, "cosim_done": root / "results" / "cosim_output.txt"}


@dataclass(kw_only=True)
class CosimLatencyStep(BuildStep):
    """Each activation's latency, from the cosim transaction report, and the throughput targets.

    Vitis numbers the kernel calls (transactions) in the order the testbench makes them, which
    is the order of data/scenarios.txt, so transaction k is scenario k.
    """
    description = "Per-activation cosim latency, and the throughput targets."
    consumes = ["cosim_done", "scenario_list"]
    produces = {"cosim_latency": Path("results/cosim_latency.json")}
    params = {}

    def run(self, config: BuildConfig, **_) -> dict:
        root = config.root_dir
        rpt = root / "reports" / "result.transaction.rpt"
        lat = {}
        for line in rpt.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*transaction\s+(\d+):\s+(\d+)", line)
            if m:
                lat[int(m.group(1))] = int(m.group(2))
        names = _names(root)
        if len(lat) != len(names):
            raise RuntimeError(f"{len(lat)} transactions in {rpt}, {len(names)} scenarios")
        per = {n: lat[k] for k, n in enumerate(names)}
        targets = {}
        for name in names:
            meta = json.loads((data_dir(root) / name / "scenario.json").read_text())
            t = meta.get("target")
            if t:
                targets[t] = {"scenario": name, "required_max_cycles": TARGETS[t],
                              "measured_cycles": per[name],
                              "verdict": "PASS" if per[name] <= TARGETS[t] else "FAIL",
                              "source": "reports/result.transaction.rpt"}
        out = _write_json(root / "results" / "cosim_latency.json",
                          {"per_activation_cycles": per, "targets": targets,
                           "source": "reports/result.transaction.rpt (transaction k = "
                                     "scenario k of data/scenarios.txt)"})
        for t, v in targets.items():
            print(f"    target {t}: {v['measured_cycles']} cycles (<= {v['required_max_cycles']}) "
                  f"{v['verdict']}")
        return {"cosim_latency": out}


@dataclass(kw_only=True)
class VerificationTableStep(BuildStep):
    """One row per scenario: what it tests, the expected status, and what each stage produced.

    Written from the files the checks read (``expected_status.json`` and each stage's
    ``status.json``) and the check verdicts, so results.md can cite one readable file.
    """
    description = "Write results/verification.md, a row per scenario and stage."
    consumes = ["check_model", "check_csim", "check_cosim", "cosim_latency"]
    produces = {"verification_table": Path("results/verification.md")}
    params = {}

    def run(self, config: BuildConfig, **art) -> dict:
        from ntt import NttError
        root = config.root_dir
        checks = {s: json.loads(Path(art[f"check_{s}"]).read_text())
                  for s in ("model", "csim", "cosim")}
        lat = json.loads(Path(art["cosim_latency"]).read_text())["per_activation_cycles"]

        def st(d: Path) -> str:
            s = json.loads(d.read_text())
            err = NttError(s["error"]).name
            return (f"{'halted' if s['halted'] else 'ok'} {err} tx={s['tx_id']} "
                    f"unread={s['unread_words']}")

        rows = ["| # | Scenario | Purpose | Expected status | Model | C sim | Co-sim | "
                "Co-sim status | Co-sim cycles |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, name in enumerate(_names(root)):
            d = data_dir(root) / name
            meta = json.loads((d / "scenario.json").read_text())
            verdict = {s: ("PASS" if checks[s][name] == "PASS" else "FAIL")
                       for s in ("model", "csim", "cosim")}
            rows.append(f"| {k} | `{name}` | {meta['purpose']} | {st(d / 'expected_status.json')} "
                        f"| {verdict['model']} | {verdict['csim']} | {verdict['cosim']} | "
                        f"{st(d / 'cosim' / 'status.json')} | {lat[name]} |")
        text = "\n".join([
            "# Verification, per scenario",
            "",
            "Each row is one kernel activation (one ap_start, fresh streams), in run order.  "
            "Expected status comes from the scenario's intent (scenarios.py); each stage is "
            "PASS when its output bursts, TLAST flags, status registers and unread-word count "
            "all equal the expected ones.",
            "",
            "Sources: results/check_model.json, results/check_csim.json, "
            "results/check_cosim.json (verdicts); data/<scenario>/cosim/status.json (co-sim "
            "status, as recorded by ntt_tb.cpp); reports/result.transaction.rpt via "
            "results/cosim_latency.json (cycles).  `unread` = input words the kernel left "
            "unread, counted by the testbench after the activation.",
            "",
            *rows,
            "",
        ])
        out = root / "results" / "verification.md"
        out.write_text(text, encoding="utf-8")
        return {"verification_table": out}


@dataclass(kw_only=True)
class SummaryStep(BuildStep):
    description = "Collect every check, the targets and the resources."
    consumes = ["model_check", "checker_selftest", "check_model", "check_csim", "check_cosim",
                "cosim_latency", "resources", "loop_df", "layout", "verification_table"]
    produces = {"summary": Path("results/summary.json")}
    params = {}

    def run(self, config: BuildConfig, **art) -> dict:
        def load(key):
            return json.loads(Path(art[key]).read_text(encoding="utf-8"))
        summary = {
            "checks": {k.removeprefix("check_"): load(k)
                       for k in ("check_model", "check_csim", "check_cosim")},
            "latency": load("cosim_latency"),
            "resources": load("resources"),
            "loops": Path(art["loop_df"]).read_text(encoding="utf-8"),
        }
        out = _write_json(config.root_dir / "results" / "summary.json", summary)
        return {"summary": out}


def build_ntt_dag() -> BuildDag:
    dag = BuildDag()
    dag.add(SourceStep(artifact="ntt_source", path=_SOURCE_DIR / "ntt.py",
                       description="Schemas, the pure model and the module."))
    dag.add(SourceStep(artifact="scenarios_source", path=_SOURCE_DIR / "scenarios.py",
                       description="The scenarios, their expected responses, and the checker."))
    dag.add(SourceStep(artifact="model_check_source", path=_SOURCE_DIR / "model_check.py",
                       description="The model's independent check."))

    dag.add(ScenariosStep(name="scenarios"))
    dag.add(ModelCheckStep(name="model_check"))
    dag.add(CheckerSelfTestStep(name="checker_selftest"))
    dag.add(ModelStep(name="py_model"))
    dag.add(CheckStep(name="check_model", stage="model", done_artifacts=("model_done",)))
    dag.add(LayoutStep(name="layout"))

    dag.add(HlsGenIncludeStep(name="gen_include"))
    dag.add(ZetasStep(name="zetas"))
    dag.add(SourcesStep(name="sources"))
    dag.add(HlsCodegenStep(name="gen_kernel", comp_class=NttAccel,
                           source_artifact="kernel_sources", output_dir="gen", impl_dir="."))

    dag.add(CSimStep(name="csim"))
    dag.add(CheckStep(name="check_csim", stage="csim", done_artifacts=("csim_done",)))
    dag.add(SynthCosimStep(name="synth_cosim"))
    dag.add(CheckStep(name="check_cosim", stage="cosim", done_artifacts=("cosim_done",)))
    dag.add(InspectSynthStep(
        name="inspect_synth", comp_class=NttAccel, top_name=TOP,
        elaborate_params=("in_bw", "out_bw"), params={"in_bw": WORD_BW, "out_bw": WORD_BW},
        report_artifact="report_dir", seconds_artifact=None,
        output=Path("results/resources.json"), output_artifact="resources",
        require_ii1=False, loop_csv=Path("results/loops.csv"), loop_artifact="loop_df"))
    dag.add(CosimLatencyStep(name="cosim_latency"))
    dag.add(VerificationTableStep(name="verification_table"))
    dag.add(SummaryStep(name="summary"))
    return dag


def main() -> None:
    run_dag_cli(build_ntt_dag, description="Build the batched NTT-256 accelerator.",
                default_through="check_model", root_dir=_SOURCE_DIR)


if __name__ == "__main__":
    sys.path.insert(0, str(_SOURCE_DIR))
    main()
