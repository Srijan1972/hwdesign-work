# Verification, per scenario

Each row is one kernel activation (one ap_start, fresh streams), in run order.  Expected status comes from the scenario's intent (scenarios.py); each stage is PASS when its output bursts, TLAST flags, status registers and unread-word count all equal the expected ones.

Sources: results/check_model.json, results/check_csim.json, results/check_cosim.json (verdicts); data/<scenario>/cosim/status.json (co-sim status, as recorded by ntt_tb.cpp); reports/result.transaction.rpt via results/cosim_latency.json (cycles).  `unread` = input words the kernel left unread, counted by the testbench after the activation.

| # | Scenario | Purpose | Expected status | Model | C sim | Co-sim | Co-sim status | Co-sim cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `single` | nbatch = 1; the 1-polynomial throughput target | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 1628 |
| 1 | `batch16` | nbatch = 16 (the limit); the 16-polynomial throughput target | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 25928 |
| 2 | `param_change` | nbatch 3, 1, 16, 2 in one activation | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 35660 |
| 3 | `edge_values` | all q-1, all 0, alternating 0/q-1, and the impulses of the hand-computed examples, in one batch of 8 | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 12968 |
| 4 | `lenient_headers` | command headers without TLAST, and END with nbatch = 200: not errors | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 4872 |
| 5 | `end_only` | END alone: no output, clean status | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 4 |
| 6 | `bad_cmd` | cmd_type 0x7F: BAD_CMD, nothing written or read after it | halted BAD_CMD tx=50 unread=514 | PASS | PASS | PASS | halted BAD_CMD tx=50 unread=514 | 4 |
| 7 | `bad_cmd_recover` | fresh activation after bad_cmd | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 8 | `bad_cmd_zero` | cmd_type 0 (an all-zero opcode) is BAD_CMD, not END | halted BAD_CMD tx=51 unread=514 | PASS | PASS | PASS | halted BAD_CMD tx=51 unread=514 | 4 |
| 9 | `bad_cmd_zero_recover` | fresh activation after bad_cmd_zero | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 10 | `bad_param_0` | nbatch = 0: BAD_PARAM | halted BAD_PARAM tx=60 unread=258 | PASS | PASS | PASS | halted BAD_PARAM tx=60 unread=258 | 4 |
| 11 | `bad_param_0_recover` | fresh activation after bad_param_0 | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 12 | `bad_param_17` | nbatch = 17: BAD_PARAM; its 17 bursts stay unread | halted BAD_PARAM tx=61 unread=4353 | PASS | PASS | PASS | halted BAD_PARAM tx=61 unread=4353 | 4 |
| 13 | `bad_param_17_recover` | fresh activation after bad_param_17 | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 14 | `bad_coeff_q` | word 100 of polynomial 1 is q: BAD_COEFF | halted BAD_COEFF tx=70 unread=412 | PASS | PASS | PASS | halted BAD_COEFF tx=70 unread=412 | 1729 |
| 15 | `bad_coeff_q_recover` | fresh activation after bad_coeff_q | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 16 | `bad_coeff_ffff` | word 255 of polynomial 1 is 0xFFFFFFFF: BAD_COEFF | halted BAD_COEFF tx=71 unread=1 | PASS | PASS | PASS | halted BAD_COEFF tx=71 unread=1 | 1884 |
| 17 | `bad_coeff_ffff_recover` | fresh activation after bad_coeff_ffff | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 18 | `tlast_early` | TLAST on word 99 of polynomial 2 of 3: TLAST_EARLY | halted TLAST_EARLY tx=80 unread=1 | PASS | PASS | PASS | halted TLAST_EARLY tx=80 unread=1 | 3348 |
| 19 | `tlast_early_recover` | fresh activation after tlast_early | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 20 | `tlast_missing` | no TLAST on word 255 of polynomial 1 of 2 | halted TLAST_MISSING tx=90 unread=1 | PASS | PASS | PASS | halted TLAST_MISSING tx=90 unread=1 | 3245 |
| 21 | `tlast_missing_recover` | fresh activation after tlast_missing | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 22 | `tlast_missing_first` | no TLAST on polynomial 0 of 2: polynomial 1 is never read | halted TLAST_MISSING tx=91 unread=257 | PASS | PASS | PASS | halted TLAST_MISSING tx=91 unread=257 | 1625 |
| 23 | `tlast_missing_first_recover` | fresh activation after tlast_missing_first | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 24 | `coeff_and_early` | word 49 is q and carries an early TLAST: BAD_COEFF wins | halted BAD_COEFF tx=100 unread=257 | PASS | PASS | PASS | halted BAD_COEFF tx=100 unread=257 | 58 |
| 25 | `coeff_and_early_recover` | fresh activation after coeff_and_early | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 26 | `coeff_and_missing` | word 255 is >= q and has no TLAST: BAD_COEFF wins, nothing output | halted BAD_COEFF tx=101 unread=1 | PASS | PASS | PASS | halted BAD_COEFF tx=101 unread=1 | 264 |
| 27 | `coeff_and_missing_recover` | fresh activation after coeff_and_missing | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
| 28 | `second_cmd_error` | the second command fails (TLAST_EARLY) after the first succeeded | halted TLAST_EARLY tx=111 unread=1 | PASS | PASS | PASS | halted TLAST_EARLY tx=111 unread=1 | 4882 |
| 29 | `second_cmd_error_recover` | fresh activation after second_cmd_error | ok NO_ERROR tx=0 unread=0 | PASS | PASS | PASS | ok NO_ERROR tx=0 unread=0 | 3250 |
