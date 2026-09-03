# Round 4A.5 Interim Summary: P2 Determinism Investigation

Status: `P2_SAME_PATH_NONDETERMINISM = BLOCKED`

This document records diagnostic evidence only.  It does not change the frozen
P2 placement, numerical contract, `B_pair`, coverage target, or production
execution semantics.  P3 remains `NOT_RUN`.

## Reproduced blocker

In a clean A3 disposable container, frozen P2 still produced non-identical
greedy outputs in the same serving process:

| Prompt | 64-token unique outputs / 10 | Repeat | Prefix |
|---|---:|---|---|
| `v_en_01` | 9 | fail | fail |
| `v_struct_01` | 10 | fail | fail |

The failure also remains with `SGLANG_KT_HYBRID_NO_CPU_STREAM=1`, so it is not
an overlap-only issue.

## First valid stage capture

For a matched teacher-forced `v_en_01` history, all captured Layer 1, 9, and
17 values matched byte-for-byte.  At Layer 26, the input, router IDs/weights,
and NPU partial matched, while the CPU output and merged routed output differed.
This establishes a first observed divergent stage, not a completed cause.

## Layer bisection

Diagnostic placements preserve the original P2 CPU experts.  Each uses ten
64-token `v_en_01` sequential-control repeats.

| CPU layers | Result | Unique hashes |
|---|---|---:|
| `{1}` | exact | 1 |
| `{9}` | exact | 1 |
| `{17}` | P1-established control | 1 |
| `{26}` | exact | 1 |
| `{17,26}` | exact | 1 |
| `{1,17}` | nondeterministic | 5 |
| `{9,17}` | nondeterministic | 3 |

Thus the current minimal known failing sets are `{1,17}` and `{9,17}`.

## A/B results

- Per-wrapper CPUInfer/WorkerPool: four distinct instances were created; full
  P2 remained nondeterministic (8 unique `v_en_01` hashes).
- Private LLAMAFILE scratch and private TP merge-output diagnostic buffers:
  each remained nondeterministic in `{1,17}`.
- CPU worker count changes the `v_en_01` outcome for `{1,17}`:
  1/2/4 workers were exact; 8 workers failed with two hashes; frozen P2 uses
  16 workers.
- Full P2 with one CPU worker made `v_en_01` exact but left `v_struct_01`
  nondeterministic.  It is a localization control, not a fix.

## Review and next work

The branch adds default-off diagnostics and evidence only.  The experimental
isolation switches are not proposed as production fixes.  Next work must
capture actual divergent requests with lightweight C++ tile/write-coverage
instrumentation, separately for `v_en_01` and `v_struct_01`, then prove the
minimal ownership or kernel fix before re-running P2 acceptance gates.

Evidence hashes and commands are indexed in `evidence/README.md`; detailed
findings are in `01_FIRST_DIVERGENCE_CAPTURE.md`, `02_LAYER_BISECTION.md`, and
`03_PARALLEL_EXECUTION_AB.md`.
