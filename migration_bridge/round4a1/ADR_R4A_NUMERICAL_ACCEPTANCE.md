# ADR: Round 4A Numerical Acceptance

## Status

`FINAL — BLOCKED_REQUIRES_ACCEPTANCE_DECISION`. P1 remains blocked and P2/P3
are not allowed under the frozen acceptance contract.

## Context

Round 4A1 requires exact P1 tokens and <=0.20 maximum absolute logprob delta
before P2/P3. Two correctable numerical mismatches were demonstrated and fixed:
missing BF16 expert boundaries and premature BF16 rounding of Hybrid partials.
P1 improved from 30/45 to 42/45 exact.

The remaining failure is one first-divergence history shared by three requested
lengths. Source weights, hidden inputs, routes and visible BF16 boundaries are
correct. Real expert comparisons and all 15 non-empty CPU-expert subsets show
sparse, input-dependent BF16 differences after the down GEMM. Some inputs for
the same expert are bitwise exact and others differ in only a few elements.
E6 or E25 can individually trigger the remaining near-tie token flip, while
E36 can either add error or cancel E25 at the decision boundary. This is
consistent with different ARM LLAMAFILE SGEMM and Ascend GMM reduction trees,
not a fixed bad expert or routing error.

## Decision

1. Keep the two minimal, independently verified Case A/B fixes.
2. Do not relax 45/45 or the 0.20 logprob gate.
3. Do not ship the rejected six-finalize/per-route experimental ABI.
4. Do not add a token-specific logit/tie-break patch.
5. Do not copy an Ascend kernel or restore a historical KML backend inside this
   numerical-closure round.
6. Stop before P2/P3 while the residual remains.

## Consequences

- P1 is operational and much closer numerically, but not accepted.
- P2/P3 remain untested rather than being contaminated by a known red base.
- A future explicitly authorized backend investigation may evaluate a pinned
  KML/BLAS or another CPU GEMM implementation in the disposable container. It
  must demonstrate the full 45/45 gate and preserve performance/ownership
  constraints; low stagewise rel-L2 alone is insufficient.

## Rejected alternatives

- acceptance threshold relaxation: violates the task;
- per-token correction: non-general and hides root cause;
- six NPU finalize calls plus per-route CPU ABI: material complexity and no
  observed token benefit;
- executing CPU-owned experts on NPU: invalidates Hybrid placement;
- host KML installation: outside scope and violates environment isolation.
