# Git Commits

## Child: `third_party/sglang`

```text
8a4bb4c850860377d9880b8c0fa570ca2d0d19e1
fix(moe): preserve fp32 hybrid expert contributions
```

This commit contains the explicitly negotiated FP32 NPU partial contribution,
the one-cast KTEP merge, and default-off, arm-file-gated numerical capture used
to prove the boundary behavior.

## Parent production fixes

```text
427cfb0e8d54375f6dc3821341dc8d3083aede61
fix(llamafile): align bf16 expert numerical boundaries

5064dffff885bc040450f183d5bceea2370f8bbd
fix(moe): preserve configured cpu expert output dtype
```

The commits are ordered by dependency: child first, LLAMAFILE numerical
semantics second, CPU/NPU merge output contract plus child gitlink third.

## Documentation and evidence

The numerical tools, compact JSON evidence and Round 4A1 report are committed
separately from production code. The exact documentation commit is recorded by
the follow-up ledger update after that commit exists.

No Round 4A1 commit has been pushed. A remote push requires a separate explicit
request.
