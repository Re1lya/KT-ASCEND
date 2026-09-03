# Round 4A.5 Evidence

This directory contains compact, reviewable evidence produced by the clean P2
reproduction and subsequent root-cause experiments. Large tensor dumps remain in
the A3 host artifact directory and are referenced by manifest and SHA256.

## Clean reproduction

| File | SHA256 |
|---|---|
| `p2-minimal-same-path.json` | `9d1125747de32ca2914089e3eed5501de43b767c529c082fcf755ed12ae67ce8` |
| `p2-minimal-same-path.log` | `05e29e281393c69dc74f3bb6a3d8ca216cac8f650b51b11236102c0215039f57` |

The JSON payload's internal canonical SHA256 is
`7a4e74ce97b6eab803b084b9eb6259fea33c8f9023cf30aebf80c0e80fe54ef8`.

## First matched-history capture

| File | SHA256 |
|---|---|
| `p2-v-en-01-token1-stage-comparison.json` | `8db80e2ccc25ec54170ca8f24bd2fb4c95e9bf2c03d0c3950f6428bda2f9ddb6` |
| `p2-v-en-01-per-wrapper-cpuinfer-repeats.json` | `43183d8c0bd081738381d1d56f8cc06dc3a108bf0110ea39ec1753afc03b8027` |

The matched-history capture first differs at Layer 26 `cpu_output`, while the
Layer 26 input, router values, and NPU partial are byte-identical.  The
per-wrapper CPUInfer diagnostic created four distinct WorkerPools but retained
same-path nondeterminism (8 unique outputs in 10 repeats); it is evidence
against shared CPUInfer queueing as the sole cause.

## Layer bisection (in progress)

| File | SHA256 | Result |
|---|---|---|
| `p2-bisection-l26-repeat.json` | `f9b6e6791faa710fcda61974bce19e461fdf4a7ff5ee7a5cda70a0086002df64` | exact, 1/10 hashes |
| `p2-bisection-l17-l26-repeat.json` | `1daf63623fd31c32e03da92d629b508a03ed0108bf4a0353c5f01c3bae6c6bfa` | exact, 1/10 hashes |
| `p2-bisection-l9-repeat.json` | `60572bdf01d1d615fe5b488098ea9de51f81cc27a7967215d3e5ccd0584a7b3d` | exact, 1/10 hashes |
| `p2-bisection-l9-l17-repeat.json` | `efb884c0026359cde3b99598a7ae3f7f89e1b73421a2e9ce2b06dafea886588b` | nondeterministic, 3/10 hashes |

The four diagnostic placements preserve frozen P2 expert IDs.  They are not
acceptance placements.  `{9,17}` is the current minimal known failing set;
the bisection remains incomplete.
