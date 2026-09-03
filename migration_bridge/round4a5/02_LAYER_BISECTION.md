# Round 4A.5：P2 layer bisection（阶段性结果）

状态：`IN_PROGRESS`

## 方法

每个诊断 placement 都从冻结 P2 placement 派生：保留被选层的原始 P2 CPU expert IDs，
并将其余 P2 CPU-enabled 层完全归还 NPU。没有重新选择 expert，诊断 placement 不可用作
P2 acceptance。所有下列运行均为 NPU0、TP1、BF16、P2 GGUF、Graph/Deferred/Dynamic/MTP/
Speculative OFF，并采用 `SGLANG_KT_HYBRID_NO_CPU_STREAM=1`。

每项只运行 `v_en_01` 的 10 次 64-token greedy repeat；prefix、numerical contract 和
coverage 不在本诊断步骤重复执行。

## 已完成组合

| CPU layers | CPU experts | Result | Unique hashes |
|---|---:|---|---:|
| `{26}` | 4 | exact repeat | 1 |
| `{17,26}` | 8 | exact repeat | 1 |
| `{9}` | 4 | exact repeat | 1 |
| `{9,17}` | 8 | `SAME_PATH_NONDETERMINISM` | 3 |

因此，当前最小已知 failing set 是 `{9,17}`，但它不是已证明的最小集：`{17}` 由 P1
资格验证稳定，`{9}` 在本轮稳定，而 `{1}`、`{17,1}`、full P2 的诊断 bisection
仍待完成。此结果证明至少一个跨层交互与 failure 相关；不能据此归因于 Layer 9 或 Layer 17
中的任一层，也不能改变最终 P2 placement。

## 下一步

完成 `{1}` 与 `{17,1}`，然后针对 `{9,17}` 添加 Layer-local task generation、CPU output
write coverage 与 poison/zero diagnostics。此前发现的 full-P2 Layer 26 CPU-output 差异仍须
在这个最小 failing set 中重新捕获，确认它是首个差异还是 downstream symptom。
