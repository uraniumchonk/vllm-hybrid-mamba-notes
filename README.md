# vLLM hybrid Mamba / GDN field notes

Closest stack that actually stays up, then a lookup table of
combinations that explode. Reasons live in
[FIELD_NOTES.md](FIELD_NOTES.md).

Measured on Qwen3.8-27B-AWQ-INT4 (hybrid GDN), 2× RTX 3090 TP2,
FlashInfer, 2026-08-19.

## Closest-to-good

```
vLLM 0.25.1
LMCache 0.5.1
prefix cache ON
MTP OFF
mamba-cache-mode=align
block-size 800
max-num-batched-tokens 1536
```

**LMCache or MTP. Pick one.** Both on, plus a prefix-cache hit, is
the perfect storm: later turns / another agent reuse the shared
prefix and output becomes garbage (draft accept can sit at 100%).

If you need MTP, turn prefix cache and LMCache off.
If you need prefix TTFT, leave MTP off.

## Lookup

| vLLM | LMCache | Prefix | MTP | What happens |
|------|---------|--------|-----|----------------|
| **0.25.1** | **0.5.1** | on | **off** | **Use this.** Stable. |
| 0.25.1 | 0.5.1 | on | **on** | Starts. Type-bench looks fast. Prefix-hit agent turns can emit zeros / salad at 100% accept. |
| 0.25.1 | off | on | on | Still the EAGLE-on-Mamba scheduler bug. Safer than LMCache, not clean. |
| 0.25.1 | 0.5.1 | **off** | on | MTP can work. You give up prefix TTFT. |
| 0.27.1 | 0.5.1 / 0.5.2 | on | any | **Will not start.** `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` |
| 0.27.1 | 0.5.3 | on | off | Starts, then `cudaMemcpy error 1` and LMCache **SEGV**. |
| 0.27.1 | 0.5.3 | on | on | MTP poisoning is fixed upstream. You still eat the 0.5.3 SEGV. |
| 0.26.x | 0.5.2 / 0.5.3.dev | on | any | Silent multilingual salad on shared-prefix hits (FlashInfer hybrid). |

No shipping trio of “new vLLM + new LMCache + MTP” is safe on this
class of model yet.

## Symptoms → which row

| You see | You are in |
|---------|------------|
| First turn fine, next agent / next turn is `000…` or tool-call junk | MTP + prefix hit (0.25.1 row with MTP on) |
| Draft accept 0% **or** 100% and text is trash | Same bug, two faces |
| Startup: mamba tensor list vs Tensor | 0.27.1 + LMCache ≤ 0.5.2 |
| `cudaMemcpy` then SEGV of the cache daemon | 0.27.1 + LMCache 0.5.3 |
| Unique short prompts at 150 tok/s | MTP working. Does **not** prove the hit path is safe. |

Details, versions, benches, and upstream issue links:
[FIELD_NOTES.md](FIELD_NOTES.md).
