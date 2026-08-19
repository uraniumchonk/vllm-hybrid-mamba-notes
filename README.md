# vLLM hybrid Mamba / GDN field notes

Closest stacks that actually stay up, then a lookup table of
combinations that explode. Reasons live in
[FIELD_NOTES.md](FIELD_NOTES.md).

Measured on Qwen3.8-27B-AWQ-INT4 (hybrid GDN), 2× RTX 3090 TP2,
FlashInfer, 2026-08-19.

## Closest-to-good (2026-08-19 updated)

### Fast path with spec decode (new, recommended)

```
vLLM 0.27.1 (+ DFlash2 patches)
LMCache OFF
prefix cache ON
spec decode = DFlash2 draft (method=dflash, 7 tokens)
mamba-cache-mode=align
block-size 1600
max-num-batched-tokens 2048
```

**DFlash2 + prefix cache hit + no LMCache is clean.** Shared-prefix
hits keep sane output, acceptance stays ~0.5-0.89, prefix hit rate
~70%. This is the first spec-decode combo that survives the hit path
on this hardware.

Patches on top of stock 0.27.1 (all unmerged upstream, applied by
hand to site-packages):

- vLLM#52816 + #52883 — DFlash2 drafter (block conv + candidate
  selector) and its unquantized-LM-head guard fix
- vLLM#52460 — DSpark/MRv2 mamba `all`→`align` fallback (DSpark
  still untested on the hit path here; use it only with align)

### Stable no-spec path

```
vLLM 0.25.1
LMCache 0.5.1
prefix cache ON
MTP OFF
mamba-cache-mode=align
block-size 800
max-num-batched-tokens 1536
```

## Lookup

| vLLM | LMCache | Prefix | Spec | What happens |
|------|---------|--------|------|----------------|
| **0.27.1** | **off** | on | **DFlash2** | **Use this for spec decode.** Clean hit path, acceptance 0.5-0.89. |
| **0.25.1** | **0.5.1** | on | **off** | **Use this for no-spec.** Stable. |
| 0.25.1 | 0.5.1 | on | MTP on | Starts. Type-bench looks fast. Prefix-hit agent turns can emit zeros / salad at 100% accept. |
| 0.25.1 | off | on | MTP on | Still the EAGLE-on-Mamba scheduler bug. Safer than LMCache, not clean. |
| 0.25.1 | 0.5.1 | **off** | MTP on | MTP can work. You give up prefix TTFT. |
| 0.27.1 | 0.5.1 / 0.5.2 | on | any | **Will not start.** `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` |
| 0.27.1 | 0.5.3 | on | off | Starts, then `cudaMemcpy error 1` and LMCache **SEGV**. |
| 0.27.1 | 0.5.3 | on | MTP on | MTP poisoning is fixed upstream. You still eat the 0.5.3 SEGV. |
| 0.27.1 | off | on | MTP / DSpark | Starts, hit path looks clean in synthetic bench. Agent-turn validation pending. |
| 0.26.x | 0.5.2 / 0.5.3.dev | on | any | Silent multilingual salad on shared-prefix hits (FlashInfer hybrid). |

No shipping trio of “new vLLM + new LMCache + spec decode” is safe on
this class of model yet. DFlash2 without LMCache is the closest to
good as of 2026-08-19; LMCache on 0.27.1 still needs a fixed 0.5.3
(LMCache#4155) or newer.

## Symptoms → which row

| You see | You are in |
|---------|------------|
| First turn fine, next agent / next turn is `000…` or tool-call junk | MTP + prefix hit (0.25.1 row with MTP on) |
| Draft accept 0% **or** 100% and text is trash | Same bug, two faces |
| Startup: mamba tensor list vs Tensor | 0.27.1 + LMCache ≤ 0.5.2 |
| `cudaMemcpy` then SEGV of the cache daemon | 0.27.1 + LMCache 0.5.3 |
| Unique short prompts at 150 tok/s | Spec decode working. Does **not** prove the hit path is safe. |

Details, versions, benches, and upstream issue links:
[FIELD_NOTES.md](FIELD_NOTES.md).

## Models (HuggingFace)

| Role | Repo | Notes |
|------|------|-------|
| DFlash2 draft | [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | 3.85 GB, arch `DFlash2DraftModel`. Mirror: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| DSpark draft | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | 2.72 GB. Fix `architectures` to `Qwen3DSparkModel` in config.json before serving |
| Target (original) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Official FP8 |
| Target (AWQ-INT4, used here) | [cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | Hybrid GDN, 5 shards |
| Target (BF16-INT4 variant) | [cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4) | linear_attn kept full BF16, 28.8 GB |

Related: [z-lab/dflash](https://github.com/z-lab/dflash) (DFlash repo +
eval harness), [DFlash2 blog](https://inco.ai/blog/dflash2/).
