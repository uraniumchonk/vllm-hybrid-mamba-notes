# vLLM hybrid Mamba / GDN field notes

Closest stacks that actually stay up, then a lookup table of
combinations that explode. Reasons live in
[FIELD_NOTES.md](FIELD_NOTES.md).

Measured on Qwen3.8-27B-AWQ-INT4 (hybrid GDN), 2× RTX 3090 TP2,
FlashInfer, last updated 2026-08-28.

## Closest-to-good (2026-08-28)

### Fast path: 0.28.0 + DFlash2 + LMCache (recommended)

```
vLLM 0.28.0
  + port of #54165 (DFlash is not EAGLE for mamba last-block)
  + port of #50885 (FlashInfer native FULL decode CUDA graphs)
LMCache 0.5.4rc5 ON (LMCacheMPConnector, kv_both)
prefix cache ON
spec = DFlash2-W4A16-GPTQ, method=dflash, K=7
mamba-cache-mode=align
block-size 1600 / chunk-size 1600 / max-num-batched-tokens 2048
gpu-memory-utilization 0.84 (FULL graphs OOM at 0.88 here)
```

Stock 0.28.0 is **not** enough. Without #54165, LMCache retrieve
is 100% byte salad (vLLM#53505) — this is the 8/21 “second bug”.
Without #50885, FlashInfer+spec silently drops `FULL_AND_PIECEWISE`
→ `PIECEWISE` and code/mixed decode falls ~35–50% vs 0.27.1.

With both ports: MeowChat 360/360 clean; typev2 n=1 matches 0.27.1
(code 268 vs 246 tok/s, mixed 97 vs 91, realjob 165 vs 162).

#52816 / #52883 are **in** 0.28.0. Still need `_dense_kv_rows()`
for packed W4/W8 drafts.

### Older: 0.27.1 DFlash2, LMCache off

```
vLLM 0.27.1 (+ DFlash2 patches)
LMCache OFF
prefix cache ON
native KV offload OK (not LMCache)
spec = DFlash2, K=7
block-size 1600 / max-num-batched-tokens 2048
```

Clean GPU-prefix hit. Use if you cannot port #54165/#50885.

0.27.1 + LMCache 0.5.4rc5 + local #4253 stopped SEGV and the
1600→64 layout salad, but long multi-session switch still
near-degraded. That leftover **is** #53505 / missing #54165.

### Older: no-spec

```
vLLM 0.25.1 + LMCache 0.5.1, MTP OFF, block 800 / batched 1536
```

## Lookup

| vLLM | LMCache | Prefix | Spec | What happens |
|------|---------|--------|------|----------------|
| **0.28.0 + #54165 + #50885** | **0.5.4rc5** | on | **DFlash2 K=7** | **Use this.** Retrieve-ON + agent soak clean; typev2 ≈ 0.27.1. |
| 0.28.0 stock | 0.5.4 / 0.5.5 | on | DFlash2 | **#53505.** Retrieve hit = instant U+FFFD salad. Native offload store = 0. |
| 0.28.0 stock | off | on | DFlash2 | Not salad; FlashInfer spec → PIECEWISE; code/mixed −35–50%. |
| **0.27.1** | **off** | on | **DFlash2** | Previous daily spec path. Clean GPU-prefix hit. |
| **0.25.1** | **0.5.1** | on | **off** | Previous no-spec + LMCache. Stable. |
| 0.27.1 | **0.5.4rc5 + #4253-shaped patch** | on | DFlash2 | No SEGV. Layout salad gone. **Long multi-session still near-degrades** (#53505). |
| 0.27.1 | 0.5.4rc5 stock | on | DFlash2 | Starts. Full salad on LMCache hit (missing `subpaged-attention-view`; FlashInfer 1600→64). |
| 0.25.1 | 0.5.1 | on | MTP on | Starts. Type-bench looks fast. Prefix-hit agent turns can emit zeros / salad at 100% accept. |
| 0.25.1 | off | on | MTP on | Still the EAGLE-on-Mamba scheduler bug. Safer than LMCache, not clean. |
| 0.25.1 | 0.5.1 | **off** | MTP on | MTP can work. You give up prefix TTFT. |
| 0.27.1 | 0.5.1 / 0.5.2 | on | any | **Will not start.** `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` |
| 0.27.1 | 0.5.3 | on | off | Starts, then `cudaMemcpy error 1` and LMCache **SEGV**. |
| 0.27.1 | 0.5.3 | on | MTP on | MTP poisoning is fixed upstream. You still eat the 0.5.3 SEGV. |
| 0.27.1 | off | on | MTP / DSpark | Starts, hit path looks clean in synthetic bench. |
| 0.26.x | 0.5.2 / 0.5.3.dev | on | any | Silent multilingual salad on shared-prefix hits (FlashInfer hybrid). |

Daily: 0.28.0 + #54165 + #50885 + LMCache 0.5.4rc5. Stock 0.28.0
is not that. 0.27.1 without LMCache remains the no-port fallback.

## Symptoms → which row

| You see | You are in |
|---------|------------|
| First turn fine, next agent / next turn is `000…` or tool-call junk | MTP + prefix hit (0.25.1 row with MTP on) |
| Draft accept 0% **or** 100% and text is trash | Same bug, two faces |
| Startup: mamba tensor list vs Tensor | 0.27.1 + LMCache ≤ 0.5.2 |
| `cudaMemcpy` then SEGV of the cache daemon | 0.27.1 + LMCache 0.5.3 |
| Full multilingual salad on 2nd request, accept → 0% | 0.27.1 + stock 0.5.4rc5, missing `subpaged-attention-view` |
| `KV cache group edits applied: {'mamba-unified-view': 48}` only | Same. Need `subpaged-attention-view` too. |
| Single-prefix hit OK; long session switch is sticky / near-garbage | 0.5.4rc5 + #4253 without #54165. This is #53505. |
| Retrieve hit → U+FFFD / mixed-script salad | 0.28.0 DFlash + LMCache without #54165 |
| `FULL_AND_PIECEWISE` … `UNIFORM_SINGLE_TOKEN_DECODE` → PIECEWISE | FlashInfer+spec on sm_86; missing #50885 |
| Unique short prompts at 150 tok/s | Spec decode working. Does **not** prove the hit path is safe. |

Details, versions, benches, and upstream issue links:
[FIELD_NOTES.md](FIELD_NOTES.md).

## Models (HuggingFace)

| Role | Repo | Notes |
|------|------|-------|
| DFlash2 draft | [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | 3.85 GB, arch `DFlash2DraftModel`. Mirror: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| DFlash2 draft (W4A16 GPTQ) | local `Qwen3.8-27B-DFlash2-W4A16-GPTQ` | Packed quant; needs `_dense_kv_rows()` on 0.28.0 |
| DSpark draft | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | 2.72 GB. Fix `architectures` to `Qwen3DSparkModel` in config.json before serving |
| Target (original) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Official FP8 |
| Target (AWQ-INT4, used here) | [cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | Hybrid GDN, 5 shards |
| Target (BF16-INT4 variant) | [cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4) | linear_attn kept full BF16, 28.8 GB |

Related: [z-lab/dflash](https://github.com/z-lab/dflash) (DFlash repo +
eval harness), [DFlash2 blog](https://inco.ai/blog/dflash2/).

## Reports (rendered HTML)

Raw HTML in `reports/` renders as plain text on GitHub — open them
via htmlpreview for a rendered view (screenshot-friendly).

| Report | Raw | Rendered |
|--------|-----|----------|
| Spec-drafter comparison: DSpark vs DFlash2 vs MTP (AWQ-INT4, 2026-08-19) | [raw](reports/qwen38_int4_spec_drafter_compare_20260819.html) | [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_int4_spec_drafter_compare_20260819.html) |
| Qwen3.8-27B-FP8 on dual RTX 3090 (2026-08-18) | [raw](reports/qwen38_fp8_forum_post_3090_20260818.html) | [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_fp8_forum_post_3090_20260818.html) |
| Qwen3.8-27B-FP8 on dual RTX 3080 (2026-08-15) | [raw](reports/qwen38_fp8_forum_post.html) | [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_fp8_forum_post.html) |
