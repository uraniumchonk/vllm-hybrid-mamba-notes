# vLLM hybrid Mamba / GDN field notes

Closest stacks that actually stay up, then a lookup table of
combinations that explode. Reasons live in
[FIELD_NOTES.md](FIELD_NOTES.md).

Measured on Qwen3.8-27B-AWQ-INT4 (hybrid GDN), 2× RTX 3090 TP2,
FlashInfer, last updated 2026-08-21.

## Closest-to-good (2026-08-21)

### Fast path with spec decode (recommended)

```
vLLM 0.27.1 (+ DFlash2 patches)
LMCache OFF
prefix cache ON
native KV offload OK (not LMCache)
spec decode = DFlash2 draft (method=dflash, 7 tokens)
mamba-cache-mode=align
block-size 1600
max-num-batched-tokens 2048
```

**DFlash2 + vLLM GPU prefix-cache hit + no LMCache is clean.**
Shared-prefix hits keep sane output. This is still the daily path.

Patches on top of stock 0.27.1 (all unmerged upstream, applied by
hand to site-packages):

- vLLM#52816 + #52883 — DFlash2 drafter (block conv + candidate
  selector) and its unquantized-LM-head guard fix
- vLLM#52460 — DSpark/MRv2 mamba `all`→`align` fallback

### LMCache 0.5.4rc5 + local #4253 (experimental, not daily)

```
vLLM 0.27.1 (+ DFlash2 patches)
LMCache 0.5.4rc5 + local kv_cache_group_edits.py patch
prefix cache ON
spec decode = DFlash2
block-size 1600 / chunk-size 1600
FlashInfer
```

0.5.4rc5 **stops the 0.5.3 SEGV**. A local equivalent of
[LMCache#4253](https://github.com/LMCache/LMCache/pull/4253) stops
the **full multilingual salad** on a single shared-prefix round-trip
(startup fingerprint `mamba-unified-view: 48, subpaged-attention-view: 21`).

**Not daily:** switching between several long agent sessions still
occasionally hits a bad chunk — not full salad, but close
(sticky reasoning / near-garbage). That leftover is a **second
bug** (LMCache MP retrieve / hybrid group alignment). Do not fold
it back into #4253. #4253 itself is still unmerged (zero reviews).

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
| **0.27.1** | **off** | on | **DFlash2** | **Use this for spec decode.** Clean GPU-prefix hit path. |
| **0.25.1** | **0.5.1** | on | **off** | **Use this for no-spec + LMCache.** Stable. |
| 0.27.1 | **0.5.4rc5 + #4253-shaped patch** | on | DFlash2 | Starts. No SEGV. Single-prefix round-trip no longer salads. **Long multi-session switch still near-degrades.** Experimental. |
| 0.27.1 | 0.5.4rc5 stock | on | DFlash2 | Starts. Full salad on LMCache hit (missing `subpaged-attention-view`; FlashInfer 1600→64). |
| 0.25.1 | 0.5.1 | on | MTP on | Starts. Type-bench looks fast. Prefix-hit agent turns can emit zeros / salad at 100% accept. |
| 0.25.1 | off | on | MTP on | Still the EAGLE-on-Mamba scheduler bug. Safer than LMCache, not clean. |
| 0.25.1 | 0.5.1 | **off** | MTP on | MTP can work. You give up prefix TTFT. |
| 0.27.1 | 0.5.1 / 0.5.2 | on | any | **Will not start.** `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` |
| 0.27.1 | 0.5.3 | on | off | Starts, then `cudaMemcpy error 1` and LMCache **SEGV**. |
| 0.27.1 | 0.5.3 | on | MTP on | MTP poisoning is fixed upstream. You still eat the 0.5.3 SEGV. |
| 0.27.1 | off | on | MTP / DSpark | Starts, hit path looks clean in synthetic bench. |
| 0.26.x | 0.5.2 / 0.5.3.dev | on | any | Silent multilingual salad on shared-prefix hits (FlashInfer hybrid). |

Daily: DFlash2 **without** LMCache. 0.5.4rc5 + local #4253 is the
closest “new vLLM + new LMCache + spec” has gotten; the remaining
pit is retrieve-depth, not the SEGV and not the 1600/64 layout.

## Symptoms → which row

| You see | You are in |
|---------|------------|
| First turn fine, next agent / next turn is `000…` or tool-call junk | MTP + prefix hit (0.25.1 row with MTP on) |
| Draft accept 0% **or** 100% and text is trash | Same bug, two faces |
| Startup: mamba tensor list vs Tensor | 0.27.1 + LMCache ≤ 0.5.2 |
| `cudaMemcpy` then SEGV of the cache daemon | 0.27.1 + LMCache 0.5.3 |
| Full multilingual salad on 2nd request, accept → 0% | 0.27.1 + stock 0.5.4rc5, missing `subpaged-attention-view` |
| `KV cache group edits applied: {'mamba-unified-view': 48}` only | Same. Need `subpaged-attention-view` too. |
| Single-prefix hit OK; long session switch is sticky / near-garbage | 0.5.4rc5 + #4253-shaped patch. Second bug. |
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

## Reports (rendered HTML)

Raw HTML in `reports/` renders as plain text on GitHub — open them
via htmlpreview for a rendered view (screenshot-friendly).

| Report | Raw | Rendered |
|--------|-----|----------|
| Spec-drafter comparison: DSpark vs DFlash2 vs MTP (AWQ-INT4, 2026-08-19) | [raw](reports/qwen38_int4_spec_drafter_compare_20260819.html) | [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_int4_spec_drafter_compare_20260819.html) |
| Qwen3.8-27B-FP8 on dual RTX 3090 (2026-08-18) | [raw](reports/qwen38_fp8_forum_post_3090_20260818.html) | [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_fp8_forum_post_3090_20260818.html) |
| Qwen3.8-27B-FP8 on dual RTX 3080 (2026-08-15) | [raw](reports/qwen38_fp8_forum_post.html) | [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_fp8_forum_post.html) |
