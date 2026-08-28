# vLLM hybrid Mamba / GDN field notes

The stack that works. Why it failed before, and every combo that
still explodes: [FIELD_NOTES.md](FIELD_NOTES.md).

Qwen3.8-27B-AWQ-INT4 (hybrid GDN), 2× RTX 3090 TP2, FlashInfer.
Last updated 2026-08-28.

## The good stack (2026-08-28)

```
vLLM 0.28.0
  + #54165 (DFlash is not EAGLE for mamba last-block)
  + #50885 (FlashInfer native FULL decode CUDA graphs)
  + _dense_kv_rows() for packed W4/W8 drafts
LMCache 0.5.4rc5  LMCacheMPConnector kv_both
prefix cache ON
spec = DFlash2-W4A16-GPTQ  method=dflash  K=7
mamba-cache-mode=align
block-size 1600 / chunk-size 1600 / max-num-batched-tokens 2048
gpu-memory-utilization 0.94
```

n=1 decode tok/s, same INT4 target, 2×3090. 
DFlash from 2026-08-28 (this stack).

| type | no DFlash | DFlash2 W4A16 K=7 | |
|------|-----------|----------------|--|
| code | 71 | **268** | 3.8× |
| realjob | ~70 | **165** | 2.4× |
| mixed | 70 | **97** | 1.4× |
| prose | 70 | 76 | 1.1× |

**Why this is the goat.** DFlash multiplies decode on structured
output (code, real agent jobs). LMCache is the
other half: GPU KV is only +343k tokens and dies on restart; L2 is
a 1.5 TB disk you can fill. The 20–28k agent system/tool prefix
stores once, then comes back in 300~500 ms. Spec makes tokens cheap;
LMCache makes the prompt free the second time. Together you keep
DFlash speed *and* prefix TTFT across engine restarts.

Do not run stock 0.28.0 with LMCache+DFlash. Do not pass
`--kv-offloading-*` next to LMCacheMPConnector.

## Lookup

| vLLM | LMCache | Spec | Result |
|------|---------|------|--------|
| **0.28.0 + #54165 + #50885** | **0.5.4rc5** | DFlash2 K=7 | **This is it.** |
| 0.28.0 stock | on | DFlash2 | #53505 retrieve salad |
| 0.28.0 stock | off | DFlash2 | PIECEWISE; code/mixed slow |
| 0.27.1 | off | DFlash2 | Previous daily (no retrieve) |
| 0.25.1 | 0.5.1 | off | Previous no-spec |

Full matrix in [FIELD_NOTES.md](FIELD_NOTES.md).

## Models

| Role | Repo |
|------|------|
| Target | [cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) |
| DFlash2 BF16 | [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |
| DFlash2 W4A16 GPTQ | Quant from bf16 to w4a16 use our real chat text for acceptance rate. |
| DSpark | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) |

## Full Performance Reports

| | raw | preview |
|--|-----|---------|
| **The good stack (2026-08-28)** | [html](reports/qwen38_0280_dflash2_lmcache_20260828.html) | [preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_0280_dflash2_lmcache_20260828.html) |
| Drafter compare 2026-08-19 | [html](reports/qwen38_int4_spec_drafter_compare_20260819.html) | [preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_int4_spec_drafter_compare_20260819.html) |
| FP8 3090 | [html](reports/qwen38_fp8_forum_post_3090_20260818.html) | [preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_fp8_forum_post_3090_20260818.html) |
| FP8 3080 | [html](reports/qwen38_fp8_forum_post.html) | [preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/uraniumchonk/vllm-hybrid-mamba-notes/main/reports/qwen38_fp8_forum_post.html) |
