# Hybrid Mamba/GDN + MTP + prefix cache + LMCache

Field notes from a working home-lab serving stack.
Last updated: 2026-08-28 (vLLM 0.28.0 + #54165 + #50885 + LMCache).

This is not an upstream design doc. It is what we measured, what
broke, and which GitHub issues match. Numbers below are from this
machine unless marked otherwise.

**One-line rule (old):** MTP + prefix-cache hit + LMCache on a hybrid
Mamba/GDN model is the perfect storm. It looks fast until a later
turn or another agent reuses the shared prefix, then output becomes
garbage while draft acceptance can sit at 100%.

**One-line rule (2026-08-28):** vLLM 0.28.0 + DFlash2 + LMCache
retrieve is **clean** after a local port of #54165 (DFlash is not
EAGLE for mamba `last_cache_position`). Decode speed matches 0.27.1
only after a local port of #50885 (FlashInfer native FULL graphs
under spec). Stock 0.28.0 has both bugs. The 8/21 “second bug”
(multi-session near-degrade after #4253) **is** #53505.

**One-line rule (2026-08-21):** LMCache 0.5.4rc5 no longer SEGVs.
Stock 0.5.4rc5 still full-salads without #4253. After #4253, long
multi-session switch still near-degraded — that leftover needed
#54165, not more layout patches.

**One-line rule (2026-08-19):** DFlash2 + GPU prefix-cache hit on
vLLM 0.27.1 **without LMCache** is clean.

---

## 1. Hardware and software we actually ran

| Piece | Value |
|-------|--------|
| GPUs | 2× RTX 3090 24 GB, tensor parallel 2, P2P on (PCIe switch) |
| Extra GPUs | 2× RTX 3080 20 GB (not used by the LLM) |
| Model | Qwen3.8-27B-AWQ-INT4 (HuggingFace architecture `Qwen3_5ForConditionalGeneration`) |
| Layout | Hybrid: ~48 Gated-DeltaNet layers + 16 full-attention layers |
| Official `mamba_ssm_dtype` in `config.json` | `float32` |
| Serving | vLLM OpenAI server behind llama-swap |
| Attention backend | FlashInfer |
| KV connector | `LMCacheMPConnector`, `kv_role=kv_both` |

### Stack that stays up (current)

| Component | Version | Notes |
|-----------|---------|--------|
| vLLM | **0.25.1** | Hybrid align-mode prefix cache works |
| LMCache | **0.5.1** | Multiprocess server, L1 RAM + L2 disk |
| torch | 2.11.0+cu130 | |
| flashinfer | 0.6.13 + jit-cache 0.6.14+cu130 | Set `FLASHINFER_DISABLE_VERSION_CHECK=1` |
| MTP | **off** | Pulled 2026-08-19 after live agent-switch corruption |
| Prefix cache | on | `--enable-prefix-caching` |
| Chunked prefill | on | |
| `mamba-cache-mode` | `align` | Required for GDN + LMCache |
| `--block-size` | 800 | Attention and mamba both resolve to 800 without MTP |
| `--max-num-batched-tokens` | 1536 | Must satisfy `block_size ≤ x < 2 * block_size` when LMCache is on |
| `--kv-cache-dtype` | fp8 | |
| `--mamba-ssm-cache-dtype` | bfloat16 | Overrides official fp32 so block 800 is legal |
| `--max-model-len` | 256k | |
| `--max-num-seqs` | 16 | |

### Stack that stays up with spec decode (new, 2026-08-19)

| Component | Version | Notes |
|-----------|---------|--------|
| vLLM | **0.27.1** + patches | Patched with DFlash2 (#52816+#52883) and DSpark MRv2 fallback (#52460); contains MTP prefix-cache fix #51113 |
| LMCache | **off** | Not used in this combo |
| torch | 2.13.0+cu130 | |
| flashinfer | 0.6.16.post3 + jit-cache | |
| Spec decode | **DFlash2** (method=dflash, 7 tokens) | Draft: `incoai/Qwen3.8-27B-DFlash2` (arch `DFlash2DraftModel`, 3.85 GB) |
| Prefix cache | on | `--enable-prefix-caching` |
| `mamba-cache-mode` | `align` | |
| `--block-size` | 1600 | Spec decode inflates mamba page from 800 to 1600 |
| `--max-num-batched-tokens` | 2048 | Satisfies `1600 ≤ x < 3200` |
| `--kv-cache-dtype` | fp8 | |
| `--mamba-ssm-cache-dtype` | bfloat16 | |
| `--gpu-memory-utilization` | 0.88 | DSpark/DFlash2 draft eats extra VRAM; 0.94 OOMs (marlin workspace) |

Also works in the same stack: MTP (method=mtp, 3 tokens, built-in
head) and DSpark (method=dspark, 7 tokens, `draft_sample_method:
probabilistic`, draft at `RadixArk/Qwen3.8-27B-DSpark` with
architectures fixed to `Qwen3DSparkModel`). MTP/DSpark hit-path
synthetic bench was clean too, but DFlash2 is the one with
acceptance + per-position numbers we trust.

### Stack that stays up: 0.28.0 + LMCache + DFlash2 (2026-08-28)

| Component | Version | Notes |
|-----------|---------|--------|
| vLLM | **0.28.0** + ports | #52816/#52883 in-tree. Local: #54165, #50885, `_dense_kv_rows()` |
| LMCache | **0.5.4rc5** | `LMCacheMPConnector`, `kv_both`, chunk 1600. Do **not** pass `--kv-offloading-*` (0.28 silently overrides the connector) |
| flashinfer | 0.6.16.post3 | Has `q_len_per_req`; 0.28.0 never advertised UNIFORM_BATCH |
| Spec | **DFlash2-W4A16-GPTQ**, K=7 | Packed draft needs `_dense_kv_rows()` |
| `--block-size` / chunk / batched | 1600 / 1600 / 2048 | Force `--mamba-ssm-cache-dtype bfloat16` or block becomes 3200 |
| `--gpu-memory-utilization` | **0.84** | FULL decode graphs OOM at 0.88 on 2×24 GB + 256k |

**#54165:** `use_eagle()` includes dflash, so 0.28 backs `last_cache_position` by one mamba block and marks every offload group eagle. DFlash must not do that. Retrieve hit → 100% salad (#53505). After the port: retrieve-ON CLEAN; MeowChat 360/360; native offload store non-zero. **#48375 is not sufficient** (short cell only).

**#50885:** FlashInfer+spec dropped FULL→PIECEWISE on sm_86. typev2 n=1 ITL 29→44 ms; code 246→129. After the port (FULL + PIECEWISE prefill):

| type | 0.27.1 W4 | 0.28 PIECEWISE W4 | 0.28 FULL+#54165 W4 |
|------|-----------|-------------------|---------------------|
| code | 246 | 129 | **268** |
| mixed | 91 | 59 | **97** |
| prose | 72 | 45 | **76** |
| realjob | 162 | — | **165** |
| chitchat | 62 | — | **67** |

W4 accept: code 80.8%, mixed 24.3%, realjob 34.0%, prose 14.5%, chitchat 12.4%.

### Stack that does **not** stay up (tried 2026-08-18)

| Component | Version | Result |
|-----------|---------|--------|
| vLLM | 0.27.1 | Needs LMCache ≥ 0.5.3 |
| LMCache | 0.5.1 / 0.5.2 | Will not even start on 0.27.1 |
| LMCache | 0.5.3 | Starts, then store-path `cudaMemcpy` errors and a process SEGV |
| MTP | on, 3 tokens | Fast on structured output; corrupts on prefix-hit agent turns |

vLLM 0.27.1 + LMCache 0.5.3 also had a separate FlashInfer / JIT
regression on this box (we rolled the whole stack back). That is
orthogonal to the hybrid-cache bugs below.

### Stack tried 2026-08-21 (LMCache 0.5.4rc5)

| Component | Version | Result |
|-----------|---------|--------|
| vLLM | 0.27.1 + DFlash2 patches | Starts |
| LMCache | **0.5.4rc5** stock | No SEGV. FlashInfer + block 1600 **full-salads** on hit. Startup: `{'mamba-unified-view': 48}` only. Draft accept 15% → **0%** on hit; target text is garbage (not bad drafts getting accepted). |
| LMCache | 0.5.4rc5 + local #4253-shaped `kv_cache_group_edits.py` | Salad gone on single-prefix round-trip. Fingerprint: `{'mamba-unified-view': 48, 'subpaged-attention-view': 21}` (21 = 16 attn + DFlash draft). Same prompt twice: 1389ms / 567ms, both coherent. |
| Same patched combo | long multi-session switch | **Near-degrades** (sticky reasoning, not full salad). Second bug: MP retrieve / hybrid group alignment. Not daily. |

#4253 upstream: still open, zero human reviews, last activity 2026-08-17.
ApostaC could not repro on H200 FlashAttention (no 1600→64 split).
Do not hold #4253 for the leftover retrieve bug.

---

## 2. Feature cheat sheet

| Knob | What it does | Safe with hybrid GDN? |
|------|----------------|------------------------|
| Prefix cache | Hash token blocks, reuse KV | Yes, on 0.25.1 + align mode, **without MTP** |
| LMCache | Offload those blocks to RAM/disk, restore later | Yes on **0.25.1 + 0.5.1**. 0.27.1 needs ≥ 0.5.3 to start; 0.5.3 SEGVs; 0.5.4rc5 lives but FlashInfer+1600 salads until #4253-shaped edit; long-session retrieve still near-degrades. |
| MTP / EAGLE-style spec decode | Draft N tokens from a colocated MTP head, verify | Fast when cold / unique prompt. **Unsafe** with prefix hit + LMCache |
| `mamba-cache-mode=align` | Snapshot recurrent state only at block boundaries | Required for GDN + prefix cache / LMCache |
| `mamba-cache-mode=all` | Snapshot every `i * block_size` | Not the LMCache-validated combo; DSpark+MRv2 also crashes here |
| `--prefix-match-unit N` | Finer-than-block hits (0.27.x) | We saw occasional misalignment with LMCache and removed it |
| KV fp8 | Smaller attention pages | Works. Does **not** cause the MTP salad. |
| Mamba state bf16 vs fp32 | Page size / block-size math | Precision tweak. Does **not** fix poisoned slots. |

### Hybrid block-size constraint (do not skip)

vLLM forces attention page ≥ mamba page. Official GDN temporal state
is fp32, which often pushes the aligned attention block to ~1600.
We force `--mamba-ssm-cache-dtype bfloat16` so `--block-size 800`
sticks.

With LMCache on vLLM 0.25.1:

```
block_size ≤ max_num_batched_tokens < 2 * block_size
```

Practical pick: `2 * block_size - 1`, then snap to a nearby 2^n or
1.5×2^n if your kernels care. We use 1536 for block 800.

Turning MTP **on** grows the conv state (`num_spec` extra steps).
On this model the scheduler then lifts attention `block_size` from
800 to **1600**. `max-num-batched-tokens` must rise with it (we used
2048). Forget that and the engine asserts at startup:

```
In Mamba cache align mode, block_size (1600) must be <= max_num_batched_tokens (1536).
```

LMCache `chunk-size` must be an integer multiple of the vLLM block
size. We keep chunk 1600 (`1600 % 800 == 0`, also fine for 1600).

---

## 3. Compatibility matrix

Read rows as "can we leave this combination running for real traffic".

| vLLM | LMCache | Prefix cache | Spec | Result |
|------|---------|--------------|-----|--------|
| **0.28.0 + #54165 + #50885** | **0.5.4rc5** | on | **DFlash2 K=7** | **Stable (2026-08-28).** Retrieve-ON + 360 agent turns clean. typev2 ≈ 0.27.1. |
| 0.28.0 stock | 0.5.4 / 0.5.5 | on | DFlash2 | **#53505.** First retrieve hit = U+FFFD salad. Native offload store = 0. |
| 0.28.0 stock | off | on | DFlash2 | Not salad; FlashInfer spec → PIECEWISE; code/mixed −35–50%. |
| **0.25.1** | **0.5.1** | on | **off** | **Stable.** Previous no-spec production. |
| **0.27.1** | **off** | on | **DFlash2** | **Stable on the hit path (2026-08-19).** First spec-decode combo that survives shared-prefix reuse. |
| **0.27.1** | **off** | on | MTP / DSpark | Synthetic hit-path bench clean; acceptance numbers not captured for these two (script bug). Use align mode for DSpark. |
| 0.25.1 | 0.5.1 | on | MTP on | Starts. Type-bench looks great. Later prefix-hit turns can emit garbage with **100% draft accept**. |
| 0.25.1 | off | on | MTP on | Still the EAGLE-on-Mamba scheduler bug. Safer than LMCache (no unaligned external tokens) but not clean. |
| 0.25.1 | 0.5.1 | **off** | MTP on | MTP can work. You give up prefix TTFT. |
| 0.25.1 | 0.5.3 | — | — | Pointless: 0.5.3 is for the new mamba page format. |
| 0.27.1 | 0.5.1 / 0.5.2 | on | any | **Hard fail at register_kv_caches.** Adapter still expects `[conv_state, ssm_state]`; 0.27.1 registers one padded tensor. |
| 0.27.1 | 0.5.3 | on | off | Starts. Then `cudaMemcpy error 1` storms and a **SEGV** of the LMCache process. |
| 0.27.1 | 0.5.3 | on | MTP on | MTP poisoning is *fixed* upstream (#51113 / #47861 landed). You still eat the 0.5.3 SEGV and DSpark/MRv2 issues. |
| 0.27.1 | **0.5.4rc5 stock** | on | DFlash2 | Starts. No SEGV. **Full salad** on LMCache hit (FlashInfer 1600 paged as 64; `subpaged-attention-view` absent). |
| 0.27.1 | **0.5.4rc5 + #4253-shaped patch** | on | DFlash2 | Single-prefix hit clean. Long multi-session switch still near-degrades. **Not daily.** |
| 0.26.x | 0.5.2 / 0.5.3.dev | on | any | Silent multilingual salad on shared-prefix hits for hybrid + FlashInfer. [LMCache#4247](https://github.com/LMCache/LMCache/issues/4247). |

Daily is 0.28.0 + #54165 + #50885 + LMCache 0.5.4rc5. Stock 0.28.0
is not that. The 8/21 leftover after #4253 **is** #53505.

---

## 4. Symptom lookup

Match what you see, then jump.

| What you see | When | Likely cause | Do this |
|--------------|------|--------------|---------|
| Engine exits at startup: `expected a Mamba [conv_state, ssm_state] tensor list, got Tensor` | 0.27.1 + LMCache ≤ 0.5.2 | Old adapter vs fused mamba page | Pair 0.27.1 only with ≥ 0.5.3, **or** stay on 0.25.1 + 0.5.1 |
| Engine exits: `block_size (1600) must be <= max_num_batched_tokens` | MTP just enabled | MTP inflates mamba page; block jumped | Raise `--max-num-batched-tokens` to `< 2 * new_block` (e.g. 2048) |
| Engine exits: `max_num_batched_tokens` vs `2 * block_size` ValueError | LMCache + hybrid | Align-mode snapshot invariant | Set `block ≤ batched < 2*block` |
| `cudaMemcpy failed with error code 1` in `lmcache_memcpy_async_d2h`, then **SEGV** of the LMCache daemon | 0.5.3 + hybrid + `kv_both` + engine-driven transfer | Staging pointer treated as the wrong device / invalid async D2H range. Partial-chunk path is lethal. | 0.5.4rc5 stops the SEGV. Do not "tune L1" as the fix. |
| Same `cudaMemcpy error 1` bursts under high concurrency, service stays up | 0.5.1 | Async D2H vs block reuse race | Usually drop-that-chunk only. Watch, don't panic. |
| First turn fine, second turn / other agent is token salad, `<\|tool_call\|>` leak, or a single-digit loop | MTP + prefix hit | Recurrent state snapshot does not match the hash (EAGLE peek-and-drop applied to Mamba) | Disable MTP **or** disable prefix cache/LMCache |
| Draft acceptance **→ 0%**, output garbage | MTP + prefix hit, older write-up | Draft and target disagree after a poisoned restore | Same as above |
| Draft acceptance **→ 100%**, output is all `0` / repeated junk | MTP + prefix hit after a long shared system prompt (agent switch) | Draft and target **agree** because they share the same poisoned GDN state | Same as above. Restart flushes GPU slots; L2 may still hold the bad prefix. |
| Short unique prompts are brilliant (150+ tok/s numeric) | MTP on, cold / anti-cache | MTP is doing its job | Not a counter-example. The bug is on the **hit** path. |
| Thinking eats `max_tokens`, content empty | MTP + long agent turn | Same poisoning, or reasoning loop | Disable MTP for agent workloads |
| TTFT did not improve after a vLLM upgrade | Any | Stale `~/.cache/vllm/modelinfos/` | Delete modelinfos + torch_compile + triton caches |
| OOM killer loops on the LMCache process | L1 54 GB + another fat GPU job | RAM accounting, not a KV bug | Shrink `--l1-size-gb` or don't co-locate |
| Full multilingual salad on 2nd request; draft accept → 0%; no exception | 0.27.1 + stock 0.5.4rc5 + FlashInfer + block 1600 | `_SubpagedAttentionViewEdit` still requires `ndim == 5`; rank-4 fused KV never matches. Attention stays paged at 64. | Local #4253-shaped edit. Startup must show `subpaged-attention-view`. |
| `KV cache group edits applied: {'mamba-unified-view': 48}` only | Same | Attention edit did not fire | Same |
| Single-prefix hit OK; long session switch is sticky / near-garbage | 0.5.4rc5 + #4253-shaped patch | Second bug: MP retrieve / hybrid group alignment (last chunk / ext>0) | Daily: LMCache off. Do not patch `kv_cache_group_edits.py` further. |

Two faces of the same MTP bug:

1. **0% accept** — draft is wrong, target is still somewhat sane.
2. **100% accept + zeros** — both heads read the same bad recurrent
   snapshot, so verify is a green light on garbage.

We hit (2) in production on 2026-08-19 after several healthy
multi-turn sessions, then switching to another agent that reused the
long shared prefix.

---

## 5. Why MTP + cache hit is the perfect storm

### Invariant (align mode)

Mamba/GDN block-table slot `p` must hold the recurrent state **after
exactly `(p + 1) * block_size` tokens**. That snapshot cannot be
rewound like a token-KV block.

### What EAGLE / MTP does to the KV side

Spec decode peeks one extra block, then drops it
(`drop_eagle_block`). That is legal for attention KV.

### What went wrong

vLLM applied the same peek-and-drop to the **Mamba group**. The
draft tower has no mamba layers, so the finder never drops, hit
length overruns, and/or a mid-block prefill chunk is hashed as a
boundary snapshot.

Concurrent prefills share the token budget. One request stops at
e.g. `state@364`. A later chunk crosses the block boundary and
publishes that same slot as `state@1600`. Every later request that
restores that hash silently loads a truncated state.

A **single** request is accidentally safe (low slots are null and
skipped). That is why unit benches and "reply OK" scripts stay
green, and why the first session looks fine.

LMCache makes it worse:

- It is the "unaligned externally computed tokens" path called out
  in [vLLM#51113](https://github.com/vllm-project/vllm/pull/51113).
- A poisoned slot that reaches L2 survives process restart.
- Finer `prefix-match-unit` (0.27.x) multiplies mid-block edges.

### Upstream fixes (vLLM)

| ID | What | In 0.25.1 | In 0.27.1 |
|----|------|-----------|-----------|
| [vLLM#47861](https://github.com/vllm-project/vllm/pull/47861) | Do not EAGLE-drop MambaSpec; align non-final prefill chunks | **Missing** (0.25.1-shaped patch exists, we tried a local backport) | Present |
| [vLLM#51113](https://github.com/vllm-project/vllm/pull/51113) | Same invariant, re-derived after later scheduler refactors; also stops unaligned resume | Missing (function shape differs) | Merged 2026-08-06 |
| [vLLM#52317](https://github.com/vllm-project/vllm/issues/52317) | DSpark + `mamba_cache_mode=all` crashes on Model Runner V2 | N/A (V1) | Open |

A local 0.25.1 backport of #47861 did **not** stop the 100%-accept
zero loop on a real agent-switch prefix. The 0.27.1-only half
(unaligned resume / LMCache external tokens) is still missing.

### Changing dtypes will not save you

We considered `--mamba-ssm-cache-dtype float32` and
`--kv-cache-dtype bf16`.

- Official temporal state is already fp32 in the model config.
  We only overrode it to bf16 for block-size math.
- The failure is a **wrong snapshot bound to a hash**, not rounding
  noise. FP8 / AWQ-INT4 / W8A16 all hit the same MTP+prefix bug
  ([vLLM#36872](https://github.com/vllm-project/vllm/issues/36872)
  and our own 2026-08-15 runs).
- bf16 KV roughly halves token capacity. Costly, off-target.

---

## 6. Why LMCache 0.5.3 dies

0.5.3 (2026-08-05) is the first release that understands vLLM
0.26/0.27's **fused mamba page** (`conv|ssm|pad` in one tensor).
That is why 0.5.1 cannot start on 0.27.1, and why 0.5.3 is the
only candidate if you upgrade vLLM.

The new engine-driven transfer path is also where it dies.

Our 2026-08-18 journal (LMCache 0.5.3, hybrid, `kv_both`):

1. Stores start failing: `RuntimeError: cudaMemcpy failed with error code 1` at `lmcache_memcpy_async_d2h`.
2. Prefetch still works, so the daemon stays up for tens of minutes.
3. Process SIGSEGV (`status=11`). systemd restarts it. vLLM drops
   with it.

Same box on 0.5.1: zero store errors for a full day.

Matching upstream (still **open** as of 2026-08-19):

| Issue / PR | State | What it is |
|------------|-------|------------|
| [LMCache#4155](https://github.com/LMCache/LMCache/issues/4155) | open | Whole server SIGSEGV on **partial-chunk** transfers. Fallback wraps a CUDA staging `data_ptr()` as a CPU tensor. Full loads accidentally work (contiguous + UVA). Partial loads AVX2-copy a device pointer and die. |
| [LMCache#4288](https://github.com/LMCache/LMCache/pull/4288) | open, unmerged | Validate async memcpy host ranges before launch. |
| [LMCache#4575](https://github.com/LMCache/LMCache/pull/4575) | open, unmerged | Rebuild LMCache object pointers on the device they actually live on. |
| [LMCache#4154](https://github.com/LMCache/LMCache/issues/4154) | closed | Native transfer kernels `CUDA invalid argument` on mixed-format KV (MiniMax-M3). Fallback "works" and then hits #4155. |
| [LMCache#4247](https://github.com/LMCache/LMCache/issues/4247) | open | Silent salad on shared-prefix hybrid under vLLM 0.26 + FlashInfer. |
| [LMCache#4253](https://github.com/LMCache/LMCache/pull/4253) | open | Fix for #4247 (fused KV layout in group edits). Comment on 2026-08-17 still asking for a merge. |

0.5.4 is only rc1–rc4. Nightlies exist. Neither is a tagged
"this SEGV is gone" release. Until #4155 lands in a stable tag,
**do not** pair 0.27.1 with 0.5.3 in production.

---

## 7. Numbers (2026-08-19, 0.25.1, INT4, TP2 3090)

MTP **on**, 3 speculative tokens, same prompts as 2026-08-18.
Anti-cache nonce. `max_tokens=256`.

| type | n | AGG tok/s | avg decode | MTP accept |
|------|--:|----------:|-----------:|-----------:|
| numeric | 1 | 148.5 | 156.4 | 99.2% |
| numeric | 8 | 730.1 | 106.6 | 99.2% |
| code | 1 | 126.3 | 143.1 | 87.8% |
| code | 8 | 522.4 | 100.2 | 87.8% |
| prose | 1 | 89.2 | 92.3 | 46.0% |
| prose | 8 | 432.6 | 63.2 | 46.0% |
| mixed | 1 | 93.6 | 97.8 | 54.3% |
| mixed | 8 | 478.2 | 70.8 | 54.3% |

Within a few percent of the 2026-08-18 MTP run. Structured output
loves MTP. Natural-language prose does not. These benches use
**unique** prefixes, so they do **not** exercise the poison path.

Prefill, MTP still on, unique prefix, first call = cold, second =
hot (LMCache + GPU prefix):

| prompt tokens | cold TTFT | cold tok/s | hot TTFT | hot tok/s |
|--------------:|----------:|-----------:|---------:|----------:|
| 648 | 0.373 s | 1737 | 0.372 s | 1740 |
| 2397 | 1.274 s | 1881 | 0.453 s | 5287 |
| 9358 | 4.987 s | 1877 | 0.796 s | 11752 |
| 18801 | 10.323 s | 1821 | 0.759 s | 24781 |
| 37525 | 22.437 s | 1672 | 0.576 s | 65160 |

648 tok hot == cold because attention block was 1600 under MTP:
nothing below one block can hit. 2k+ hot TTFT is real cache, not
magic FLOPs. Do not calibrate and then call the next request
"cold" — the calibrate write is already in L1/L2.

Without MTP, decode on this box sits around 65 tok/s class for
general traffic. MTP's 95–156 tok/s is real. We still pulled it,
because agent prefix-hits are the default workload here.

## 7b. Numbers (2026-08-19, 0.27.1, INT4, TP2 3090, DFlash2, no LMCache)

DFlash2 drafter (7 draft tokens) on the same AWQ-INT4 target, prefix
cache ON, align mode, block 1600. Two runs: full prefix-hit repro
(8 shared-prefix turns + 4 unique-prefix control) and a 5-turn
acceptance-only run.

| Metric | Value |
|--------|-------|
| Hit-path garbage turns | 0 / 8 shared + 0 / 4 control |
| Acceptance (5-turn run) | 0.50 / 0.886 / 0.518 / 0.857 / 0.886, mean **0.729** |
| Overall (full run, cumulative counters) | accepted 349 / drafted 644 = **54.2%** |
| Per-position acceptance (pos 0..6) | 0.75 / 0.63 / 0.58 / 0.55 / 0.49 / 0.43 / 0.36 |
| Prefix cache hit rate | 101,333 queries / 70,400 hits = **69.5%** |
| Prefix tokens served from cache | 70,400 |

Per-position tail (0.36 at pos 6) is the DFlash2 signature: smooth
decay vs DSpark's collapse (0.76/0.45/0.31/0.14/0.10/0.02/0.00) and
matches z-lab's published tables. Output on the shared-prefix hit
path stayed in the requested [Q][A][C] format on every turn — no
salad, no tool-call leak, no loops.

Caveat: MTP and DSpark got the same clean garbage verdict in the
full repro, but their acceptance counters were lost to a metrics
parse bug in the harness; only DFlash2 has numbers. Agent-turn
validation with real traffic was ongoing at the time of writing.

---

## 8. What we recommend

**Run today**

No-spec production (stable since 2026-08-13):

```
vLLM 0.25.1
LMCache 0.5.1
prefix cache ON
MTP OFF
mamba-cache-mode=align
block-size 800
mamba-ssm-cache-dtype bfloat16
kv-cache-dtype fp8
max-num-batched-tokens 1536
LMCache chunk-size 1600
```

Spec-decode path (daily, 2026-08-19 — DFlash2 + GPU prefix hit is clean):

```
vLLM 0.27.1 + DFlash2 patches (#52816+#52883) + #52460
LMCache OFF
prefix cache ON
speculative-config '{"method":"dflash",
  "model":"<DFlash2 draft path>","num_speculative_tokens":7}'
mamba-cache-mode=align
block-size 1600
max-num-batched-tokens 2048
kv-cache-dtype fp8
mamba-ssm-cache-dtype bfloat16
gpu-memory-utilization 0.88
```

Native vLLM KV offload is fine. LMCache is not required for this path.

**If you want LMCache on 0.27.1 anyway**

- 0.5.4rc5: lives (no SEGV). Stock still salads on FlashInfer+1600.
- Hand-apply #4253-shaped `kv_cache_group_edits.py` (accept rank-4 fused
  KV; fail loud if attention is still kernel-paged). Confirm
  `subpaged-attention-view` in the startup log.
- Expect leftover near-degradation on long multi-session switch.
  That is a second bug. Do not stack more edits on that file.

Clear `~/.cache/vllm/modelinfos/`, torch compile cache, and Triton
cache after any vLLM upgrade.

**If you need MTP**

Pick one:

- On 0.28.0: DFlash2 + LMCache **with #54165 and #50885**. Do not
  run stock 0.28.0 retrieve-ON. Do not pair native kv-offload with
  LMCacheMPConnector.
- MTP **or** prefix cache / LMCache, not both, on anything older.
- On 0.27.1 without LMCache, DFlash2 + prefix hit is clean.
- Wait for #54165 / #50885 to merge before dropping the local ports.

**If you already poisoned a process**

Restart vLLM to drop GPU slots. If LMCache L2 stored that prefix,
delete those keys or wipe L2, or every future hit will restore the
same trash.

**Do not**

- Jump 0.25.1 → 0.27.1 just to "get the MTP fix" while staying on
  LMCache 0.5.1 (it will not start).
- Jump LMCache 0.5.1 → 0.5.3 without a SEGV plan. 0.5.4rc5 is the
  SEGV fix; it is not the salad fix.
- Treat a green type-bench as proof MTP is safe for agents.
- Expect fp32 mamba / bf16 KV to fix hash-level poisoning.
- Leave `--prefix-match-unit` on 0.27.x + LMCache without a
  dedicated misalignment test.

---

## 9. Upstream index

vLLM:

- [vLLM#43559](https://github.com/vllm-project/vllm/issues/43559) — accuracy drop with prefix cache + MTP on Qwen3.6
- [vLLM#47087](https://github.com/vllm-project/vllm/issues/47087) — MTP garbage loops on deep agent chats
- [vLLM#47194](https://github.com/vllm-project/vllm/issues/47194) — tool-call leakage + recall fail, hybrid + MTP3
- [vLLM#47861](https://github.com/vllm-project/vllm/pull/47861) — MTP prefix-cache correctness for hybrid Mamba (closed unmerged; scheduler half landed via #51113)
- [vLLM#51113](https://github.com/vllm-project/vllm/pull/51113) — keep align prefill chunks block-aligned past `last_cache_position` (in 0.27.0+)
- [vLLM#52317](https://github.com/vllm-project/vllm/issues/52317) — DSpark + `mamba_cache_mode=all` MRv2 crash (workaround: align mode)
- [vLLM#52460](https://github.com/vllm-project/vllm/pull/52460) — MRv2 `all`→`align` fallback (unmerged; we hand-patched it)
- [vLLM#52816](https://github.com/vllm-project/vllm/pull/52816) — DFlash2 drafter (**in 0.28.0**)
- [vLLM#52883](https://github.com/vllm-project/vllm/pull/52883) — DFlash2 unquantized LM head guard (**in 0.28.0**)
- [vLLM#53505](https://github.com/vllm-project/vllm/issues/53505) — hybrid mamba + spec + KV connector corruption
- [vLLM#54165](https://github.com/vllm-project/vllm/pull/54165) — DFlash is not EAGLE for mamba last-block / offload groups (OPEN; we ported)
- [vLLM#50885](https://github.com/vllm-project/vllm/pull/50885) — FlashInfer native FULL decode CUDA graphs under spec (OPEN; we ported)
- [vLLM#48375](https://github.com/vllm-project/vllm/pull/48375) — MambaManager `drop_eagle_block` (OPEN; **not** sufficient for #53505)

DFlash2 model + notes:

- [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) — DFlash 2 draft model (mirror: z-lab)
- [z-lab/dflash](https://github.com/z-lab/dflash) — DFlash repo, eval harness, acceptance tables

LMCache:

- [LMCache#4155](https://github.com/LMCache/LMCache/issues/4155) — server SIGSEGV on partial-chunk transfer
- [LMCache#4247](https://github.com/LMCache/LMCache/issues/4247) — silent hybrid corruption on shared-prefix hits. Two bugs in one thread: (1) missing rank-4 subpaged edit, (2) leftover retrieve-depth / multi-session near-degrade. We posted both on this issue.
- [LMCache#4253](https://github.com/LMCache/LMCache/pull/4253) — fused KV layout group-edit fix. **Open, zero reviews, last ping 2026-08-17.** Necessary for FlashInfer 1600→64. Not sufficient for long-session retrieve.
- [LMCache#4288](https://github.com/LMCache/LMCache/pull/4288) — validate async memcpy host ranges
- [LMCache#4600](https://github.com/LMCache/LMCache/pull/4600) — mark failed MP retrieves as load errors (fail closed; still open)

---

## 10. How to reproduce the MTP hit-path (high level)

A short unique prompt will **not** do it.

You need:

1. Hybrid Mamba/GDN model, `mamba-cache-mode=align`, prefix cache on.
2. MTP or any `use_eagle=true` spec decode.
3. A **long shared prefix** (system + tools + soul, several k tokens).
4. Several successful turns that write cache (including LMCache if enabled).
5. A new request that **hits** that prefix and diverges only at the tail
   (new user turn, or a different agent with the same system prefix).

Failure modes to log:

- `spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`
  for that window (0% **or** ~100% are both bad if the text is junk).
- Whether `finish_reason=length` with empty content / digit loops.
- Prefix-cache hit counters vs a unique-prefix control.

Grade against a control engine with prefix cache off, or MTP off,
on the same weights.

---

## 11. Disclaimer

Home lab, consumer GPUs, one model family. Other backends
(FlashAttention vs FlashInfer), other block sizes, and other
connectors will shift which row you land on. The scheduler
invariant and the LMCache 0.5.3 pointer bug are not specific to
this box.

If a row above disagrees with a later upstream release, trust the
release and send a PR to this file.
