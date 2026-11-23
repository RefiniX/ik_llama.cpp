# CPU performance patches for ik_llama.cpp

CPU-side patches on top of ik_llama.cpp `06a141f3` (2026-11-20), measured on a dual-socket Cascade Lake box. My own patches are free to take, rewrite or submit under your own name, no attribution needed. Two are other people's work under MIT; please credit them: NUMA weight/KV mirroring (mikechambers84, rebased by sultanqasim) and the qwen35 pre-tokenizer split (Kabir Potdar, ggml-org/llama.cpp#22110). I will not maintain these branches or turn them into PRs.

**AI disclosure:** the investigation, patches and benchmarks were done by an AI agent (Claude Code) on my machine, under my direction. I run these builds daily but have not re-derived the analysis myself. Test on your own hardware.

## Branches

- `cpu-perf-06a141f3`: 22 patches, none NUMA-specific; each is useful on its own.
- `numa-mirror-06a141f3`: `cpu-perf-06a141f3` + NUMA mirroring, expert row placement and token-generation patches that rely on the mirror's thread pinning (10 commits). Only matters on multi-socket machines.
- Older snapshots (`*-8d50c6ff`, `*-41d123e9`) are left unchanged. Snapshots are never rebased.

Marks: ~~struck~~ = ik or mainline already has it (or a better form); **rewrite** = my own review found a simpler form I have not done; **fix** = a small known bug. One-line notes here; the full text for patches 1-14 and the NUMA ones is in [PATCHES.md on numa-mirror-8d50c6ff](https://github.com/RefiniX/ik_llama.cpp/blob/numa-mirror-8d50c6ff/PATCHES.md), for the newer ones in the commit messages.

Output check: the source of `numa-mirror-06a141f3` is identical to my daily build minus the three PRs listed at the end (and my fix on top of one of them). `cpu-perf-06a141f3` is a subset of it and was not run separately.

Numbers are per patch (A/B or isolated benchmark), 2x Xeon Gold 5220 (18 cores each, AVX-512 VNNI, 6x DDR4-2666 per socket), `-t 32`, most with the NUMA mirror. Slow cores and two sockets make some patches look bigger than on a desktop; re-measure.

## cpu-perf-06a141f3

| # | commit | what | measured here | notes |
|---|---|---|---|---|
| 1 | `llama : bound per-sequence KV cell scans to the used range` | `seq_rm` / `seq_pos_max` stop at a high-water mark instead of scanning all `-c` cells | Qwen3.6 MTP, `-c 262144`: 30.3 -> 35.4 t/s | **fix:** save the mark with checkpoints |
| 2 | ~~`llama : multi-threaded per-step recurrent state restore on CPU`~~ | SSM state restore after a rejected draft on several threads | restore 18.3 -> 2.1 ms | mainline avoids the copy (#22673; #31669 open) |
| 3 | `iqk : split KV across threads for small-batch CPU flash attention` | 2-16 query rows no longer read the whole KV per head | attention at 32k B=5 x0.45 time | not bit-identical for 2-16 rows; **fix:** exclude MLA (`nek2 > 1`) |
| 4 | `sampling : apply repetition penalties only to penalized tokens` | one lookup per window token instead of per vocab entry | presence 1.5: 38.8 -> 46.7 t/s | **fix:** skip negative ids |
| 5 | `ggml : flush denormals to zero in CPU compute threads on x86` | FTZ/DAZ in compute threads | image encoding x1.8-3.5, prompt +8% | **rewrite:** merge with 10 into one save/set/restore |
| 6 | ~~`sampling : skip the full vocab sort when nothing truncates the candidates`~~ | no full sort when `top_k <= 0` | gpt-oss top_k 0: 14.5 -> 17.0 t/s | mainline's `do_sort` flag is simpler |
| 7 | `ggml : quantize few src1 rows on one thread` | fewer rows than threads: one thread quantizes, no cache-line ping-pong | Qwen3.6 x1.04 (MTP x1.08) | **rewrite:** one-line split change |
| 8 | `unicode : port the Qwen3.5 pre-tokenizer split from mainline` | no `std::regex` compile per tokenization | sampler init per MTP round 0.93 ms -> 16 us | port of mainline #22110 |
| 9 | `server : create interval context checkpoints during speculative decoding` | checkpoints during MTP generation too | divergence near the end reprocesses ~50 instead of ~1590 tokens | **fix:** move the alignment check after the early return |
| 10 | `ggml : restore the caller's MXCSR after a graph is computed` | fix for 5 | - | **rewrite:** see 5 |
| 11 | `iqk : run the indexer top-k of several query rows on separate threads` | Qwen3.8 QSA top-k per row on its own thread | with 12, MTP: 64k 13.5 -> 16.8 t/s | |
| 12 | `iqk : spread the per-row calls of the indexed flash attention over the threads` | indexed sparse FA split over all threads | see 11 | **fix:** chunk only for 8/12 heads |
| 13 | `llama : copy large state tensors on a few threads when saving or restoring into memory` | checkpoint / prompt-cache copies on up to 8 threads | Qwen3.8 checkpoint save 32 -> 6 ms | **fix:** serial fallback if thread creation throws |
| 14 | `server : reuse an evicted checkpoint's buffer and do not zero new ones` | no zero-fill before overwrite | with 13: creation 97 -> 20 ms | **rewrite:** use ik's `no_init<T>` |
| 15 | `ggml : keep the barrier counters on their own cache lines` | as mainline #9598 | within noise here | |
| 16 | `server : share checkpoint data between a slot and its prompt-cache copies` | prompt cache shares checkpoint data instead of copying (mainline #30137's approach) | Qwen3.8 cache save 96-150 -> 50 ms (worst 3047 -> 671) | |
| 17 | `iqk : AVX-512 IQ4_XS_R8 x Q8_K kernel that keeps the AVX2 results` | single-token kernels went to AVX2 (tuned on Zen) | with 18: Qwen3.8 x1.07, Qwen3.6 x1.04, Gemma 4 x1.05 | bit-identical |
| 18 | `iqk : AVX-512 single-row kernels for Q4_K/Q5_K/Q6_K R4, Q4_0/MXFP4/Q8_0 R8 that keep the AVX2 results` | same, other repacked types | see 17 (gpt-oss x1.01) | bit-identical |
| 19 | `iqk : AVX-512 Q2_K_R4 x Q8_K kernel that keeps the AVX2 results` | same, Q2_K | kernel x1.2-1.4, not visible in the server | bit-identical |
| 20 | `iqk : AVX-512 kernel for Q6_K that is not repacked x Q8_2_X4, same results as AVX2` | same, plain Q6_K | as 19 | bit-identical |
| 21 | `common : take top_k straight from the logits when nothing before it needs the full vocabulary` | no copy of the whole vocab into a candidate array per token | Qwen3.6 / Gemma 4 x1.04, Qwen3.8 x1.02 | falls back on ties, grammar, penalties, DRY; bit-identical |
| 22 | `iqk : AVX-512 activation quantization (Q8_K, Q8_K32, Q8_2_X4, Q8_1_X4) with the AVX2 results` | activation quantization before matmuls in AVX-512 | Qwen3.8 x1.014 | bit-identical; Q8_2_X4 only x1.1 (scalar per-32 scales) |

## numa-mirror-06a141f3 (adds)

| # | commit | what | measured here | notes |
|---|---|---|---|---|
| 23 | `numa : NUMA weight/KV mirroring (Mike Chambers' implementation)` | one weight copy per node, threads pinned per node | Qwen3.6-class: +30-60% vs interleave | use `--numa mirror --numa-mirror weights`; **rewrite:** drop KV mirroring, ~250-line weights-only version |
| 24 | `numa : add --numa-mirror dense for models that do not fit twice` | mirror only tensors every token reads in full | Qwen3.8 x1.17, gpt-oss x1.23 | **rewrite:** select by graph use, not name |
| 25 | `numa : mirror the output projection of tied-embedding models with --numa-mirror dense` | include Gemma's tied output | Gemma 4 dense x1.04 | unneeded after 24's rewrite |
| 26 | `numa : place routed expert rows on the node whose threads compute them (experimental)` | one expert copy, rows placed where they are computed | with dense: Qwen3.8 x1.32, gpt-oss x1.46 | `MPOL_BIND`, assumes `-t == -tb`, 2 nodes only |
| 27 | `numa : select expert row placement with --numa-mirror rows / shard instead of an env var` | `rows`, `shard` = `dense,rows` | gpt-oss dense -> shard x1.27 | should become its own flag |
| 28 | `ggml : dissemination barrier for the NUMA-pinned token generation path` | log2(nth)-round barrier, 2.8 -> 1.1 us | Qwen3.8 x1.07 | **does not build on Windows**; **rewrite:** general barrier |
| 29 | `ggml : fuse small token-generation nodes (...)` | idle threads compute small matrices; two element-wise fusions | x1.03 together | MULTI_ADD+SCALE partly ~~superseded~~ by ik #2957 (after this base); **rewrite:** drop element-wise fusions |
| 30 | `ggml : quantize a large few-row src1 on the first 4 threads` | big few-row inputs quantized by 4 threads | B=6 -4.5% time | **rewrite:** fold into 7 |
| 31 | `iqk : split the fused MoE up/gate evenly, rotate it per expert, masked SiLU tail` | even split, no idle threads | B=1 -3.9% time | uses mirror functions |
| 32 | `ggml : split single-row element-wise ops across threads, results unchanged` | ADD, MULTI_ADD, FMA, SOFTCAP etc. no longer on thread 0 only | Gemma 4, Qwen3.8 (no MTP) x1.015 | needs 29 (on this branch only); MULTI_ADD for Qwen3.8's hc mix overlaps ik #2957 |

28-31 together in the server: Qwen3.8 x1.14 (MTP x1.16), Qwen3.6 x1.08, Gemma 4 x1.13, gpt-oss x1.03.

## Not included: other people's open PRs in my daily build

- ikawrakow/ik_llama.cpp#2873: avoid the integer division in `DataInfo::src1_row`
- ikawrakow/ik_llama.cpp#2654: real bilinear upscale on CPU (Qwen-VL position embedding)
- ikawrakow/ik_llama.cpp#2655: causal mask for M-RoPE images by (t, y, x). With #2924's fast KQ-mask fill (in this base) images become non-causal again in prompt batches; my build adds a fix that compares (y, x) in the fast fill.

Build: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_IQK_FA_ALL_QUANTS=ON -DGGML_EXPERT_CHUNKING=OFF -DGGML_USE_I32GATHER=ON` (the last two were faster on this Intel box).
