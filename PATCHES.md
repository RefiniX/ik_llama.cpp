# CPU performance patches for ik_llama.cpp

A set of CPU-side patches on top of ik_llama.cpp `41d123e9`, written and benchmarked on a dual-socket Cascade Lake box. They are offered as-is: anyone may take, rewrite or submit any of them, under their own name, with no attribution needed. I will not be maintaining this branch or turning these into PRs myself.

**AI disclosure:** the investigation, the patches and the benchmarks were done by an AI agent (Claude Code, Claude Opus) on my machine, under my direction. I have been running these builds in daily use; I have not independently re-derived the analysis. Please test on your own hardware before relying on any of it.

## Branches

- `cpu-perf-41d123e9`: eight independent patches (one commit each). Each one is useful without the others.
- `numa-mirror-41d123e9`: `cpu-perf-41d123e9` plus NUMA weight mirroring (two commits). This is invasive and only matters on multi-socket machines; see #2396 for the upstream discussion.

These branches are a snapshot and will not be rebased. A newer base, if ever published, gets new branch names.

Numbers are per patch (A/B with and without it, or an isolated benchmark of the changed function), measured between 2026-11-07 and 11-13 on slightly different bases, most with the NUMA mirror patch applied. Re-measure on your own hardware.

Test machine: 2x Xeon Gold 5220 (Cascade Lake, AVX-512 VNNI, 18 cores per socket used, no SMT), 128 GB, CPU only, `-t 32`. Not tested on AMD, ARM, Windows, macOS or GPU builds.

## cpu-perf-41d123e9

| commit | what | output | measured here | known issues |
|---|---|---|---|---|
| `llama : bound per-sequence KV cell scans to the used range` | `seq_rm`, `seq_pos_max`, ... stop at a high-water mark instead of scanning all `-c` cells (the server calls them every token with MTP) | bit-identical | Qwen3.6-35B-A3B fine-tune, NUMA mirror, MTP n_max=4, story prompt: `-c 262144` 30.3 -> 35.1-35.6 t/s (`-c 8192`: 35.3-37.7) | every future writer of `cells[i].pos` must raise the mark. If a whole-cache state restore throws midway (truncated data), the cells read so far stay invisible to the scans; the `return false` paths are cleared by the caller |
| `llama : multi-threaded per-step recurrent state restore on CPU` | restoring SSM state after a rejected MTP draft used one thread (CPU has no async copy) | bit-identical | qwen35moe 35B-A3B fine-tune, IQ4_XS, KV q8_0: restore 18.3 -> 2.1 ms (~60 MiB of state) | gain comes from low single-core bandwidth; without OpenMP threads are created per call; Metal buffers are `is_host` (untested). Mainline avoids the copy entirely: since ggml-org/llama.cpp#22673 a rollback only selects the snapshot to read next (`rs_idx`) |
| `iqk : split KV across threads for small-batch CPU flash attention` | 2-16 query rows (MTP verification, `-np 2`) fell back to a path where every q head reads the whole KV | not bit-identical for 2-16 rows (chunked softmax merge; depends on `-t`); 1 row unchanged | same model (16 q / 2 KV heads, head size 256), KV q8_0, attention time: depth 32k B=5 0.45x, depth 64k B=2 0.33x | **MLA models (1 KV head, large rk2) are not excluded: work buffer up to ~2 GB at B=16.** Thresholds tuned on this machine only |
| `sampling : apply repetition penalties only to penalized tokens` | one hash lookup per vocab entry per token -> only the `penalty_last_n` entries | bit-identical | same model, no MTP, top_k 20, presence 1.5: 38.8 -> 46.7 t/s (= presence 0) | only differs if a caller passes duplicate token ids |
| `ggml : flush denormals to zero in CPU compute threads on x86` | sets FTZ/DAZ in MXCSR; denormals cost a microcode assist on Intel | values < 1.2e-38 become 0 | Qwen3.6-35B-A3B fine-tune + BF16 mmproj (qwen3vl): image encoding 1.8-3.5x faster, text prompt processing +8%, generation unchanged | **does not restore MXCSR: thread 0 is the caller's thread, so an embedding application inherits FTZ/DAZ; with OpenMP the pooled worker threads keep it too, affecting other OpenMP code in the process.** Not active with MSVC (`__SSE__` undefined). Should be save/restore + an opt-out before upstreaming |
| `sampling : skip the full vocab sort when nothing truncates the candidates` | `top_k <= 0` was "keep all and sort"; DIST sorted unsorted candidates although the draw does not need order (as mainline did in ggml-org/llama.cpp#13173 and ggml-org/llama.cpp#15704) | identical when candidates reach DIST sorted (e.g. `top_k > 0`); otherwise same distribution, different draw for the same seed | gpt-oss-120b MXFP4, top_k 0, top_p 1.0: 14.5 -> 17.0 t/s | C API callers that relied on sorting: `llama_sample_dist` no longer leaves a fully sorted array, and `llama_sample_top_k` with `k <= 0` no longer sorts |
| `ggml : quantize few src1 rows on one thread` | with fewer src1 rows than threads (expert-down inputs, MTP verification) each row was quantized by a different thread; the following matmul then reads rows from other cores' modified cache lines | bit-identical | Qwen3.6-35B-A3B fine-tune, NUMA mirror: generation x1.04, MTP n_max=8 x1.08; Gemma 4 26B-A4B x1.02 | measured on 2 sockets only, single-socket gain unknown. Splitting one row over all threads (the mainline way) was much slower here |
| `unicode : port the Qwen3.5 pre-tokenizer split from mainline` | the qwen35 pre-tokenizer regex had no custom split, so every tokenization compiled a `std::regex` (~0.46 ms) and could overflow the stack on long inputs; MTP tokenizes the generation prompt every round | token ids identical (mainline's qwen35 vocab tests, Chinese / code / English corpora) | sampler init per MTP round 0.93 ms -> 16 us; with the row above, Qwen3.6 fine-tune with MTP x1.11 | port of ggml-org/llama.cpp#22110 (Kabir08), unchanged |

## numa-mirror-41d123e9

| commit | what | measured here | known issues |
|---|---|---|---|
| `numa : NUMA weight/KV mirroring (Mike Chambers' implementation)` | one copy of the weights per node, threads pinned per node, two-level barrier for small batches. Original code by mikechambers84, rebased by sultanqasim; this port adds resyncs for KV writes that bypass the graph | Qwen3.6-35B-A3B Q6_K-class fine-tunes: generation +30-60% vs `numactl --interleave` | use `--numa mirror --numa-mirror weights`: the default `all` (also mirrors KV) is 4-21% slower. `ggml_tensor` grows by 16 bytes. Silent if `mbind` is blocked. `-t` and `-tb` should be equal |
| `numa : add --numa-mirror dense for models that do not fit twice` | mirror only the tensors every token reads in full (attention/SSM, router, shared experts, output) | +2.7-2.9 GiB; Qwen3.8-Flash-Next IQ4_XS 1.18x per token, gpt-oss-120b 18.6 -> 22.9 t/s | selects tensors by name; does not force `--no-mmap`; one allocation per tensor |

Build used: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_IQK_FA_ALL_QUANTS=ON -DGGML_EXPERT_CHUNKING=OFF -DGGML_USE_I32GATHER=ON` (the last two were faster on this Intel box; retest elsewhere).
