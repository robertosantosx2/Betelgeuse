# Qwen3.5-35B-A3B Betelgeuse: MSI validation record

Date: 2026-11-18

## Scope

This record documents validation of the trunk-only expert bundle and the native CUDA pager data path on the target MSI workstation. It does **not** claim end-to-end Qwen3.5 inference.

## Target workstation

- MSI Thin 15 B12UC
- Intel Core i7-12650H
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM (sm_86)
- 14 GiB system RAM; 32 GiB swap
- Ubuntu
- NVIDIA driver 580.178.04
- CUDA Toolkit 13.4 (nvcc 13.4.92)

## Source model

- File: `Qwen_Qwen3.5-35B-A3B-IQ2_XXS.gguf`
- Size: 10,676,139,840 bytes
- SHA-256: `479cc80b2500f7ba5f6131986da188ecb406cf7855094c5d8ae0ec332a715105`
- GGUF v3; architecture `qwen35moe`
- 41 expert-bearing blocks: 40 trunk layers plus MTP/NextN block 40
- 256 experts per block; 8 routed experts per token
- No PLE companion is present or required

## Bundle generation and integrity

The real-model dry run reported:

- 40 trunk layers
- 256 experts per layer
- 10,240 trunk experts
- 8,380,219,392 expert payload bytes
- MTP layer 40 preserved in the dense GGUF
- PLE not required

The real pack completed and `--verify-all` verified all 10,240 trunk experts.

Generated bundle facts:

- Dense GGUF: 2,295,920,640 bytes
- Expert pack: 8,380,219,392 bytes
- Expert-pack SHA-256: `a0d90eac671ae1a6c49116eb4200e8015fd10ad09e3e42b7b3f82b6f17280408`

## CUDA build

CMake configured with:

```bash
cmake -S . -B build-qwen35 \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release
```

The CUDA pager test target built successfully after installing the host build dependencies `librsvg2-bin` and `liburing-dev`. The CUDA runtime library path was supplied from the installed CUDA 13.4 toolkit at configure time:

```bash
cmake -S . -B build-qwen35 \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda-13.4/targets/x86_64-linux/lib"
cmake --build build-qwen35 --target test_moe_pager -j2
```

Build result: `[100%] Built target test_moe_pager`.

## Native pager result

Command shape:

```bash
build-qwen35/bin/test_moe_pager \
  /path/to/Qwen3.5-35B-A3B-Dense.gguf \
  40 256 8 \
  /path/to/experts.bin \
  /path/to/native_experts.txt \
  /path/to/Qwen_Qwen3.5-35B-A3B-IQ2_XXS.gguf
```

Observed result:

```text
fetch(L 0,E  0): EXACT MATCH
fetch(L 0,E  1): EXACT MATCH
fetch(L 7,E123): EXACT MATCH
fetch(L23,E255): EXACT MATCH
fetch(L39,E255): EXACT MATCH
async prefetch+commit: 4/4 EXACT MATCH
capacity-rejection: PASS
size-rejection: PASS
write-allocate: EXACT MATCH
reuse-stability: PASS
prefetch-batch: PASS
batch(L 0,N36,pass0): EXACT MATCH
batch(L 0,N36,pass1): EXACT MATCH
batch(L20,N36,pass0): EXACT MATCH
batch(L20,N36,pass1): EXACT MATCH
batch(L39,N36,pass0): EXACT MATCH
batch(L39,N36,pass1): EXACT MATCH

result: PASS   io_ops=229 io_bytes=190603264 pf_hit=8 disk_wait_us=45835 dma_ms=22.549
```

The process exited and the log ended with `result: PASS`. The test was launched detached with a persistent log, so the terminal remained available.

## What this establishes

- The real-model trunk expert pack was generated and all 10,240 packed experts passed byte verification.
- The native CUDA pager can read selected packed experts and match the original GGUF slices byte-for-byte.
- The synchronous, asynchronous prefetch/commit, write-allocate, cache-reuse, prefetch-batch and batch-read paths exercised by `test_moe_pager` passed on this MSI.
- The tested pager data path ran with the RTX 3050 CUDA build.

## What this does not establish

- No Qwen3.5 forward pass or token generation was run.
- No model graph integration, scheduler integration, or end-to-end runtime startup was validated.
- No throughput, latency, context-length, or sustained-memory benchmark for inference is claimed.
- The test is a pager/data-path validation, not a model-inference test.

## Next engineering gate

Before attempting inference, inspect and document the actual runtime launch path and prove that it:

1. recognizes the Qwen3.5 trunk-only metadata;
2. initializes the pager for exactly 40 trunk layers;
3. activates pack mode before any read of stripped trunk expert payloads;
4. leaves MTP/NextN block 40 resident and available to the model loader;
5. keeps the existing non-Qwen3.5 and non-pack paths unchanged.

Only after these conditions are met should a minimal one-token or logits-equivalence integration test be designed.