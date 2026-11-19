# Qwen3.5-35B-A3B trunk-only bundle

This path packages the single-file Qwen3.5-35B-A3B GGUF for Betelgeuse's existing native expert pager. It deliberately does **not** use the Qwen3.8 PLE bundle format.

## Model layout

The target GGUF has 41 expert-bearing blocks:

- `blk.0` through `blk.39`: 40 decoder-trunk layers, 256 routed experts each. These 120 expert tensors are moved byte-for-byte to `experts.bin`.
- `blk.40`: the Qwen3.5 MTP/NextN block. Its gate/up/down expert tensors and the other NextN tensors remain in the dense GGUF.
- There is no `per_layer_token_embd.weight`; no PLE file is created or required.

The packer refuses models that do not expose exactly 41 contiguous expert layers and 256 experts per layer. It does not quantize, reorder within an expert, or rewrite expert payload bytes.

## Build the bundle

Run from the Betelgeuse repository root with Python, NumPy and the repository's `gguf-py` package available:

```bash
python tools/qwen35_trunk_pack.py \
  --model /path/to/Qwen_Qwen3.5-35B-A3B-IQ2_XXS.gguf \
  --output /path/to/Qwen3.5-35B-A3B-Betelgeuse
```

The output directory must be new/empty. The resulting files are:

```text
Qwen3.5-35B-A3B-Betelgeuse/
├── Qwen3.5-35B-A3B-Dense.gguf
├── experts.bin
├── expert-pack.json
└── native_experts.txt
```

Inspect without writing:

```bash
python tools/qwen35_trunk_pack.py --model MODEL.gguf --output OUT --dry-run
```

Sample-check the packed experts, or compare all 10,240 trunk experts byte-for-byte:

```bash
python tools/qwen35_trunk_pack.py --model MODEL.gguf --output OUT --verify
python tools/qwen35_trunk_pack.py --model MODEL.gguf --output OUT --verify-all
```

## Runtime contract

The existing Qwen3.5 model loader distinguishes trunk and MTP flags and exposes the 40-layer trunk separately from `n_layer_all`. Start the pager with the **trunk layer count (40)**, not the total GGUF block count (41), and pass the pack plus its index to `moe_pager::use_pack()`.

The dense GGUF retains original tensor descriptors for all 40 trunk expert tensors, but their payload offsets intentionally point beyond EOF. The pager must be in pack mode before any expert read. The `blk.40` MTP expert payloads remain physically present. The metadata keys `betelgeuse.qwen35_trunk_only`, `betelgeuse.trunk_layers`, and `betelgeuse.mtp_layer` identify this packaging variant. No `betelgeuse.ple_file` key is written.

Do not load this stripped dense GGUF with upstream llama.cpp or a Betelgeuse build without the native pager.

## Native pager byte-check

The CUDA build also produces `build/bin/test_moe_pager`. The test accepts an optional pack/index pair and an unstripped source GGUF as an independent byte reference. After building the bundle, run:

```bash
build/bin/test_moe_pager \
  /path/to/Qwen3.5-35B-A3B-Betelgeuse/Qwen3.5-35B-A3B-Dense.gguf \
  40 256 8 \
  /path/to/Qwen3.5-35B-A3B-Betelgeuse/experts.bin \
  /path/to/Qwen3.5-35B-A3B-Betelgeuse/native_experts.txt \
  /path/to/original/Qwen_Qwen3.5-35B-A3B-IQ2_XXS.gguf
```

This initializes the pager against the stripped dense GGUF with 40 trunk layers, switches it to pack mode, and compares its synchronous, prefetch, and batch reads against the corresponding slices in the original GGUF. It exercises the pager data path; it does not run model inference or validate graph scheduling.

## Validation status

The synthetic regression suite checks:

- the pack contains only layers 0–39;
- all 10,240 packed trunk experts round-trip byte-for-byte;
- MTP layer 40 expert payloads remain in the dense GGUF;
- trunk descriptors remain available while trunk payloads are absent from the dense file;
- the Qwen3.8 PLE metadata/file is not emitted;
- invalid layer counts are rejected.

The real Qwen3.5-35B-A3B GGUF has also been processed on the target MSI:

- dry-run, real pack creation and `--verify-all` completed; all 10,240 trunk experts verified;
- CUDA build of `test_moe_pager` completed for `sm_86`;
- native pager test against the original GGUF completed with `result: PASS`;
- synchronous fetch, async prefetch/commit, write-allocate, cache reuse and batch reads were byte-exact in the exercised cases.

The detailed hardware, commands, pack hashes and test output are recorded in [qwen35-msi-validation-2026-11-18.md](qwen35-msi-validation-2026-11-18.md).

These results validate the packer and pager data path on the MSI. They do **not** establish end-to-end model inference, graph scheduling, runtime launch integration, throughput, or context-length capability. Do not infer Qwen3.5 inference support solely from this packer or pager test.

## Runtime audit update (2026-11-19)

The runtime path has now been inspected and hardened in the PR branch:

- the loader identifies stripped bundles from explicit GGUF metadata;
- it verifies all three routed expert tensors in blocks 0-39 have external payloads and all three block-40 MTP expert tensors remain resident;
- only external trunk expert payloads are skipped during normal tensor loading, so MTP/NextN experts are still loaded from the dense GGUF;
- stripped bundles are rejected unless direct I/O, the GGUF path, and the pack directory are configured;
- pager initialization and pack activation are mandatory for stripped bundles; failure aborts startup before the MoE execution pool is allocated;
- the special behavior is metadata-gated, preserving existing non-stripped loading paths.

Audit details and the pending acceptance checklist are in [qwen35-runtime-audit-2026-11-19.md](qwen35-runtime-audit-2026-11-19.md).

## Next validation gate

The Linux CI build and tests must now prove that the runtime changes compile and pass. After that, verify startup failure for an absent/invalid pack and successful pack activation with the valid bundle. Do not run inference yet.

The runtime launch path must continue to satisfy:

1. recognizes the Qwen3.5 trunk-only metadata;
2. initializes the pager for exactly 40 trunk layers;
3. activates pack mode before any read of stripped trunk expert payloads;
4. leaves MTP/NextN block 40 resident and available to the model loader;
5. keeps existing non-Qwen3.5 and non-pack paths unchanged.

Only after these conditions are met should a minimal logits-equivalence or one-token integration test be designed.