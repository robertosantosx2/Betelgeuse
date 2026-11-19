# Qwen3.5 trunk-only runtime audit

Date: 2026-11-19

## Scope

This audit covers the runtime boundary for the Qwen3.5-35B-A3B trunk-only bundle. It is deliberately Linux-only. Windows jobs and Windows-specific behavior are outside this validation target.

The real-model packer and pager data path were validated separately on the MSI on 2026-11-18. This document does not claim end-to-end inference.

## Runtime contract

The bundle contains 40 paged decoder-trunk layers (blocks 0-39), 256 experts per layer, and 8 routed experts per token. Block 40 is the MTP/NextN block and remains resident in the dense GGUF. The runtime must:

1. recognize `betelgeuse.stripped_experts=true` and `betelgeuse.qwen35_trunk_only=true`;
2. validate `betelgeuse.trunk_layers=40` and `betelgeuse.mtp_layer=40`;
3. verify all 120 trunk expert tensors are external to the dense file and all three block-40 expert tensors remain resident;
4. require `--moe-direct-io`, `--moe-gguf`, and `--moe-pack-dir` before loading a stripped bundle;
5. initialize the pager with exactly 40 layers and 256 experts;
6. activate `experts.bin` + `native_experts.txt` pack mode before allocating the paged MoE execution pool;
7. fail closed if pager initialization or pack activation fails;
8. keep ordinary, non-stripped model loading and non-pack pager behavior unchanged.

## Findings and changes

### 1. MTP expert tensors were covered by the broad expert-skip predicate

The loader previously treated every tensor matching `_exps` as externally paged when `moe_direct_io` was enabled. That included block 40, even though its payload is present in the dense GGUF and is not in the 40-layer pack.

The loader now uses the GGUF tensor's `external` classification for stripped bundles. Only external trunk expert payloads are skipped during mmap prefetch and tensor loading. The payload skip happens before any mapped pointer is derived and applies to every load mode, including non-mmap/direct reads, so an out-of-file trunk offset is never dereferenced. External trunk tensors are placed in zero-sized placeholder contexts rather than allocating their absent payloads; the same classification is used in the memory-fit and real-load passes. Resident block-40 MTP experts continue through the normal loading path. The constructor also verifies the expected payload placement for all three expert roles in blocks 0-40.

### 2. Stripped bundles could proceed without a usable pack

The runtime previously logged a warning when pack activation failed and continued building the MoE pool. That is unsafe for a dense GGUF whose trunk expert payloads are intentionally absent.

The model loader now rejects stripped Qwen3.5 bundles unless direct I/O, a GGUF path, and a pack directory are configured. It validates the architecture, 40+1 layer geometry, 256 experts, top-8 routing, and the bundle metadata. Context initialization now throws on pager initialization or pack activation failure and verifies pack mode before creating the execution pool.

### 3. Non-pack compatibility

The special handling is gated by the explicit `betelgeuse.stripped_experts` metadata. For ordinary GGUFs, the existing expert-skip behavior under `moe_direct_io` is unchanged. No Windows implementation or platform-specific code was added.

## Validation plan

The following gates are required before attempting any model forward pass:

- [ ] Linux CPU build and tests pass.
- [ ] Linux CUDA build passes with `GGML_CUDA=ON` and `CMAKE_CUDA_ARCHITECTURES=86`.
- [ ] Loader tests/CI confirm the new C++ paths compile.
- [ ] Static inspection confirms MTP block 40 is loaded from the dense GGUF and blocks 0-39 are skipped only as external expert payloads.
- [ ] Runtime startup with a deliberately missing/invalid pack fails before pool allocation.
- [ ] Runtime startup with the valid pack logs pager initialization and pack mode before pool allocation.
- [ ] Existing non-stripped MoE path remains unchanged.

No inference, token generation, logits comparison, throughput measurement, or context-length claim is authorized by this audit. After CI and code review pass, the next step is to prepare a minimal integration test; actual execution on the MSI requires a separate explicit go-ahead.
