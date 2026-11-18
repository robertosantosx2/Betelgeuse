# Qwen3.5 native-pack runtime guard

This note documents the runtime-side contract for the Qwen3.5-35B-A3B trunk-only bundle introduced by PR #1 and hardened by PR #2.

## Bundle boundary

- Decoder trunk: `blk.0` through `blk.39`, 40 layers × 256 experts.
- Native `experts.bin`: gate, up and down payloads for those 10,240 trunk experts.
- MTP/NextN: `blk.40` remains in the dense GGUF, including its routed experts. It is not indexed by the trunk pager.
- No PLE companion is required.

The dense GGUF retains metadata descriptors for the external trunk expert tensors. The loader marks those weights as external, prevents them from entering lazy-read ranges, and skips their absent payload bytes. A stripped model must activate the native pack; it must never fall back to reading the missing trunk payloads from the dense GGUF.

## Native-experts-v3 validation

Before enabling pack mode, `moe_pager::use_pack()` validates:

1. Every non-comment row has exactly the eight v3 fields.
2. Every trunk layer occurs exactly once; duplicates and missing layers are rejected.
3. Gate and up types match the GGUF gate/up tensors, and down type matches the GGUF down tensor.
4. `blob_bytes` equals the sum of the GGUF gate, up and down expert byte sizes.
5. Layer and expert offsets are checked without overflowing; every aligned direct-I/O read fits inside a regular `experts.bin` file and the pager's 32-bit slice limits.
6. The index is built in temporary vectors. The GGUF index remains active and a newly opened pack descriptor is closed if any validation fails.

Only after every row passes are the index vectors, staging size and pack descriptor committed to the pager.

The asynchronous submit path drains all queued SQEs and treats hard submission/completion errors as fatal rather than returning a request that could wait indefinitely. Interrupted completion waits are retried. The pager owns and releases the allocated `io_uring` object as well as its queue.

## MSI regression command

After building the CUDA test target, run the pack-mode test with the stripped dense GGUF, native pack/index and original GGUF reference:

```bash
cd "$HOME/betelgeuse-qwen35-validation"

DENSE="$HOME/benchmarks/qwen3.5-35b-a3b-betelgeuse/Qwen3.5-35B-A3B-Dense.gguf"
PACK="$HOME/benchmarks/qwen3.5-35b-a3b-betelgeuse"
ORIGINAL="$HOME/models/tandem-qwen35/Qwen_Qwen3.5-35B-A3B-IQ2_XXS.gguf"

cmake --build build-qwen35 --target test_moe_pager -j2

build-qwen35/bin/test_moe_pager \
  "$DENSE" 40 256 8 \
  "$PACK/experts.bin" \
  "$PACK/native_experts.txt" \
  "$ORIGINAL"
```

When the pack arguments are supplied, the test first checks rejection of duplicate and missing layer rows, a mismatched tensor type, an out-of-range layer offset and a truncated pack. Each rejection must leave pack mode disabled and must not retain a pack descriptor. It then activates the valid pack and runs the existing byte-exact synchronous, asynchronous and batch pager checks against the original GGUF.

The malformed-index fixtures are created as temporary files. The truncated-pack fixture is a sparse temporary file beside `experts.bin`; the existing bundle files are not modified.

## Validation boundary

The previously recorded MSI PASS covers pack generation/byte verification and the existing pager data path. The new malformed-index cases and the PR #2 loader/runtime changes still require a fresh MSI build and execution. A pager PASS is not an end-to-end model inference result.
