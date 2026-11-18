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

The regression suite synthesizes a 41-layer / 256-expert GGUF and checks:

- the pack contains only layers 0–39;
- all 10,240 packed trunk experts round-trip byte-for-byte;
- MTP layer 40 expert payloads remain in the dense GGUF;
- trunk descriptors remain available while trunk payloads are absent from the dense file;
- the Qwen3.8 PLE metadata/file is not emitted;
- invalid layer counts are rejected.

Passing these tests validates the packer format, not end-to-end model inference. The real model dry-run, full pack/verify, CUDA build, runtime startup, and MSI benchmark remain separate validation gates. In particular, do not infer Qwen3.5 inference support solely from the existence of this packer.
