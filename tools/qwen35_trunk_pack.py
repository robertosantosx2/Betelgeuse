#!/usr/bin/env python3
"""Build a trunk-only Betelgeuse bundle for single-file Qwen3.5-35B-A3B GGUFs.

The 40 decoder-trunk expert layers are moved byte-for-byte to experts.bin.
The blk.40 MTP/NextN block remains in the dense GGUF, including its experts.
No PLE companion is expected or synthesized.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT / "tools"))
import moe_pack
from gguf import GGUFEndian, GGUFReader, GGUFWriter

TRUNK_LAYERS = 40
MTP_LAYER = 40
TOTAL_LAYERS = 41
EXPERTS = 256
DENSE_NAME = "Qwen3.5-35B-A3B-Dense.gguf"
PACK_NAMES = ("experts.bin", "expert-pack.json", "native_experts.txt")


def qwen35_layout(model):
    reader = GGUFReader(model)
    architecture = reader.fields.get("general.architecture")
    block_count = reader.fields.get("qwen35moe.block_count")
    if architecture is None or architecture.contents() != "qwen35moe":
        raise ValueError("expected a GGUF with general.architecture=qwen35moe")
    if block_count is None or block_count.contents() != TOTAL_LAYERS:
        found = None if block_count is None else block_count.contents()
        raise ValueError(f"Qwen3.5-35B-A3B requires qwen35moe.block_count={TOTAL_LAYERS}; found {found}")
    names = {tensor.name for tensor in reader.tensors}
    layout, experts, total = moe_pack.expert_layout(model)
    if len(layout) != TOTAL_LAYERS:
        raise ValueError(f"Qwen3.5-35B-A3B requires {TOTAL_LAYERS} expert layers (40 trunk + blk.40 MTP); found {len(layout)}")
    if experts != EXPERTS:
        raise ValueError(f"Qwen3.5-35B-A3B requires {EXPERTS} experts per layer; found {experts}")
    for layer in range(TOTAL_LAYERS):
        for role in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            if name not in names:
                raise ValueError(f"missing required tensor {name}")
    return layout[:TRUNK_LAYERS], experts, sum(row["blob"] * experts for row in layout[:TRUNK_LAYERS])


def write_pack(model, output, layout, experts):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in PACK_NAMES):
        raise FileExistsError("output pack already exists; refusing to overwrite")
    total = sum(row["blob"] * experts for row in layout)
    padded = (total + 4095) // 4096 * 4096
    if shutil.disk_usage(output).free < padded + 1024 * 1024:
        raise OSError(f"not enough free space: need at least {padded / 2**30:.2f} GiB")
    with tempfile.TemporaryDirectory(prefix=".qwen35-pack-", dir=output) as tmp:
        tmp = Path(tmp)
        digest = hashlib.sha256()
        with model.open("rb") as source, (tmp / "experts.bin").open("xb") as target:
            for row in layout:
                for expert in range(experts):
                    data = moe_pack.read_expert(source, row, expert)
                    target.write(data)
                    digest.update(data)
                print(f"packed trunk layer {row['layer'] + 1}/{TRUNK_LAYERS}", flush=True)
            padding = bytes(padded - total)
            target.write(padding)
            digest.update(padding)
            target.flush()
            os.fsync(target.fileno())
        (tmp / "native_experts.txt").write_text(moe_pack.index_text(layout), encoding="ascii")
        metadata = {
            "format": "native-experts-v3",
            "model_family": "qwen35moe",
            "mode": "trunk-only",
            "source": model.name,
            "source_bytes": model.stat().st_size,
            "layers": TRUNK_LAYERS,
            "experts_per_layer": experts,
            "mtp_layer_preserved_in_dense": MTP_LAYER,
            "bytes": padded,
            "sha256": digest.hexdigest(),
            "weights": "unaltered source bytes",
        }
        (tmp / "expert-pack.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for name in PACK_NAMES:
                destination = output / name
                os.link(tmp / name, destination)
                published.append(destination)
        except OSError:
            for path in published:
                path.unlink()
            raise
    return metadata


def write_trunk_stripped_dense(model, output):
    reader = GGUFReader(model)
    writer = GGUFWriter(output, reader.fields["general.architecture"].contents(), endianess=reader.endianess)
    writer.data_alignment = int(reader.alignment)
    writer.kv_data[0].clear()
    for key, field in reader.fields.items():
        if key not in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            writer.add_key_value(key, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_bool("betelgeuse.stripped_experts", True)
    writer.add_bool("betelgeuse.qwen35_trunk_only", True)
    writer.add_uint32("betelgeuse.trunk_layers", TRUNK_LAYERS)
    writer.add_uint32("betelgeuse.mtp_layer", MTP_LAYER)
    writer.add_uint32("betelgeuse.expert_alignment", 4096)

    trunk_experts = []
    dense = []
    for tensor in reader.tensors:
        match = moe_pack.EXPERT_PATTERN.fullmatch(tensor.name)
        if match and int(match.group(1)) < TRUNK_LAYERS:
            trunk_experts.append(tensor)
        else:
            dense.append(tensor)
    if not trunk_experts or not dense:
        raise ValueError("GGUF must contain trunk experts and dense/MTP tensors")
    tensors = dense + trunk_experts
    offsets = {}
    offset = 0
    for tensor in tensors:
        offsets[tensor.name] = offset
        offset += writer.ggml_pad(tensor.n_bytes, writer.data_alignment)
        writer.add_tensor_info(tensor.name, tuple(reversed(tensor.shape)), np.dtype(np.float32),
                               tensor.n_bytes, raw_dtype=tensor.tensor_type)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        target = writer.fout[0]
        byte_order = "<" if reader.endianess == GGUFEndian.LITTLE else ">"
        for tensor in tensors:
            for part in tensor.field.parts[:-1]:
                target.write(part.tobytes())
            target.write(struct.pack(byte_order + "Q", offsets[tensor.name]))
        writer.write_padding(target, target.tell())
        with model.open("rb") as source:
            for tensor in dense:
                moe_pack.copy_tensor_bytes(source, target, tensor.data_offset, tensor.n_bytes)
                writer.write_padding(target, tensor.n_bytes)
        target.flush()
        os.fsync(target.fileno())
    finally:
        writer.close()


def verify(model, output, all_experts=False):
    layout, experts, total = qwen35_layout(model)
    pack = output / "experts.bin"
    expected_size = (total + 4095) // 4096 * 4096
    if pack.stat().st_size != expected_size:
        raise ValueError("pack byte size does not match Qwen3.5 trunk layout")
    metadata = json.loads((output / "expert-pack.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "format": "native-experts-v3",
        "model_family": "qwen35moe",
        "mode": "trunk-only",
        "layers": TRUNK_LAYERS,
        "experts_per_layer": experts,
        "mtp_layer_preserved_in_dense": MTP_LAYER,
        "bytes": expected_size,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("expert-pack.json does not describe the Qwen3.5 trunk-only pack")
    digest = hashlib.sha256()
    with pack.open("rb") as packed:
        for chunk in iter(lambda: packed.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata.get("sha256"):
        raise ValueError("experts.bin SHA-256 does not match expert-pack.json")
    actual = [line.split() for line in (output / "native_experts.txt").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    expected = [line.split() for line in moe_pack.index_text(layout).splitlines()[1:]]
    if actual != expected:
        raise ValueError("native expert index does not match the 40-layer trunk")
    selected = range(experts) if all_experts else sorted({0, experts // 2, experts - 1})
    checked = 0
    with model.open("rb") as source, pack.open("rb") as packed:
        for row in layout:
            for expert in selected:
                packed.seek(row["offset"] + expert * row["blob"])
                if packed.read(row["blob"]) != moe_pack.read_expert(source, row, expert):
                    raise ValueError(f"weight mismatch: layer {row['layer']}, expert {expert}")
                checked += 1
    return {"verified_experts": checked, "total_trunk_experts": TRUNK_LAYERS * experts,
            "all": all_experts}


def build(model, output):
    model = model.resolve(strict=True)
    layout, experts, total = qwen35_layout(model)
    output.mkdir(parents=True, exist_ok=True)
    required_space = model.stat().st_size + 2 * 1024 * 1024
    if shutil.disk_usage(output).free < required_space:
        raise OSError(f"not enough free space: need approximately {required_space / 2**30:.2f} GiB for the complete bundle")
    names = (DENSE_NAME,) + PACK_NAMES
    if any(os.path.lexists(output / name) for name in names):
        raise FileExistsError("bundle output already exists; choose an empty directory")
    with tempfile.TemporaryDirectory(prefix=".qwen35-bundle-", dir=output) as tmp:
        tmp = Path(tmp)
        metadata = write_pack(model, tmp, layout, experts)
        write_trunk_stripped_dense(model, tmp / DENSE_NAME)
        published = []
        try:
            for name in names:
                os.link(tmp / name, output / name)
                published.append(output / name)
        except OSError:
            for path in published:
                path.unlink()
            raise
    return {"dense_file": DENSE_NAME, "dense_bytes": (output / DENSE_NAME).stat().st_size,
            "expert_bytes": metadata["bytes"], "trunk_layers": TRUNK_LAYERS,
            "mtp_layer_preserved": MTP_LAYER, "experts_per_layer": experts,
            "ple_companion": None, "pack_sha256": metadata["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="original single-file Qwen3.5-35B-A3B GGUF")
    parser.add_argument("--output", type=Path, required=True, help="new, empty bundle directory")
    parser.add_argument("--dry-run", action="store_true", help="inspect the exact 40-layer trunk layout")
    parser.add_argument("--verify", action="store_true", help="sample-check the existing trunk pack")
    parser.add_argument("--verify-all", action="store_true", help="byte-verify all 10,240 trunk experts")
    args = parser.parse_args()
    if args.verify and args.verify_all:
        parser.error("--verify and --verify-all are mutually exclusive")
    try:
        model = args.model.resolve(strict=True)
        if args.dry_run:
            layout, experts, total = qwen35_layout(model)
            result = {"trunk_layers": len(layout), "experts_per_layer": experts,
                      "trunk_experts": len(layout) * experts,
                      "expert_payload_bytes": total,
                      "pack_bytes_aligned_4096": (total + 4095) // 4096 * 4096,
                      "mtp_layer_preserved": MTP_LAYER, "ple_required": False}
        elif args.verify or args.verify_all:
            result = verify(model, args.output, args.verify_all)
        else:
            result = build(model, args.output)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
