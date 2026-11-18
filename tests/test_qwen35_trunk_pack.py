import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen35_trunk_pack", ROOT / "tools/qwen35_trunk_pack.py")
qwen35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen35)
from gguf import GGUFReader, GGUFWriter


class HeaderOnlyReader(GGUFReader):
    def _build_tensors(self, start_offs, fields):
        self.descriptors = {bytes(field.parts[1]).decode(): field for field in fields}


class TestQwen35TrunkPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "qwen35.gguf"
        self.output = self.root / "bundle"

    def write_model(self):
        writer = GGUFWriter(self.model, "qwen35moe")
        writer.add_string("general.name", "synthetic-qwen35moe")
        writer.add_uint32("qwen35moe.block_count", 41)
        writer.add_tensor("token_embd.weight", np.arange(16, dtype=np.float32).reshape(4, 4))
        for layer in range(41):
            for role in ("gate", "up", "down"):
                values = np.full((2, 2, 256), layer + (0 if role == "gate" else 1), dtype=np.float32)
                writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_layout_excludes_mtp_layer(self):
        self.write_model()
        layout, experts, total = qwen35.qwen35_layout(self.model)
        self.assertEqual(len(layout), 40)
        self.assertEqual(experts, 256)
        self.assertEqual([row["layer"] for row in layout], list(range(40)))
        self.assertEqual(total, sum(row["blob"] * experts for row in layout))

    def test_bundle_keeps_mtp_payload_and_strips_only_trunk_payload(self):
        self.write_model()
        result = qwen35.build(self.model, self.output)
        self.assertEqual(result["trunk_layers"], 40)
        self.assertEqual(result["mtp_layer_preserved"], 40)
        self.assertIsNone(result["ple_companion"])
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {qwen35.DENSE_NAME, "experts.bin", "expert-pack.json", "native_experts.txt"})
        self.assertEqual(qwen35.verify(self.model, self.output, all_experts=True)["verified_experts"], 40 * 256)

        source = GGUFReader(self.model)
        dense = HeaderOnlyReader(self.output / qwen35.DENSE_NAME)
        self.assertTrue(dense.fields["betelgeuse.qwen35_trunk_only"].contents())
        self.assertEqual(dense.fields["betelgeuse.trunk_layers"].contents(), 40)
        self.assertEqual(dense.fields["betelgeuse.mtp_layer"].contents(), 40)
        self.assertNotIn("betelgeuse.ple_file", dense.fields)
        source_tensors = {t.name: t for t in source.tensors}
        with (self.output / qwen35.DENSE_NAME).open("rb") as stripped:
            for name, descriptor in dense.descriptors.items():
                tensor = source_tensors[name]
                offset = dense.data_offset + int(descriptor.parts[-1][0])
                if name.startswith("blk.40.ffn_") and name.endswith("_exps.weight"):
                    self.assertLessEqual(offset + tensor.n_bytes, (self.output / qwen35.DENSE_NAME).stat().st_size)
                    stripped.seek(offset)
                    self.assertEqual(stripped.read(tensor.n_bytes), tensor.data.tobytes())
                elif ".ffn_" in name and "_exps.weight" in name:
                    self.assertGreaterEqual(offset, (self.output / qwen35.DENSE_NAME).stat().st_size)
                else:
                    self.assertLessEqual(offset + tensor.n_bytes, (self.output / qwen35.DENSE_NAME).stat().st_size)

    def test_rejects_non_qwen35_layer_count(self):
        self.write_model()
        # A valid 41-layer source is required; changing the declared architecture
        # depth must not cause a silent 40-layer truncation.
        original_layout = qwen35.moe_pack.expert_layout
        with mock.patch.object(qwen35.moe_pack, "expert_layout",
                               side_effect=lambda model: (original_layout(model)[0][:40], 256, 1)):
            with self.assertRaisesRegex(ValueError, "requires 41 expert layers"):
                qwen35.qwen35_layout(self.model)


if __name__ == "__main__":
    unittest.main()
