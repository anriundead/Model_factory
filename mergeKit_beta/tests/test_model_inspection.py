import json
import os
import sys
import tempfile
import unittest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from app.model_inspection import (  # noqa: E402
    assert_language_compatible,
    inspect_model,
    model_weight_fingerprint,
    resolve_vlm_base,
)


class ModelInspectionTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.text_path = self.model_dir(
            "text",
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "vocab_size": 152064,
            },
            ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"],
        )
        self.vlm_path = self.complete_vlm("vlm")

    def tearDown(self):
        self._tmpdir.cleanup()

    def write_json(self, path, name, value):
        with open(os.path.join(path, name), "w", encoding="utf-8") as handle:
            json.dump(value, handle)

    def model_dir(self, name, config, weight_keys):
        path = os.path.join(self._tmpdir.name, name)
        os.makedirs(path)
        self.write_json(path, "config.json", config)
        self.write_json(
            path,
            "model.safetensors.index.json",
            {"weight_map": {key: "model-00001-of-00001.safetensors" for key in weight_keys}},
        )
        with open(os.path.join(path, "model-00001-of-00001.safetensors"), "wb") as handle:
            handle.write((name + "-weights").encode("utf-8"))
        return path

    def complete_vlm(self, name, visual_key="visual.blocks.0.attn.qkv.weight"):
        path = self.model_dir(
            name,
            {
                "model_type": "qwen2_5_vl",
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "vision_config": {"model_type": "qwen2_5_vl"},
                "text_config": {
                    "hidden_size": 3584,
                    "num_hidden_layers": 28,
                    "vocab_size": 152064,
                },
                "image_token_id": 151655,
            },
            [visual_key, "model.language_model.layers.0.self_attn.q_proj.weight"],
        )
        self.write_json(path, "processor_config.json", {"processor_class": "Qwen2_5_VLProcessor"})
        return path

    def test_textonly_name_does_not_override_text_config(self):
        path = self.model_dir(
            "Qwen2.5-VL-7B-TextOnly",
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "vocab_size": 152064,
            },
            ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"],
        )

        info = inspect_model(path)

        self.assertFalse(info.is_vlm)
        self.assertFalse(info.is_complete_vlm)

    def test_complete_vlm_requires_config_processor_tokens_and_visual_weights(self):
        info = inspect_model(self.vlm_path)

        self.assertTrue(info.is_vlm)
        self.assertTrue(info.is_complete_vlm)
        self.assertGreater(info.visual_weight_count, 0)

    def test_preprocessor_config_processor_class_is_supported(self):
        path = self.complete_vlm("qwen-preprocessor")
        os.unlink(os.path.join(path, "processor_config.json"))
        self.write_json(path, "preprocessor_config.json", {"processor_class": "Qwen2_5_VLProcessor"})

        info = inspect_model(path)

        self.assertEqual(info.processor_class, "Qwen2_5_VLProcessor")
        self.assertTrue(info.is_complete_vlm)

    def test_preprocessor_image_processor_type_is_processor_evidence(self):
        path = self.complete_vlm("qwen-image-processor")
        os.unlink(os.path.join(path, "processor_config.json"))
        self.write_json(path, "preprocessor_config.json", {"image_processor_type": "Qwen2VLImageProcessor"})

        info = inspect_model(path)

        self.assertEqual(info.processor_class, "Qwen2VLImageProcessor")
        self.assertTrue(info.is_complete_vlm)

    def test_known_visual_weight_prefixes_are_recognized(self):
        prefixes = (
            "visual.blocks.0.weight",
            "model.visual.blocks.0.weight",
            "vision_tower.blocks.0.weight",
            "model.vision_tower.blocks.0.weight",
            "vision_model.blocks.0.weight",
        )

        for index, visual_key in enumerate(prefixes):
            with self.subTest(visual_key=visual_key):
                info = inspect_model(self.complete_vlm("prefix-%d" % index, visual_key))
                self.assertTrue(info.is_complete_vlm)
                self.assertEqual(info.visual_weight_count, 1)

    def test_arbitrary_architecture_name_containing_vl_is_not_a_vlm_signal(self):
        path = self.model_dir(
            "neutral",
            {
                "model_type": "custom_text",
                "architectures": ["NovelVLEncoderForCausalLM"],
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "vocab_size": 152064,
            },
            ["model.layers.0.self_attn.q_proj.weight"],
        )

        info = inspect_model(path)

        self.assertFalse(info.is_vlm)
        self.assertFalse(info.is_complete_vlm)

    def test_resolve_uses_first_complete_parent_and_fails_without_one(self):
        selected = resolve_vlm_base({"model_paths": [self.text_path, self.vlm_path]})

        self.assertEqual(selected.path, os.path.realpath(self.vlm_path))
        with self.assertRaisesRegex(ValueError, "vlm_base_missing"):
            resolve_vlm_base({"model_paths": [self.text_path]})

    def test_incomplete_admin_override_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "vlm_base_missing"):
            resolve_vlm_base(
                {"model_paths": [self.vlm_path]},
                override_path=self.text_path,
            )

    def test_recorded_vlm_base_fingerprint_mismatch_fails(self):
        expected_hash = inspect_model(self.vlm_path).config_sha256
        config_path = os.path.join(self.vlm_path, "config.json")
        with open(config_path, encoding="utf-8") as handle:
            changed_config = json.load(handle)
        changed_config["image_token_id"] = 151656
        self.write_json(self.vlm_path, "config.json", changed_config)

        with self.assertRaisesRegex(ValueError, "source_fingerprint_mismatch"):
            resolve_vlm_base(
                {
                    "vlm_base": {
                        "source_path": self.vlm_path,
                        "config_sha256": expected_hash,
                    },
                    "model_paths": [self.complete_vlm("fallback")],
                }
            )

    def test_weight_fingerprint_changes_when_weight_content_is_replaced(self):
        first = model_weight_fingerprint(self.vlm_path)
        shard = os.path.join(self.vlm_path, "model-00001-of-00001.safetensors")
        with open(shard, "wb") as handle:
            handle.write(b"replacement-weight-content")
        second = model_weight_fingerprint(self.vlm_path)

        self.assertNotEqual(first["weights_sha256"], second["weights_sha256"])
        self.assertEqual(first["source_path"], os.path.realpath(self.vlm_path))
        self.assertEqual(first["weight_files"][0]["path"], "model-00001-of-00001.safetensors")

    def test_weight_fingerprint_includes_safetensors_index_content(self):
        first = model_weight_fingerprint(self.vlm_path)
        index_path = os.path.join(self.vlm_path, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        index["metadata"] = {"total_size": 123}
        self.write_json(self.vlm_path, "model.safetensors.index.json", index)
        second = model_weight_fingerprint(self.vlm_path)

        self.assertNotEqual(first["weights_sha256"], second["weights_sha256"])
        self.assertEqual(second["index_files"][0]["path"], "model.safetensors.index.json")

    def test_recorded_vlm_weight_fingerprint_mismatch_fails(self):
        fingerprint = model_weight_fingerprint(self.vlm_path)
        shard = os.path.join(self.vlm_path, "model-00001-of-00001.safetensors")
        with open(shard, "wb") as handle:
            handle.write(b"same-path-new-weights")

        with self.assertRaisesRegex(ValueError, "source_fingerprint_mismatch"):
            resolve_vlm_base(
                {
                    "vlm_base": {
                        "source_path": self.vlm_path,
                        "config_sha256": inspect_model(self.vlm_path).config_sha256,
                        "weights_sha256": fingerprint["weights_sha256"],
                    },
                    "model_paths": [self.vlm_path],
                }
            )

    def test_recorded_fingerprint_mismatch_precedes_fallback_when_source_is_no_longer_complete(self):
        expected_hash = inspect_model(self.vlm_path).config_sha256
        config_path = os.path.join(self.vlm_path, "config.json")
        with open(config_path, encoding="utf-8") as handle:
            changed_config = json.load(handle)
        changed_config.update({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
        changed_config.pop("vision_config")
        changed_config.pop("image_token_id")
        self.write_json(self.vlm_path, "config.json", changed_config)

        with self.assertRaisesRegex(ValueError, "source_fingerprint_mismatch"):
            resolve_vlm_base(
                {
                    "vlm_base": {
                        "source_path": self.vlm_path,
                        "config_sha256": expected_hash,
                    },
                    "model_paths": [self.complete_vlm("fallback")],
                }
            )

    def test_language_signature_mismatch_is_rejected(self):
        incompatible = self.model_dir(
            "incompatible",
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 28,
                "vocab_size": 152064,
            },
            ["model.layers.0.self_attn.q_proj.weight"],
        )

        with self.assertRaisesRegex(ValueError, "architecture_mismatch"):
            assert_language_compatible([self.text_path, incompatible], inspect_model(self.vlm_path))


if __name__ == "__main__":
    unittest.main()
