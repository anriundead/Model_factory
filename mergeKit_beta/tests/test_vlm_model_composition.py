import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from evolution.vendor.vlm_merge.model_composition import (
    load_merged_language_model_on_cpu,
    materialize_full_vlm,
    replace_language_model_weights,
)


class FakeLanguage(torch.nn.Module):
    def __init__(self, fill=1.0, size=2):
        super().__init__()
        self.proj = torch.nn.Linear(size, size, bias=False)
        torch.nn.init.constant_(self.proj.weight, fill)


class WrongLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.nn.Linear(2, 2, bias=False)


class FakeVlm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeLanguage()
        self.visual = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.constant_(self.visual.weight, 3.0)


class FakeQwenVlm(FakeVlm):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.constant_(self.lm_head.weight, 4.0)


class WrappedQwenLanguage(torch.nn.Module):
    def __init__(self, fill=1.0, tower_size=2, head_size=2, include_head=True):
        super().__init__()
        self.model = FakeLanguage(fill, tower_size)
        if include_head:
            self.lm_head = torch.nn.Linear(2, head_size, bias=False)
            torch.nn.init.constant_(self.lm_head.weight, fill)


class SaveableQwenVlm(FakeQwenVlm):
    def save_pretrained(self, output_dir):
        Path(output_dir, "model.safetensors").write_text("model", encoding="utf-8")


class SaveableVlm(FakeVlm):
    def save_pretrained(self, output_dir):
        Path(output_dir, "model.safetensors").write_text("model", encoding="utf-8")


class FakeProcessor:
    def save_pretrained(self, output_dir):
        Path(output_dir, "processor.json").write_text("processor", encoding="utf-8")


class CompositionTest(unittest.TestCase):
    def test_loads_temporary_merged_language_model_on_cpu(self):
        sentinel = object()
        with mock.patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=sentinel,
        ) as load:
            result = load_merged_language_model_on_cpu("merged", torch.bfloat16)

        self.assertIs(result, sentinel)
        load.assert_called_once_with(
            "merged",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )

    def test_replaces_exact_language_state_and_preserves_visual_state(self):
        vlm = FakeVlm()
        merged = FakeLanguage(fill=7.0)
        visual_before = vlm.visual.weight.detach().clone()

        replace_language_model_weights(vlm, merged)

        self.assertTrue(torch.equal(vlm.language_model.proj.weight, merged.proj.weight))
        self.assertTrue(torch.equal(vlm.visual.weight, visual_before))

    def test_rejects_missing_or_mismatched_language_tensor(self):
        with self.assertRaisesRegex(ValueError, "architecture_mismatch"):
            replace_language_model_weights(FakeVlm(), WrongLanguage())

    def test_replaces_qwen_wrapper_tower_and_top_level_head(self):
        vlm = FakeQwenVlm()
        merged = WrappedQwenLanguage(fill=7.0)
        visual_before = vlm.visual.weight.detach().clone()

        replace_language_model_weights(vlm, merged)

        self.assertTrue(torch.equal(vlm.language_model.proj.weight, merged.model.proj.weight))
        self.assertTrue(torch.equal(vlm.lm_head.weight, merged.lm_head.weight))
        self.assertTrue(torch.equal(vlm.visual.weight, visual_before))

    def test_rejects_qwen_wrapper_without_top_level_head(self):
        with self.assertRaisesRegex(ValueError, "architecture_mismatch"):
            replace_language_model_weights(FakeQwenVlm(), WrappedQwenLanguage(include_head=False))

    def test_rejects_qwen_wrapper_with_mismatched_head(self):
        with self.assertRaisesRegex(ValueError, "architecture_mismatch"):
            replace_language_model_weights(FakeQwenVlm(), WrappedQwenLanguage(head_size=3))

    def test_rejects_qwen_wrapper_with_mismatched_tower(self):
        with self.assertRaisesRegex(ValueError, "architecture_mismatch"):
            replace_language_model_weights(FakeQwenVlm(), WrappedQwenLanguage(tower_size=3))

    def test_materialization_does_not_call_cuda_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            with mock.patch(
                "transformers.AutoModelForImageTextToText.from_pretrained",
                return_value=SaveableQwenVlm(),
            ), mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=WrappedQwenLanguage(),
            ), mock.patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=FakeProcessor(),
            ), mock.patch("torch.cuda.is_available") as is_available, mock.patch(
                "torch.cuda.empty_cache"
            ) as empty_cache:
                materialize_full_vlm("merged", "base", output_dir, "float32")

            self.assertTrue(os.path.isfile(os.path.join(output_dir, "model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "processor.json")))
            is_available.assert_not_called()
            empty_cache.assert_not_called()

    def test_failed_materialization_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            marker = os.path.join(output_dir, "keep.txt")
            Path(marker).write_text("keep", encoding="utf-8")
            with mock.patch(
                "transformers.AutoModelForImageTextToText.from_pretrained",
                return_value=SaveableVlm(),
            ) as load_vlm, mock.patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=FakeLanguage(),
            ), mock.patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=FakeProcessor(),
            ):
                with self.assertRaisesRegex(FileExistsError, "output_dir"):
                    materialize_full_vlm("merged", "base", output_dir, "float32")

            self.assertEqual(Path(marker).read_text(encoding="utf-8"), "keep")
            load_vlm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
