import unittest

import torch

from evolution.vendor.vlm_merge.model_composition import replace_language_model_weights


class FakeLanguage(torch.nn.Module):
    def __init__(self, fill=1.0):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
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


class CompositionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
