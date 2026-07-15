import json
import os
import tempfile
import unittest


class TestMergedTokenizerRepair(unittest.TestCase):
    def test_replaces_invalid_output_tokenizer_from_matching_valid_parent(self):
        from merge_manager import repair_merged_tokenizer

        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "output")
            parent = os.path.join(root, "parent")
            os.makedirs(output)
            os.makedirs(parent)
            config = {"vocab_size": 10, "bos_token_id": 1, "eos_token_id": 2}
            for directory in (output, parent):
                with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
                    json.dump(config, handle)
            with open(os.path.join(output, "tokenizer.json"), "w", encoding="utf-8") as handle:
                handle.write('{"broken": true} trailing')
            with open(os.path.join(parent, "tokenizer.json"), "w", encoding="utf-8") as handle:
                json.dump({"valid": True}, handle)
            with open(os.path.join(parent, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_max_length": 32}, handle)

            repaired_from = repair_merged_tokenizer(output, [parent])

            self.assertEqual(repaired_from, parent)
            with open(os.path.join(output, "tokenizer.json"), encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"valid": True})
            self.assertTrue(os.path.isfile(os.path.join(output, "tokenizer_config.json")))


if __name__ == "__main__":
    unittest.main()
