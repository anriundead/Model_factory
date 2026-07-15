import os
import subprocess
import sys
import unittest

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class TestGatewayMigrationScript(unittest.TestCase):
    def test_script_is_runnable_from_project_root(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        result = subprocess.run(
            [sys.executable, "scripts/migrate_model_gateway_storage.py", "--help"],
            cwd=root,
            env={**os.environ, "MERGEKIT_CLI_SCRIPT": "1"},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source", result.stdout)
        self.assertIn("--target", result.stdout)


if __name__ == "__main__":
    unittest.main()
