"""Serving API key and bearer-token helpers."""
import os
import re
import unittest

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class TestServingSecretHashing(unittest.TestCase):
    def test_hash_secret_is_deterministic_and_not_plaintext(self):
        from app.model_gateway.auth import hash_secret

        digest = hash_secret("mk_live_example")

        self.assertEqual(digest, hash_secret("mk_live_example"))
        self.assertNotEqual(digest, "mk_live_example")
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_generate_api_key_returns_plaintext_and_storage_fields(self):
        from app.model_gateway.auth import generate_api_key, hash_secret

        plaintext, digest, prefix, last4 = generate_api_key()

        self.assertTrue(plaintext.startswith("mk_live_"))
        self.assertGreaterEqual(len(plaintext), len("mk_live_") + 32)
        self.assertEqual(prefix, "mk_live")
        self.assertEqual(last4, plaintext[-4:])
        self.assertEqual(digest, hash_secret(plaintext))
        self.assertNotIn(plaintext, digest)
        self.assertIsNotNone(re.match(r"^[0-9a-f]{64}$", digest))


class TestBearerParsing(unittest.TestCase):
    def test_extract_bearer_token(self):
        from app.model_gateway.auth import extract_bearer_token

        self.assertEqual(extract_bearer_token("Bearer abc"), "abc")
        self.assertEqual(extract_bearer_token("bearer abc"), "abc")
        self.assertEqual(extract_bearer_token(" Bearer abc "), "abc")
        self.assertIsNone(extract_bearer_token("abc"))
        self.assertIsNone(extract_bearer_token("Bearer"))
        self.assertIsNone(extract_bearer_token(None))

    def test_admin_token_check(self):
        from app.model_gateway.auth import require_admin_token

        class Config:
            MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN = "admin-secret"

        require_admin_token(Config, "Bearer admin-secret")
        with self.assertRaises(PermissionError):
            require_admin_token(Config, "Bearer wrong")

    def test_admin_token_missing_is_runtime_error(self):
        from app.model_gateway.auth import require_admin_token

        class Config:
            MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN = ""

        with self.assertRaises(RuntimeError):
            require_admin_token(Config, "Bearer anything")


if __name__ == "__main__":
    unittest.main()
