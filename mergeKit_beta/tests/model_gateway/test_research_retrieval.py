import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class _FakeEncoder:
    def encode(self, texts):
        vectors = {
            "query": [1.0, 0.0],
            "relevant alpha": [0.9, 0.1],
            "irrelevant beta": [0.0, 1.0],
            "foreign alpha": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class RetrievalTestCase(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from app.extensions import db
        from app.model_gateway import models as gateway_models  # noqa: F401

        self.root = tempfile.mkdtemp(prefix="research_retrieval_")
        self.app = Flask(__name__)
        self.app.config.update(
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(self.root, 'gateway.sqlite')}",
            SQLALCHEMY_BINDS={"model_gateway": f"sqlite:///{os.path.join(self.root, 'gateway.sqlite')}"},
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        self.db = db
        db.init_app(self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

        from app.model_gateway.models import ResearchChunk, ResearchFile, ServingApiKey

        expiry = datetime.utcnow() + timedelta(hours=1)
        self.db.session.add_all([
            ServingApiKey(id="key-a", key_hash="a", prefix="mk_live", last4="0001", owner_label="a"),
            ServingApiKey(id="key-b", key_hash="b", prefix="mk_live", last4="0002", owner_label="b"),
            ResearchFile(id="file-a", api_key_id="key-a", original_name="a.pdf", source_kind="upload", status="ready", expires_at=expiry),
            ResearchFile(id="file-b", api_key_id="key-b", original_name="b.pdf", source_kind="upload", status="ready", expires_at=expiry),
            ResearchChunk(id="chunk-a1", file_id="file-a", api_key_id="key-a", ordinal=0, text="relevant alpha", locator={"kind": "page", "value": 3}, expires_at=expiry),
            ResearchChunk(id="chunk-a2", file_id="file-a", api_key_id="key-a", ordinal=1, text="irrelevant beta", locator={"kind": "page", "value": 4}, expires_at=expiry),
            ResearchChunk(id="chunk-b1", file_id="file-b", api_key_id="key-b", ordinal=0, text="foreign alpha", locator={"kind": "page", "value": 9}, expires_at=expiry),
        ])
        self.db.session.commit()

    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.ctx.pop()
        shutil.rmtree(self.root, ignore_errors=True)


class TestResearchRetrieval(RetrievalTestCase):
    def test_uses_persisted_chunk_vectors_and_only_encodes_the_question(self):
        from app.model_gateway.retrieval import retrieve_research_evidence
        from app.model_gateway.vectors import save_chunk_vectors

        save_chunk_vectors(self.root, "file-a", np.asarray([[0.9, 0.1], [0.0, 1.0]], dtype=np.float32))

        class QueryOnlyEncoder:
            def encode(self, texts):
                self.assertEqual(texts, ["query"])
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

            def assertEqual(self, actual, expected):
                if actual != expected:
                    raise AssertionError(f"unexpected embedding batch: {actual}")

        evidence = retrieve_research_evidence(
            self.db.session, QueryOnlyEncoder(), "key-a", ["file-a"], "query", limit=2, runtime_root=self.root
        )

        self.assertEqual([item["chunk_id"] for item in evidence], ["chunk-a1", "chunk-a2"])

    def test_returns_ranked_citation_evidence_only_from_key_owned_files(self):
        from app.model_gateway.retrieval import retrieve_research_evidence

        evidence = retrieve_research_evidence(
            self.db.session, _FakeEncoder(), "key-a", ["file-a", "file-b"], "query", limit=2
        )

        self.assertEqual([item["chunk_id"] for item in evidence], ["chunk-a1", "chunk-a2"])
        self.assertEqual(evidence[0]["locator"], {"kind": "page", "value": 3})
        self.assertEqual(evidence[0]["file_id"], "file-a")
        self.assertNotIn("foreign alpha", [item["text"] for item in evidence])


if __name__ == "__main__":
    unittest.main()
