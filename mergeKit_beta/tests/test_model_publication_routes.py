import os
import queue
import tempfile
import unittest
from types import SimpleNamespace

from flask import Flask


class PublicationRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.recipes = os.path.join(self.tmpdir.name, "recipes")
        os.makedirs(self.recipes)
        with open(os.path.join(self.recipes, "valid.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        self.state = SimpleNamespace(
            recipes_dir=self.recipes,
            tasks={},
            task_queue=queue.PriorityQueue(),
            scheduler_lock=__import__("threading").RLock(),
            config=SimpleNamespace(PUBLISHED_MODELS_PATH=os.path.join(self.tmpdir.name, "published")),
        )
        self.services = SimpleNamespace()
        app = Flask(__name__)
        app.config.update(
            TESTING=True,
            MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN="admin-token",
            SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
            SQLALCHEMY_BINDS={"model_gateway": "sqlite:///:memory:"},
        )
        from app.extensions import db
        from app.models import Model, Task
        from app.routes import register_routes

        db.init_app(app)
        with app.app_context():
            db.create_all()
            db.session.add(Model(id="model-1", name="core", path=os.path.join(self.tmpdir.name, "core"), source="base"))
            db.session.commit()
        register_routes(app, self.state, self.services, SimpleNamespace())
        self.app = app
        self.Task = Task

    def tearDown(self):
        self.tmpdir.cleanup()

    def _post(self, path, payload, **headers):
        return self.app.test_client().post(path, json=payload, headers=headers)

    def test_admin_and_idempotency_contract(self):
        payload = {"source_type": "recipe", "recipe_path": "valid.json", "display_name": "published"}
        response = self._post("/api/model-publications", payload)
        self.assertIn(response.status_code, (401, 503))

        headers = {"Authorization": "Bearer admin-token"}
        self.assertEqual(self._post("/api/model-publications", payload, **headers).status_code, 400)
        headers["Idempotency-Key"] = "key-1"
        first = self._post("/api/model-publications", payload, **headers)
        self.assertEqual(first.status_code, 202)
        second = self._post("/api/model-publications", payload, **headers)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.get_json()["task"]["id"], second.get_json()["task"]["id"])
        conflict = self._post(
            "/api/model-publications", {**payload, "display_name": "different"}, **headers
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.get_json()["error"]["code"], "idempotency_conflict")

    def test_rejects_absolute_recipe_and_gpu2_and_handles_cancellation(self):
        headers = {"Authorization": "Bearer admin-token", "Idempotency-Key": "key-2"}
        absolute = self._post(
            "/api/model-publications",
            {"source_type": "recipe", "recipe_path": os.path.join(self.recipes, "valid.json"), "display_name": "published"},
            **headers,
        )
        self.assertEqual(absolute.status_code, 400)

        created = self._post(
            "/api/model-publications",
            {"source_type": "recipe", "recipe_path": "valid.json", "display_name": "published"},
            **headers,
        ).get_json()["task"]
        validate = self._post(
            "/api/model-publications/%s/validate" % created["id"], {"gpu_ids": [2]},
            Authorization="Bearer admin-token",
        )
        self.assertEqual(validate.status_code, 400)
        canceled = self._post(
            "/api/model-publications/%s/cancel" % created["id"], {}, Authorization="Bearer admin-token"
        )
        self.assertEqual(canceled.status_code, 200)
        self.assertEqual(canceled.get_json()["task"]["status"], "canceled")

        with self.app.app_context():
            task = self.Task.query.get(created["id"])
            task.config = {**task.config, "commit_in_progress": True}
            task.status = "validating"
            from app.extensions import db
            db.session.commit()
        blocked = self._post(
            "/api/model-publications/%s/cancel" % created["id"], {}, Authorization="Bearer admin-token"
        )
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.get_json()["error"]["code"], "commit_in_progress")
