# Model Publication Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a recoverable publication layer that materializes text and VLM recipes as independent Hugging Face assets, registers them in the model factory, and only exposes serving-compatible assets to Gateway administrators.

**Architecture:** Keep search recipes, published assets, and Gateway services as separate authorities. Pure model inspection lives in one module, VLM language-weight composition lives beside the existing VLM evaluator, and filesystem publication/manifest transactions live in one core publication module. Existing Flask routes and the main task worker only validate requests and dispatch these focused functions.

**Tech Stack:** Python 3 in the `mergenetic` Conda environment, Flask, SQLAlchemy with core SQLite and Gateway PostgreSQL bind, safetensors, Transformers 5.3.0, vLLM 0.7.0, Docker Compose, vanilla JavaScript/CSS, Python `unittest`, Firefox WebDriver.

**Design:** [`MODEL_PUBLICATION_PIPELINE_DESIGN.md`](MODEL_PUBLICATION_PIPELINE_DESIGN.md)

## Global Constraints

- Work from `/home/a/Workspace/Model_factory`; the container project path must remain `/app/ServiceEndFiles/Workspaces/mergeKit_beta`.
- At execution start use `/home/a/.codex/skills/using-git-worktrees/SKILL.md`; do not implement directly in the shared checkpoint branch.
- Use `/home/a/.codex/skills/test-driven-development/SKILL.md` for every behavior change and `/home/a/.codex/skills/systematic-debugging/SKILL.md` for every unexpected failure.
- Use `/home/a/.codex/skills/ponytail/SKILL.md` to keep modules and dependencies minimal.
- For Task 6 use `/home/a/.codex/skills/frontend-design/SKILL.md`, `/home/a/.codex/skills/taste-skill/SKILL.md`, `/home/a/.codex/skills/gsap-core/SKILL.md`, and `/home/a/.codex/skills/gsap-performance/SKILL.md`; reuse the current visual system and add no animation dependency.
- Before completion use `/home/a/.codex/skills/requesting-code-review/SKILL.md` and `/home/a/.codex/skills/verification-before-completion/SKILL.md`.
- Do not modify the VLM/LLM evaluation branch, `HF_DATASETS_TRUST_REMOTE_CODE`, Ray/vLLM TP subprocess isolation, or the Ray/vLLM environment whitelist except where this plan explicitly names a call site.
- Do not upgrade vLLM, Transformers, CUDA, Docker, NVIDIA toolkit, or model dependencies.
- Never use GPU 2. Never choose another GPU implicitly. A real validation call receives explicit GPU IDs after a fresh hardware snapshot.
- Do not start a model, Ray, vLLM, fusion, or evaluation process during Tasks 1-6 automated tests.
- Use the only accepted interpreter for repository tests: `/opt/conda/envs/mergenetic/bin/python` inside `mergekit-beta`.
- Published host path: `/home/a/Model_factory_data/published_models`; container path: `/data/PublishedModels`.
- New recipes only add fields. Keep existing `vlm_path`; add `recipe_schema_version`, `artifact_type`, `capabilities`, and `vlm_base`.
- New Gateway services must reference a formal published `model_id`. Existing service rows remain grandfathered and restart only manually.
- Qwen2.5-VL must be visible to administrators as `published + blocked` under vLLM 0.7.0 and must not appear in `/v1/models` or `/research`.
- Each task ends with focused tests, the full non-GPU regression suite, `git diff --check`, and an independent commit.

## File Map

| File | Responsibility |
|---|---|
| `app/model_inspection.py` | Config-first model type detection, weight-key inspection, language signature checks, VLM base resolution and recipe provenance. |
| `evolution/vendor/vlm_merge/model_composition.py` | Strictly replace the language branch of a full VLM and save an independent Hugging Face model. |
| `app/model_publication.py` | Publication paths, `flock`, hashes, manifest validation, atomic commit, reconciliation and guarded deletion. |
| `app/model_publication_tasks.py` | Publication task phases, existing-model copy, recipe materialization, explicit-GPU functional validation and cancellation checkpoints. |
| `app/routes.py` | Admin-protected publication HTTP contract and delegation from legacy delete routes. |
| `app/services.py` | Dispatch `model_publication` tasks without absorbing publication logic. |
| `app/repositories/__init__.py` | Small task-status and published-model registration helpers. |
| `app/model_gateway/routes.py` | Publishable-asset listing, service creation by `model_id`, and soft service deletion. |
| `app/model_gateway/runtime.py` | Published-root allowance and manifest compatibility recheck before start. |
| `evolution/runner.py` | Persist additive VLM base provenance in future recipes. |
| `merge_manager.py` | Delegate VLM detection and support an explicit recipe output directory. |
| `config.py`, `docker-compose.yml` | Published-model path and bind mount. |
| `templates/model_repo.html` | Asset publication controls and publication-state display. |
| `templates/model_gateway/console.html` | Formal asset selector and read-only capability display. |
| `static/model_gateway/console.js`, `console.css` | Admin asset loading, disabled blocked states, service deletion and accessible feedback. |

---

### Task 0: Isolated Baseline And Rollback Record

**Files:**
- Modify: `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_SKILLS.md`
- Create at final acceptance: `mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260716_MODEL_PUBLICATION.md`
- Runtime evidence only: `mergeKit_beta/logs/model_gateway/acceptance/<timestamp>/`

**Interfaces:**
- Consumes: checkpoint branch containing design commits `ad75a8f` and `41d83e1`.
- Produces: isolated feature worktree, baseline outputs and per-task rollback commit IDs.

- [ ] **Step 1: Create an isolated worktree**

Invoke the installed `using-git-worktrees` skill, then create a branch from the current design HEAD. Use the path selected by that skill; the expected native command is:

```bash
git worktree add ../Model_factory-model-publication -b feature/model-publication-pipeline HEAD
```

Expected: the new worktree is on `feature/model-publication-pipeline`; the original checkpoint worktree remains unchanged.

- [ ] **Step 2: Record skill usage**

Append a table row for this feature to `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_SKILLS.md` with the exact skill paths from Global Constraints and these batches: planning, TDD/backend, debugging, frontend, verification, and review.

- [ ] **Step 3: Capture the non-GPU baseline**

```bash
mkdir -p mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline
git status --short > mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/git-status.txt
docker compose config --quiet
docker compose ps > mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/compose-ps.txt
for endpoint in /healthz /readyz /api/models /api/testset/list /api/history /model_repo /model-gateway /research; do
  curl -fsS -o /dev/null -w "$endpoint %{http_code}\n" "http://127.0.0.1:5000$endpoint"
done
docker compose exec -T mergekit-beta pwd
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
nvidia-smi -L > mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/gpu-list.txt
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits \
  > mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/gpu-memory.csv
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits \
  > mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/gpu-processes.csv
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' \
  > mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/containers.txt
```

Expected: Compose parses; project path is exact; HTTP endpoints return `200`; baseline failures, if any, are recorded before edits; no process is started by these commands.

- [ ] **Step 4: Stop if the baseline is unsafe**

Do not proceed if GPU 2 ownership cannot be identified, Docker has a new unhealthy container, the project path differs, or the baseline service cannot start. Do not change project code to hide a host/runtime failure.

- [ ] **Step 5: Commit the skill record**

```bash
git add mergeKit_beta/docs/model_gateway/IMPLEMENTATION_SKILLS.md
git commit -m "docs: record model publication implementation skills"
```

Rollback: `git revert <task-0-commit>`; baseline evidence is ignored runtime data and is not deleted.

---

### Task 1: Structural Model Inspection

**Files:**
- Create: `mergeKit_beta/app/model_inspection.py`
- Create: `mergeKit_beta/tests/test_model_inspection.py`
- Modify: `mergeKit_beta/app/services.py` in `ModelCompatibilityMixin.model_is_vlm`
- Modify: `mergeKit_beta/merge_manager.py` in `_model_is_vlm`

**Interfaces:**
- Produces: `ModelInspection`, `inspect_model(path)`, `resolve_vlm_base(recipe, override_path=None)`, `assert_language_compatible(model_paths, vlm_inspection)`.
- Consumes: standard Hugging Face `config.json`, tokenizer/processor files, safetensors index or safetensors headers.

- [ ] **Step 1: Write failing config-first tests**

Create `tests/test_model_inspection.py` with temporary model directories covering these exact cases:

```python
class ModelInspectionTest(unittest.TestCase):
    def test_textonly_name_does_not_override_text_config(self):
        path = self.model_dir("Qwen2.5-VL-7B-TextOnly", {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "vocab_size": 152064,
        }, ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"])
        info = inspect_model(path)
        self.assertFalse(info.is_vlm)
        self.assertFalse(info.is_complete_vlm)

    def test_complete_vlm_requires_config_processor_tokens_and_visual_weights(self):
        path = self.model_dir("neutral-name", {
            "model_type": "qwen2_5_vl",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "vision_config": {"model_type": "qwen2_5_vl"},
            "text_config": {"hidden_size": 3584, "num_hidden_layers": 28, "vocab_size": 152064},
            "image_token_id": 151655,
        }, ["visual.blocks.0.attn.qkv.weight", "model.language_model.layers.0.self_attn.q_proj.weight"])
        self.write_json(path, "processor_config.json", {"processor_class": "Qwen2_5_VLProcessor"})
        info = inspect_model(path)
        self.assertTrue(info.is_vlm)
        self.assertTrue(info.is_complete_vlm)
        self.assertGreater(info.visual_weight_count, 0)

    def test_resolve_uses_first_complete_parent_and_fails_without_one(self):
        selected = resolve_vlm_base({"model_paths": [self.text_path, self.vlm_path]})
        self.assertEqual(selected.path, os.path.realpath(self.vlm_path))
        with self.assertRaisesRegex(ValueError, "vlm_base_missing"):
            resolve_vlm_base({"model_paths": [self.text_path]})
```

The helper may create a minimal `.safetensors.index.json` containing a `weight_map`; it must not create fake model weights.

- [ ] **Step 2: Verify the tests fail**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_inspection.py -v
```

Expected: FAIL because `app.model_inspection` does not exist.

- [ ] **Step 3: Implement the minimal inspection module**

Define these stable types and functions:

```python
@dataclass(frozen=True)
class ModelInspection:
    path: str
    model_type: str
    architectures: Sequence[str]
    is_vlm: bool
    is_complete_vlm: bool
    processor_class: str | None
    image_token_ids: dict[str, int]
    visual_weight_count: int
    language_weight_count: int
    language_signature: tuple[int | None, int | None, int | None]
    config_sha256: str

def inspect_model(path: str) -> ModelInspection:
    model_path = os.path.realpath(os.path.abspath(path))
    if not os.path.isdir(model_path):
        raise ValueError(f"model path is not a directory: {path}")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "rb") as handle:
        config_bytes = handle.read()
    config = json.loads(config_bytes.decode("utf-8"))
    weight_keys = _read_weight_keys(model_path)
    architectures = tuple(str(value) for value in config.get("architectures") or ())
    model_type = str(config.get("model_type") or "").lower()
    token_names = ("image_token_id", "vision_start_token_id", "vision_token_id", "video_token_id")
    image_token_ids = {name: int(config[name]) for name in token_names if config.get(name) is not None}
    signal = bool(config.get("vision_config") or image_token_ids or "vl" in model_type or
                  any("vision" in value.lower() or "vl" in value.lower() for value in architectures))
    processor_class = _read_processor_class(model_path)
    visual_count = sum(1 for key in weight_keys if key.startswith(("visual.", "vision_tower.", "model.visual.")))
    language_count = len(weight_keys) - visual_count
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    signature = tuple(text_config.get(name) for name in ("hidden_size", "num_hidden_layers", "vocab_size"))
    return ModelInspection(
        path=model_path,
        model_type=model_type,
        architectures=architectures,
        is_vlm=signal,
        is_complete_vlm=bool(signal and processor_class and image_token_ids and visual_count),
        processor_class=processor_class,
        image_token_ids=image_token_ids,
        visual_weight_count=visual_count,
        language_weight_count=language_count,
        language_signature=signature,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )

def resolve_vlm_base(recipe: dict, override_path: str | None = None) -> ModelInspection:
    recorded = recipe.get("vlm_base") if isinstance(recipe.get("vlm_base"), dict) else {}
    candidates = [override_path, recorded.get("source_path"), recipe.get("vlm_path")]
    candidates.extend(recipe.get("model_paths") or [])
    for candidate in candidates:
        if not candidate:
            continue
        inspection = inspect_model(candidate)
        if inspection.is_complete_vlm:
            return inspection
    raise ValueError("vlm_base_missing: no complete VLM parent was found")

def assert_language_compatible(model_paths: list[str], vlm_inspection: ModelInspection) -> None:
    expected = vlm_inspection.language_signature
    if None in expected:
        raise ValueError("architecture_mismatch: VLM language signature is incomplete")
    for path in model_paths:
        actual = inspect_model(path).language_signature
        if actual != expected:
            raise ValueError(f"architecture_mismatch: {path} has {actual}, expected {expected}")
```

Implementation rules:

- Resolve and validate directories with `os.path.realpath`.
- Read `config.json` before considering a directory name.
- Read weight keys from `*.safetensors.index.json`; for a single shard call `safetensors.safe_open(path, framework="pt", device="cpu").keys()`.
- `is_vlm` requires a config/architecture/token signal; `is_complete_vlm` additionally requires processor metadata and at least one `visual.*` or equivalent architecture-known visual key.
- `assert_language_compatible` compares hidden size, layer count and vocabulary size and raises a message beginning with `ValueError("architecture_mismatch:")`.
- Compute only `config.json` SHA-256 during preflight; full file hashes belong to publication.

- [ ] **Step 4: Delegate both existing detectors**

Replace the directory-name-first implementations with:

```python
def _model_is_vlm(model_path: str) -> bool:
    from app.model_inspection import inspect_model
    try:
        return inspect_model(model_path).is_vlm
    except (OSError, ValueError):
        return False
```

and equivalent delegation in `ModelCompatibilityMixin.model_is_vlm`. Keep `_infer_lmms_model_backend` unchanged.

- [ ] **Step 5: Run focused and full non-GPU tests**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_inspection.py -v
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
git diff --check
```

Expected: new tests pass; existing tests have no new failure; no GPU process appears.

- [ ] **Step 6: Commit**

```bash
git add mergeKit_beta/app/model_inspection.py mergeKit_beta/tests/test_model_inspection.py \
  mergeKit_beta/app/services.py mergeKit_beta/merge_manager.py
git commit -m "fix: make VLM detection structural"
```

Rollback: `git revert <task-1-commit>` restores both legacy call sites together.

---

### Task 2: VLM Recipe Provenance And Strict Composition

**Files:**
- Create: `mergeKit_beta/evolution/vendor/vlm_merge/model_composition.py`
- Create: `mergeKit_beta/tests/test_vlm_model_composition.py`
- Create: `mergeKit_beta/tests/test_vlm_recipe_contract.py`
- Modify: `mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_eval.py`
- Modify: `mergeKit_beta/evolution/vendor/vlm_merge/vlm_fitness.py`
- Modify: `mergeKit_beta/evolution/runner.py` where `fusion_info` and recipe JSON are written

**Interfaces:**
- Consumes: Task 1 `inspect_model`, `resolve_vlm_base`, `assert_language_compatible`.
- Produces: `language_model_of(vlm)`, `replace_language_model_weights(vlm, merged_lm)`, `materialize_full_vlm(merged_llm_dir, vlm_base_path, output_dir, dtype)`, `build_recipe_vlm_fields(meta, inspection)` and additive v2 recipe fields.

- [ ] **Step 1: Write failing strict-composition tests**

Use small `torch.nn.Module` objects, not downloaded models:

```python
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
```

- [ ] **Step 2: Write failing recipe tests**

Extract recipe enrichment into a callable helper in `evolution.runner` and assert:

```python
enriched = build_recipe_vlm_fields(meta, inspection)
self.assertEqual(enriched["recipe_schema_version"], 2)
self.assertEqual(enriched["artifact_type"], "vlm")
self.assertEqual(enriched["vlm_path"], inspection.path)
self.assertEqual(enriched["vlm_base"]["config_sha256"], inspection.config_sha256)
self.assertIn("best_genotype", enriched)
```

Also assert a VLM/CMMMU task with no complete VLM raises `vlm_base_missing` before the evolution subprocess starts.

- [ ] **Step 3: Verify the tests fail**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_vlm_model_composition.py tests/test_vlm_recipe_contract.py -v
```

Expected: FAIL because the composition module and recipe helper do not exist.

- [ ] **Step 4: Implement strict shared composition**

```python
def language_model_of(vlm):
    target = getattr(vlm, "language_model", None)
    if target is None and hasattr(vlm, "model"):
        target = getattr(vlm.model, "language_model", None)
    if target is None:
        raise ValueError("architecture_mismatch: VLM language model is missing")
    return target

def replace_language_model_weights(vlm, merged_lm) -> None:
    target = language_model_of(vlm)
    source = merged_lm.state_dict()
    expected = target.state_dict()
    if source.keys() != expected.keys():
        raise ValueError("architecture_mismatch: language tensor names differ")
    for name in source:
        if source[name].shape != expected[name].shape:
            raise ValueError(f"architecture_mismatch: {name} shape differs")
    target.load_state_dict(source, strict=True)
```

`materialize_full_vlm` loads the full VLM and merged language model with Transformers, calls the strict helper, saves model and processor to `output_dir`, then releases CPU/GPU objects in `finally`. All heavy imports stay inside the function.

- [ ] **Step 5: Replace duplicate evaluator injection**

In both `run_vlm_eval.py` and `vlm_fitness.py`, replace direct `strict=False` loading with `replace_language_model_weights`. Do not change prompts, datasets, TP subprocess behavior, sample limits or metric calculation.

- [ ] **Step 6: Persist additive recipe provenance**

In `evolution.runner`:

- Resolve a complete VLM for VLM/CMMMU mode before launch.
- Preserve existing `vlm_path`.
- Add `recipe_schema_version=2`, `artifact_type`, `capabilities`, and serialized `vlm_base` to `fusion_info` and the saved recipe.
- Write recipe JSON through a temporary file plus `os.replace`; never delete existing keys.
- Keep text recipes valid with `artifact_type="text"` and no required `vlm_base`.

- [ ] **Step 7: Run focused and full tests**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_vlm_model_composition.py tests/test_vlm_recipe_contract.py -v
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
git diff --check
```

Expected: all pass; no VLM model is loaded and no GPU process begins.

- [ ] **Step 8: Commit**

```bash
git add mergeKit_beta/evolution/vendor/vlm_merge/model_composition.py \
  mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_eval.py \
  mergeKit_beta/evolution/vendor/vlm_merge/vlm_fitness.py \
  mergeKit_beta/evolution/runner.py \
  mergeKit_beta/tests/test_vlm_model_composition.py \
  mergeKit_beta/tests/test_vlm_recipe_contract.py
git commit -m "feat: persist VLM base provenance"
```

Rollback: revert this commit; v1 recipe additions remain readable by old code because `vlm_path` is unchanged.

---

### Task 3: Publication Filesystem, Manifest And Recovery

**Files:**
- Create: `mergeKit_beta/app/model_publication.py`
- Create: `mergeKit_beta/tests/test_model_publication.py`
- Modify: `mergeKit_beta/config.py`
- Modify: `docker-compose.yml`
- Modify: `mergeKit_beta/app/__init__.py`
- Modify: `mergeKit_beta/app/repositories/__init__.py`
- Modify: `mergeKit_beta/app/models.py` source comment only

**Interfaces:**
- Consumes: Task 1 `inspect_model`.
- Produces: `publication_lock`, `atomic_write_json`, `inspect_serving_compatibility`, `build_manifest`, `validate_published_asset`, `commit_staging`, `reconcile_publications`, `delete_published_asset`.

- [ ] **Step 1: Write failing filesystem transaction tests**

Create temporary publication roots and test:

```python
class PublicationFilesystemTest(unittest.TestCase):
    def test_commit_is_atomic_and_manifest_excludes_itself_from_hashes(self):
        manifest = build_manifest(
            self.staging, self.request, self.inspection,
            validation={"structural": {"status": "passed"}},
            compatibility={"serving": {"status": "ready"}},
        )
        committed = commit_staging(self.staging, self.root, manifest, register_fn=self.register)
        final = os.path.join(self.root, committed["publication_id"])
        self.assertFalse(os.path.exists(self.staging))
        self.assertTrue(os.path.isfile(os.path.join(final, "publication_manifest.json")))
        paths = {item["path"] for item in committed["files"]["entries"]}
        self.assertNotIn("publication_manifest.json", paths)

    def test_reconcile_registers_pending_once(self):
        first = reconcile_publications(self.root, register_fn=self.register)
        second = reconcile_publications(self.root, register_fn=self.register)
        self.assertEqual(first["registered"], 1)
        self.assertEqual(second["registered"], 0)

    def test_invalid_hash_is_quarantined(self):
        self.publish_then_modify_weight()
        with self.assertRaisesRegex(PublicationError, "validation_failed"):
            validate_published_asset(self.final_path, full_hash=True)
```

Also test stale staging cleanup records a small JSON diagnostic before deleting large files.

Add a compatibility test that patches `ModelRegistry.inspect_model_cls` to return for
`Qwen2ForCausalLM` and raise `ValueError` for
`Qwen2_5_VLForConditionalGeneration`. Assert the result contains the actual vLLM
version, `ready` for the former, `blocked` plus `unsupported_architecture` for the
latter, and `stale` when a stored manifest version differs from the running version.

- [ ] **Step 2: Verify tests fail**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_publication.py -v
```

Expected: FAIL because `app.model_publication` does not exist.

- [ ] **Step 3: Add configuration and mount**

Add to `Config`:

```python
PUBLISHED_MODELS_PATH = os.path.abspath(
    os.environ.get("MERGEKIT_PUBLISHED_MODELS_PATH", "/data/PublishedModels")
)
```

Add Compose environment and volume:

```yaml
- MERGEKIT_PUBLISHED_MODELS_PATH=/data/PublishedModels
- ${HOST_PUBLISHED_MODELS:-/home/a/Model_factory_data/published_models}:/data/PublishedModels
```

Create the host directory only during execution after checking owner/mode; do not change unrelated permissions.

- [ ] **Step 4: Implement publication primitives**

Implement the lock and atomic JSON primitives exactly as follows:

```python
class PublicationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

@contextmanager
def publication_lock(root: str):
    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".publication.lock")
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
```

Create these additional callables with the exact signatures:

- `inspect_serving_compatibility(architectures: Sequence[str], recorded_version: str | None = None) -> dict`
- `build_manifest(staging: str, request: dict, inspection: ModelInspection, validation: dict, compatibility: dict) -> dict`
- `validate_published_asset(path: str, full_hash: bool = False) -> dict`
- `commit_staging(staging: str, root: str, manifest: dict, register_fn: Callable[[str, dict], object]) -> dict`
- `reconcile_publications(root: str, register_fn: Callable[[str, dict], object]) -> dict`
- `delete_published_asset(publication_id: str, root: str, reference_check_fn: Callable[[str, str], bool], delete_model_fn: Callable[[str], bool]) -> dict`

Use only stdlib `hashlib`, `json`, `os`, `shutil`, `fcntl`, `tempfile`, and existing inspection helpers. `commit_staging` writes `registration_pending`, fsyncs, renames on the same filesystem, calls `register_fn`, and atomically rewrites state to `published`. Registration failure leaves the final directory pending.

`inspect_serving_compatibility` lazily imports `vllm.__version__` and
`vllm.model_executor.models.registry.ModelRegistry`, then calls
`ModelRegistry.inspect_model_cls(list(architectures))`. A successful call returns
`status="ready"`; `ValueError` returns `status="blocked"` and
`reason_code="unsupported_architecture"`. If `recorded_version` differs from the
running version, return `status="stale"` before declaring the model selectable.

- [ ] **Step 5: Add published-model registration**

Add a repository wrapper that calls existing `model_register` with:

```python
source="published"
task_id=manifest["provenance"]["task_id"]
architecture=manifest["model"]["model_type"]
is_vlm=manifest["artifact_type"] == "vlm"
size_bytes=manifest["files"]["total_bytes"]
```

Do not write formal assets to `model_repo/data/models.json`.

- [ ] **Step 6: Reconcile on application startup**

In `create_app`, after the existing model scan and within app context, call `reconcile_publications`. Recovery may register assets and mark compatibility stale; it must never start vLLM or a GPU process.

- [ ] **Step 7: Run focused, Compose and full tests**

```bash
docker compose config --quiet
docker compose exec -T mergekit-beta test -d /data/PublishedModels
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_publication.py -v
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
git diff --check
```

Expected: mount exists; transaction tests pass; startup does not start a model or alter GPU 2.

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml mergeKit_beta/config.py mergeKit_beta/app/__init__.py \
  mergeKit_beta/app/models.py mergeKit_beta/app/repositories/__init__.py \
  mergeKit_beta/app/model_publication.py mergeKit_beta/tests/test_model_publication.py
git commit -m "feat: add atomic model publication storage"
```

Rollback: revert the commit and recreate the container. Do not delete `/home/a/Model_factory_data/published_models`.

---

### Task 4: Publication Tasks, API, Cancellation And Explicit Validation

**Files:**
- Create: `mergeKit_beta/app/model_publication_tasks.py`
- Create: `mergeKit_beta/tests/test_model_publication_tasks.py`
- Create: `mergeKit_beta/tests/test_model_publication_routes.py`
- Modify: `mergeKit_beta/merge_manager.py` in `run_recipe_apply_task`
- Modify: `mergeKit_beta/app/services.py` in worker dispatch/finalization
- Modify: `mergeKit_beta/app/repositories/__init__.py`
- Modify: `mergeKit_beta/app/routes.py`

**Interfaces:**
- Consumes: Tasks 1-3 inspection, composition, manifest and registration functions.
- Produces: `run_model_publication_task(task_id, params, progress, task_control, *, copy_fn, structural_validate_fn)`, `run_publication_validation(task_id, gpu_ids, progress, task_control, *, functional_validate_fn)`, and the `/api/model-publications` HTTP contract.

- [ ] **Step 1: Write failing task-phase tests**

Use temporary real files and injected callables for the expensive model operations:

```python
class PublicationTaskTest(unittest.TestCase):
    def test_without_gpu_stops_at_validating_without_registering(self):
        result = run_model_publication_task(
            "task-a", self.request, self.progress, {"aborted": False},
            copy_fn=self.copy_model, structural_validate_fn=self.structural_validate,
        )
        self.assertEqual(result["status"], "validating")
        self.assertFalse(os.path.exists(self.final_path))

    def test_cancel_before_commit_removes_staging(self):
        control = {"aborted": True}
        result = run_model_publication_task("task-b", self.request, self.progress, control)
        self.assertEqual(result["error_code"], "canceled")
        self.assertFalse(os.path.exists(self.staging))

    def test_validate_requires_explicit_non_gpu2_ids(self):
        with self.assertRaisesRegex(PublicationError, "gpu_selection_required"):
            run_publication_validation(
                "task-a", [], self.progress, {}, functional_validate_fn=self.functional_validate
            )
        with self.assertRaisesRegex(PublicationError, "protected_gpu"):
            run_publication_validation(
                "task-a", [2], self.progress, {}, functional_validate_fn=self.functional_validate
            )
```

Injected functions are used only for phase-control tests; Gate 3/4 later exercise real model code.

- [ ] **Step 2: Write failing route contract tests**

Create a minimal Flask app using `register_routes` and fake state/services. Assert:

- Missing/wrong admin token returns `401` or `503`.
- Missing `Idempotency-Key` returns `400`.
- Same key and payload returns the same task.
- Same key and different payload returns `409 idempotency_conflict`.
- Absolute recipe paths outside `RECIPES_DIR` are rejected.
- `POST /<task_id>/validate` rejects GPU 2.
- queued cancellation is immediate; cancellation during the atomic commit returns
  `409` with `error.code="commit_in_progress"`.

- [ ] **Step 3: Verify tests fail**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_publication_tasks.py tests/test_model_publication_routes.py -v
```

Expected: FAIL because task and route implementations do not exist.

- [ ] **Step 4: Allow explicit recipe output directories**

Extend without changing current callers:

```python
def run_recipe_apply_task(task_id, params, update_progress_callback,
                          task_control=None, skip_register=False,
                          output_dir_override=None):
```

When override is set, write model files there while keeping task metadata/logs under the task directory. Existing merge, recipe apply and generation-two callers continue using the default path.

- [ ] **Step 5: Implement publication task phases**

`run_model_publication_task` must:

1. Resolve a managed recipe or core model ID.
2. Create `.staging/<publication_id>`.
3. Materialize text output or call `materialize_full_vlm` for VLM.
4. Run structural validation and create full hashes.
5. Return `status="validating"` without registration when no explicit GPU validation exists.
6. Check `task_control["aborted"]` between copies, materialization, validation and before lock acquisition.

Use this exact injectable boundary so tests never replace route or worker behavior:

```python
def run_model_publication_task(
    task_id: str,
    params: dict,
    progress: Callable[[int, str], None],
    task_control: dict,
    *,
    copy_fn: Callable = shutil.copytree,
    structural_validate_fn: Callable = validate_staging_model,
) -> dict:
```

`run_publication_validation` must recheck GPU UUID, external processes and memory using `core.gpu_topology`, set `CUDA_VISIBLE_DEVICES` only to explicit IDs, perform real text generation or VLM image/CMMMU validation, then call atomic commit.

```python
def run_publication_validation(
    task_id: str,
    gpu_ids: list[int],
    progress: Callable[[int, str], None],
    task_control: dict,
    *,
    functional_validate_fn: Callable = validate_model_functionally,
) -> dict:
```

- [ ] **Step 6: Add repository status helpers**

Add:

```python
def task_set_status(task_id: str, status: str, *, error: str | None = None,
                    config_patch: dict | None = None, model_path: str | None = None) -> Task | None:
```

Merge `config_patch` into the existing JSON instead of replacing it. Valid publication states are `queued`, `materializing`, `validating`, `registration_pending`, `completed`, `failed`, and `canceled`.

- [ ] **Step 7: Dispatch without changing other task semantics**

Add `model_publication` branches to the main worker. When a publication result is `validating`, preserve that status and do not run generic success/error finalization. All existing `merge`, `merge_evolutionary`, `eval_only`, and `recipe_apply` branches remain unchanged.

For terminal publication failures, keep the existing in-memory worker display state
`error` and let `_db_update_completion` persist `Task.status="failed"`. Use
`task_set_status` directly only for the nonterminal `validating` state and the explicit
`canceled` terminal state; do not change global worker status names.

- [ ] **Step 8: Add admin-protected routes**

Implement the exact endpoints from the design, including:

```text
POST /api/model-publications
GET /api/model-publications/<task_id>
POST /api/model-publications/<task_id>/cancel
POST /api/model-publications/<task_id>/validate
GET /api/model-publications/<publication_id>/manifest
DELETE /api/model-publications/<publication_id>
```

Reuse `require_admin_token`; do not return host paths, recipe snapshots or full manifest data to non-admin routes.

- [ ] **Step 9: Run focused and full tests**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_publication_tasks.py tests/test_model_publication_routes.py -v
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
git diff --check
```

Expected: all pass; no GPU/model process starts.

- [ ] **Step 10: Commit**

```bash
git add mergeKit_beta/app/model_publication_tasks.py \
  mergeKit_beta/tests/test_model_publication_tasks.py \
  mergeKit_beta/tests/test_model_publication_routes.py \
  mergeKit_beta/merge_manager.py mergeKit_beta/app/services.py \
  mergeKit_beta/app/repositories/__init__.py mergeKit_beta/app/routes.py
git commit -m "feat: add model publication tasks and API"
```

Rollback: stop only active `model_publication` tasks, revert this commit, then let Task 3 reconciliation preserve any already committed asset.

---

### Task 5: Registry Deletion Guards And Gateway Binding

**Files:**
- Create: `mergeKit_beta/tests/model_gateway/test_publication_gateway.py`
- Modify: `mergeKit_beta/app/model_gateway/routes.py`
- Modify: `mergeKit_beta/app/model_gateway/runtime.py`
- Modify: `mergeKit_beta/app/routes.py` legacy delete handlers
- Modify: `mergeKit_beta/app/model_publication.py`
- Modify: `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`
- Modify: `mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`

**Interfaces:**
- Consumes: Task 3 `publication_lock`, `validate_published_asset`, `delete_published_asset`.
- Produces: `GET /api/model-gateway/admin/publishable-models`, service creation by formal `model_id`, soft DELETE service endpoint, shared guarded deletion.

- [ ] **Step 1: Write failing Gateway candidate tests**

Add a published text model and a published blocked VLM to the temporary database/root:

```python
resp = self.client.get("/api/model-gateway/admin/publishable-models", headers=self.admin_headers())
self.assertEqual(resp.status_code, 200)
rows = {row["model_id"]: row for row in resp.get_json()["models"]}
self.assertTrue(rows[text.id]["selectable"])
self.assertFalse(rows[vlm.id]["selectable"])
self.assertEqual(rows[vlm.id]["blocked_reason_code"], "unsupported_architecture")
```

Assert the response is unavailable without the admin token.

- [ ] **Step 2: Write failing service creation and deletion tests**

Assert:

```python
resp = self.client.post("/api/model-gateway/admin/model-services", headers=self.admin_headers(), json={
    "model_id": self.published_text.id,
    "display_name": "Published text",
    "served_model_name": "published-text",
    "gpu_ids": [0],
})
self.assertEqual(resp.status_code, 201)
self.assertEqual(resp.get_json()["service"]["model_path"], self.published_text.path)
```

Also assert arbitrary `model_path` creation is rejected, blocked/stale assets return `409`, a running service cannot be deleted, a stopped service becomes `deleted`, and its `served_model_name` cannot be reused.

- [ ] **Step 3: Write failing cross-database delete-race tests**

Under the same publication lock:

- A formal asset referenced by any non-deleted service returns `409 asset_in_use` through both legacy model delete endpoints and the publication DELETE endpoint.
- After soft-deleting the service, asset deletion moves the directory to `.trash`, deletes the core row, and keeps recipe/usage rows.
- If core deletion raises, the directory is restored from `.trash`.

- [ ] **Step 4: Verify tests fail**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests/model_gateway -p 'test_publication_gateway.py' -v
```

Expected: FAIL because formal candidate and guards do not exist.

- [ ] **Step 5: Implement formal candidates and service creation**

`GET /publishable-models` queries core `Model.source == "published"`, validates manifest metadata, and returns admin-safe summaries. Service creation accepts `model_id`, resolves path/type server-side, and rejects `blocked` or `stale` compatibility before inserting a service row.

Keep old database service rows usable. Do not allow new path-based rows.

- [ ] **Step 6: Recheck formal assets at start**

Add `Config.PUBLISHED_MODELS_PATH` to `allowed_model_roots`. In `start_service`, when `service.model_id` resolves to `source="published"`, call `validate_published_asset` and compare the recorded vLLM version before GPU reservation or process launch. Grandfathered rows retain the existing path validation.

- [ ] **Step 7: Implement service and asset deletion**

Add:

```text
DELETE /api/model-gateway/admin/model-services/<service_id>
```

Only `stopped` and `failed` rows become `deleted`. Do not clear usage, requests, internal historical IDs or names.

Make every model-file delete route call `delete_published_asset` for `source="published"`. Inside the shared `flock`, re-query non-deleted services by both `model_id` and real path before moving the directory.

- [ ] **Step 8: Run focused and full tests**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests/model_gateway -p 'test_publication_gateway.py' -v
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests/model_gateway
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
git diff --check
```

Expected: Gateway and full suites pass; blocked Qwen2.5-VL never launches a process.

- [ ] **Step 9: Commit**

```bash
git add mergeKit_beta/app/model_gateway/routes.py mergeKit_beta/app/model_gateway/runtime.py \
  mergeKit_beta/app/routes.py mergeKit_beta/app/model_publication.py \
  mergeKit_beta/tests/model_gateway/test_publication_gateway.py \
  mergeKit_beta/tests/model_gateway/test_gateway_routes.py \
  mergeKit_beta/tests/model_gateway/test_gateway_runtime.py
git commit -m "feat: bind gateway services to published assets"
```

Rollback: stop any service created by this task, revert the commit, and leave published files intact. Never restore a model directory while a new service references its trash path.

---

### Task 6: Administrator Publication UX

**Files:**
- Modify: `mergeKit_beta/templates/model_repo.html`
- Modify: `mergeKit_beta/templates/model_gateway/console.html`
- Modify: `mergeKit_beta/static/model_gateway/console.js`
- Modify: `mergeKit_beta/static/model_gateway/console.css`
- Modify: `mergeKit_beta/tests/model_gateway/test_gateway_portal.py`
- Create: `mergeKit_beta/tests/test_model_publication_portal.py`

**Interfaces:**
- Consumes: Task 4 publication APIs and Task 5 publishable-model API.
- Produces: model-factory asset publication controls and Gateway formal-asset service selector. User `/research` remains unchanged.

- [ ] **Step 1: Read the required frontend skills and write failing DOM tests**

Before editing UI, load the four frontend skills named in Global Constraints. Add tests asserting:

```python
self.assertIn('id="publication-create-form"', model_repo_page)
self.assertIn('id="publication-status-list"', model_repo_page)
self.assertIn('/api/model-publications', model_repo_page)
self.assertIn('id="gateway-published-model"', console_page)
self.assertNotIn('id="gateway-model-path"', console_page)
self.assertIn('/api/model-gateway/admin/publishable-models', console_script)
self.assertIn('blocked_reason', console_script)
```

Keep existing research portal tests asserting no administrator navigation. Add an assertion that `research.js` only loads `/v1/models` and contains no publication endpoint.

- [ ] **Step 2: Verify tests fail**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_publication_portal.py \
  tests/model_gateway/test_gateway_portal.py -v
```

Expected: FAIL on missing publication controls and formal selector.

- [ ] **Step 3: Update the model repository page**

Add concise sections for:

- base models;
- historical outputs;
- recipes with “物化并发布”;
- formal published assets with type, capabilities, publication state and compatibility.

The publication form sends a generated `Idempotency-Key`, polls the task endpoint, supports queued/running cancellation, and opens explicit GPU validation only when state is `validating`. It must never auto-select GPU 2 or any GPU.

- [ ] **Step 4: Update the Gateway administrator console**

Replace free-form path/datalist with `gateway-published-model`. Load formal candidates with the admin token. Render blocked/stale assets in a separate explanatory list or disabled options; submit only a selectable model ID. Model type/path are read-only summaries.

Add a delete-service button only for `stopped` and `failed`. Keep start/stop behavior and all API Key/playground controls.

- [ ] **Step 5: Preserve interaction and accessibility quality**

- Use existing CSS variables, square/low-radius operational panels and Remix icons.
- Add hover/focus-visible states to new controls.
- Keep status changes in `aria-live` regions.
- Do not add background animation.
- If GSAP is used for a status transition, animate only `transform` and `opacity` and disable movement under `prefers-reduced-motion`.
- At 1440, 1024, 720 and the actual narrow Firefox viewport, controls must not overlap and `scrollWidth <= innerWidth`.

- [ ] **Step 6: Run frontend and regression tests**

```bash
node --check mergeKit_beta/static/model_gateway/console.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest tests/test_model_publication_portal.py \
  tests/model_gateway/test_gateway_portal.py -v
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
git diff --check
```

Expected: syntax and tests pass; no GPU process starts.

- [ ] **Step 7: Run Firefox WebDriver acceptance**

Use installed `/snap/bin/geckodriver`, an explicit process cleanup trap, and screenshots under ignored `logs/model_gateway/acceptance/<timestamp>/`. Verify model publication states, blocked VLM explanation, service selector, keyboard focus, reduced motion, no overlap and no admin controls in `/research`.

Afterward run:

```bash
pgrep -af 'geckodriver|firefox.*headless' || true
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
```

Expected: no browser test process remains and no new GPU process exists.

- [ ] **Step 8: Commit**

```bash
git add mergeKit_beta/templates/model_repo.html \
  mergeKit_beta/templates/model_gateway/console.html \
  mergeKit_beta/static/model_gateway/console.js \
  mergeKit_beta/static/model_gateway/console.css \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/tests/test_model_publication_portal.py
git commit -m "feat: expose model publication states to administrators"
```

Rollback: revert only this UI commit; backend publication and existing Gateway APIs remain available.

---

### Task 7: Real Text And VLM Acceptance

**Files:**
- Create: `mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260716_MODEL_PUBLICATION.md`
- Update: `mergeKit_beta/docs/model_gateway/ARCHITECTURE.md`
- Update: `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`
- Update: `mergeKit_beta/docs/model_gateway/README.md`
- Runtime evidence only: `mergeKit_beta/logs/model_gateway/acceptance/<timestamp>/`

**Interfaces:**
- Consumes: all previous tasks and real model factory/Gateway HTTP paths.
- Produces: fresh acceptance evidence, operator documentation, exact retained assets and rollback commands.

- [ ] **Step 1: Run the complete non-GPU gate after a clean recreate**

```bash
docker compose up -d --force-recreate mergekit-beta
docker compose ps
docker compose exec -T mergekit-beta pwd
for endpoint in /healthz /readyz /api/models /api/testset/list /api/history /model_repo /model-gateway /research; do
  curl -fsS -o /dev/null -w "$endpoint %{http_code}\n" "http://127.0.0.1:5000$endpoint"
done
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -c "import merge_manager; from app import app; import evolution.runner; import app.model_publication"
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python \
  -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/console.js
git diff --check
```

Expected: all endpoints return `200`, exact container path matches, imports and tests pass.

- [ ] **Step 2: Capture a fresh resource and model-pool preflight**

Record GPU UUID, bus ID, memory, compute processes and containers. Inspect every candidate parent's `config.json`, processor files and weight signature through the new inspection API. Do not assume the two text parents are compatible.

Stop before GPU use if:

- GPU 2 changes or cannot be distinguished;
- an external process occupies a candidate GPU;
- fewer than two distinct compatible language parents exist;
- the full VLM base fingerprint differs from the recipe/source record;
- free publication disk is below the formula in the design.

- [ ] **Step 3: Execute a real text publication and Gateway call**

Using the administrator UI or the actual HTTP endpoints, not direct function calls:

1. Publish an existing compatible Qwen text model.
2. Provide an explicit non-GPU-2 validation GPU only after the snapshot.
3. Verify the asset is `published + ready`.
4. Create a service by `model_id`.
5. Start the service with the minimum safe GPU set.
6. Create a temporary API Key and call `/v1/models` and `/v1/chat/completions`.
7. Verify non-zero prompt/completion token usage in the usage API.
8. Stop and soft-delete the service.
9. Revoke the temporary API Key.
10. Keep the formal asset until the user explicitly chooses whether to retain it.

Save redacted request/response JSON and service log paths. Never record plaintext API keys.

- [ ] **Step 4: Execute a real standard VLM search**

Use the model factory `/api/merge_evolutionary` path with two verified distinct compatible language parents, the verified full VLM base, VLM mode, a very small real CMMMU validation sample, minimal population/iterations accepted by the existing algorithm, and explicit safe GPUs.

Do not directly invoke the vendor script to bypass task registration. Wait for the real task terminal state. Record progress, recipe, genotype, CMMMU result, duration, GPU snapshots and cleanup behavior.

- [ ] **Step 5: Publish the resulting VLM recipe**

Use `POST /api/model-publications` with a unique idempotency key, wait for `validating`, then explicitly validate on approved non-GPU-2 devices. Verify:

- full independent model directory under `/data/PublishedModels`;
- `visual.*` and language weights both present;
- tokenizer and processor load;
- real image inference succeeds;
- a real CMMMU sample is evaluated from the published path;
- manifest recipe snapshot contains all old fields plus `vlm_path` and `vlm_base`;
- compatibility is `blocked` with `unsupported_architecture` for vLLM 0.7.0;
- Gateway admin sees the reason;
- `/v1/models` and `/research` do not expose the model;
- no vLLM process was started for the blocked VLM.

- [ ] **Step 6: Exercise recovery and deletion guards through real services**

With only a disposable publication task/service:

- stop the app after final rename but before simulated registration completion using the controlled Task 3 fault hook, restart, and verify one recovered core row;
- submit the same `Idempotency-Key` twice and verify one publication ID;
- attempt deletion while a stopped but non-deleted service references the asset and expect `409 asset_in_use`;
- soft-delete the service, then verify deletion becomes eligible without deleting the accepted VLM unless explicitly approved.

Fault hooks must be disabled and absent from production defaults after the test.

- [ ] **Step 7: Verify resource cleanup**

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
pgrep -af 'vllm|run_vlm_search|ray::|geckodriver|firefox.*headless' || true
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
find /home/a/Model_factory_data/published_models/.staging -mindepth 1 -maxdepth 1 -print 2>/dev/null
```

Expected: GPU 2/external workloads unchanged; no test vLLM/Ray/browser process; no failed large staging directory; accepted formal assets and recipes remain.

- [ ] **Step 8: Update operations documentation**

Document:

- asset publication versus service creation terminology;
- mount and environment variables;
- recipe/manifest fields;
- manual service restart rule;
- `ready`, `blocked`, `stale` meanings;
- Qwen2.5-VL vLLM 0.7.0 limitation;
- exact deletion and recovery commands;
- retained model paths and disk use.

- [ ] **Step 9: Write the acceptance record**

`ACCEPTANCE_20260716_MODEL_PUBLICATION.md` must contain exact commands, HTTP status codes, test counts, task/publication/service IDs, model fingerprints, redacted outputs, GPU before/after tables, retained files, known limitations and rollback commit IDs. Mark the VLM service portion as blocked, not complete, until a separately approved runtime-upgrade plan passes.

- [ ] **Step 10: Run final review and verification skills**

Invoke `requesting-code-review`, resolve correctness findings using `systematic-debugging`, then invoke `verification-before-completion` and rerun the complete fresh command set. Do not rely on earlier outputs.

- [ ] **Step 11: Commit documentation and acceptance**

```bash
git add mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260716_MODEL_PUBLICATION.md \
  mergeKit_beta/docs/model_gateway/ARCHITECTURE.md \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md \
  mergeKit_beta/docs/model_gateway/README.md
git commit -m "test: accept model publication pipeline"
```

Rollback: stop test services, revoke temporary keys, revert implementation commits newest-first, recreate `mergekit-beta`, and leave `/home/a/Model_factory_data/published_models` untouched for manual disposition.

## Plan Self-Review

- Tasks 1-2 cover structural VLM detection, TextOnly false positives, additive recipe provenance and reuse of strict language-weight composition in existing CMMMU paths.
- Tasks 3-4 cover the universal publication directory, manifest, hashes, atomic rename, task states, idempotency, cancellation and explicit-GPU validation resume.
- Task 5 covers cross-database race protection, all model delete entry points, formal Gateway service creation, manual restart and permanent service-name history.
- Task 6 covers administrator UX while preserving the user portal's running-model-only contract.
- Task 7 uses real model factory, worker, HTTP, Transformers, CMMMU and vLLM paths; controlled unit doubles are limited to failure-phase tests.
- No task upgrades dependencies, auto-selects GPU, touches GPU 2, or declares Qwen2.5-VL user serving complete under vLLM 0.7.0.
- The plan introduces no new database table or column. Core `Task` and `Model` rows plus existing Gateway service rows remain the authorities.
