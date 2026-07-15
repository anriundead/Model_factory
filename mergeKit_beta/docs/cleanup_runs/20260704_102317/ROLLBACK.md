# ROLLBACK

## Gate 3 Tracked File Deletions

Restore deleted tracked files with:

```bash
git restore mergeKit_beta/investigate_4479.py
git restore mergeKit_beta/verify_subset.py
git restore mergeKit_beta/verify_test.py
git restore mergeKit_beta/scripts/patch_lm_eval_transformers5.py
git restore mergeKit_beta/eval_worker.py
```

After rollback, rerun:

```bash
docker compose ps
curl -fsS http://127.0.0.1:5000/healthz
curl -fsS http://127.0.0.1:5000/readyz
curl -fsS http://127.0.0.1:5000/api/models
curl -fsS http://127.0.0.1:5000/api/testset/list
curl -fsS http://127.0.0.1:5000/api/history
docker compose exec -T mergekit-beta bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate mergenetic && python -c "import merge_manager; from app import app; import evolution.runner"'
docker compose exec -T mergekit-beta bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate mergenetic && python -m unittest discover -s tests'
```
