# Validation Log - 20260703_171018

## Gate 0: Preflight

Status: PASS

Commands executed:

```bash
docker compose config --quiet
docker compose config --services
docker image inspect mergekit-beta:latest --format '{{.Id}} {{.Created}}'
ss -ltnp 'sport = :5000'
nvidia-smi -L
docker compose config --format json
git status --short
git diff --stat
find <resolved-volume-sources> -maxdepth 0 -type d
```

Summary:

- Compose config parses.
- Service `mergekit-beta` exists.
- Image `mergekit-beta:latest` exists.
- Port 5000 was not occupied.
- GPU UUIDs in compose match `nvidia-smi -L`.
- Resolved volume sources exist.

## Gate 1: Baseline Runtime

Status: FAIL

Command:

```bash
docker compose up -d mergekit-beta
```

Failure:

```text
Error response from daemon: unknown or invalid runtime name: nvidia
```

Follow-up diagnostics:

```bash
docker info --format '{{json .Runtimes}}'
docker compose ps
docker compose logs --tail=120 mergekit-beta
docker ps -a --filter name=model_factory-mergekit-beta-1 --format '{{.Names}} {{.Status}} {{.Image}}'
```

Diagnosis:

- Docker daemon runtimes do not include `nvidia`.
- Existing `model_factory-mergekit-beta-1` container remains exited with status 128.
- Logs shown by compose are stale logs from the previous successful run on 2026-06-10; they are not evidence of a current successful startup.

## Cleanup Status

No cleanup was executed.

Not executed:

- Runtime artifact deletion.
- Tracked deprecated code deletion.
- `.gitignore` changes.
- `git rm --cached`.
- Runtime smoke tests after cleanup.

## Stop Reason

Gate 1 failed before cleanup. Per plan, subsequent cleanup gates are frozen until the Docker runtime issue is resolved.
