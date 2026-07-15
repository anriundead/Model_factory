"""Compatibility wrapper for launching vLLM from the serving runtime."""
from __future__ import annotations

import hmac
import os
import sys

from fastapi import HTTPException, Request


def _patch_transformers_tokenizers() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


def _install_internal_abort_endpoint() -> None:
    """Install a loopback-only abort endpoint before vLLM builds its app."""
    from vllm.entrypoints.openai import api_server

    if getattr(api_server, "_model_gateway_abort_installed", False):
        return
    original_build_app = api_server.build_app

    def build_app(*args, **kwargs):
        app = original_build_app(*args, **kwargs)

        @app.post("/internal/model-gateway/abort/{request_id}", include_in_schema=False)
        async def abort_request(request_id: str, request: Request):
            expected = os.environ.get("MERGEKIT_MODEL_GATEWAY_INTERNAL_API_KEY", "")
            supplied = request.headers.get("Authorization", "")
            if not expected or not hmac.compare_digest(supplied, f"Bearer {expected}"):
                raise HTTPException(status_code=401, detail="Unauthorized")
            await app.state.engine_client.abort(request_id)
            return {"status": "accepted", "request_id": request_id}

        return app

    api_server.build_app = build_app
    api_server._model_gateway_abort_installed = True


def main() -> int:
    _patch_transformers_tokenizers()
    _install_internal_abort_endpoint()
    from vllm.scripts import main as vllm_main

    return int(vllm_main() or 0)


if __name__ == "__main__":
    sys.exit(main())
