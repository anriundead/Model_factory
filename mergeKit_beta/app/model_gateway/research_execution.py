"""Citation contract shared by the future research-job executor."""
from __future__ import annotations

import json
import re
import uuid

import requests


_TASK_INSTRUCTIONS = {
    "document_qa": "Answer the question from the supplied research evidence.",
    "summary": "Write a concise evidence-grounded summary.",
    "compare": "Compare the supplied sources using only their evidence.",
    "extract": "Extract the requested structured facts from the supplied evidence.",
}


def budget_research_evidence(evidence: list[dict], max_context_chars: int) -> list[dict]:
    """Fit temporary evidence into a conservative context budget without losing a source."""
    if max_context_chars < 1:
        raise ValueError("research_context_budget_must_be_positive")
    first_per_file = []
    remaining = []
    seen_files = set()
    for item in evidence:
        file_id = item.get("file_id")
        if file_id not in seen_files:
            first_per_file.append(item)
            seen_files.add(file_id)
        else:
            remaining.append(item)
    ordered = first_per_file + remaining
    selected = []
    remaining_chars = int(max_context_chars)
    for index, item in enumerate(ordered):
        if remaining_chars < 1:
            break
        text = (item.get("text") or "").strip()
        if not text:
            continue
        slots = max(1, len(ordered) - index)
        text_limit = max(1, remaining_chars // slots)
        copy = dict(item)
        copy["text"] = text[:text_limit]
        selected.append(copy)
        remaining_chars -= len(copy["text"])
    return selected


def _locator_label(locator: dict) -> str:
    return f"{locator.get('kind', 'source')} {locator.get('value', '?')}"


def build_research_messages(task_type: str, user_input: str, evidence: list[dict], *, output_format: str = "markdown") -> list[dict]:
    """Build an internal prompt whose evidence IDs are stable and validated later."""
    instruction = _TASK_INSTRUCTIONS.get(task_type)
    if not instruction:
        raise ValueError("unsupported_research_task")
    if not evidence:
        raise ValueError("no_research_evidence")
    if output_format not in {"markdown", "json"}:
        raise ValueError("unsupported_research_output_format")
    sources = "\n\n".join(
        f"[S{index}] {item['file_id']} {_locator_label(item.get('locator') or {})}\n{item['text']}"
        for index, item in enumerate(evidence, start=1)
    )
    format_instruction = (
        'Return a valid JSON object with "answer" and "citations". "answer" must include [S<number>] citations; '
        '"citations" must list exactly those citation numbers.'
        if output_format == "json"
        else "Return concise Markdown with [S<number>] citations beside factual claims."
    )
    return [
        {
            "role": "system",
            "content": f"Use only the supplied evidence for factual claims. Cite each claim as [S<number>]. {format_instruction}",
        },
        {"role": "user", "content": f"{instruction}\n\nRequest:\n{user_input}\n\nEvidence:\n{sources}"},
    ]


def validate_research_answer(answer: str, evidence: list[dict], require_citations: bool, *, output_format: str = "markdown") -> list[int]:
    """Reject missing or fabricated evidence IDs before a result becomes visible."""
    answer_text = answer or ""
    declared_citations = None
    if output_format == "json":
        try:
            payload = json.loads(answer_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_json_output") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str) or not isinstance(payload.get("citations"), list):
            raise ValueError("invalid_json_output")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in payload["citations"]):
            raise ValueError("invalid_json_output")
        answer_text = payload["answer"]
        declared_citations = sorted(set(payload["citations"]))
    elif output_format != "markdown":
        raise ValueError("unsupported_research_output_format")
    citations = sorted({int(value) for value in re.findall(r"\[S(\d+)\]", answer_text)})
    if require_citations and not citations:
        raise ValueError("citation_required")
    if any(index < 1 or index > len(evidence) for index in citations):
        raise ValueError("invalid_citation")
    if declared_citations is not None and declared_citations != citations:
        raise ValueError("citation_mismatch")
    return citations


def call_research_model(service, messages: list[dict], *, max_tokens: int = 1024, post=requests.post) -> tuple[str, dict]:
    """Invoke a manually started vLLM service through its private loopback API."""
    if getattr(service, "vllm_host", "127.0.0.1") != "127.0.0.1":
        raise ValueError("model_not_loopback")
    if not getattr(service, "vllm_port", None) or not getattr(service, "internal_api_key", None):
        raise ValueError("model_runtime_not_ready")
    response = post(
        f"http://127.0.0.1:{service.vllm_port}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {service.internal_api_key}",
            "X-Request-Id": str(uuid.uuid4()),
        },
        json={
            "model": service.served_model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max(1, int(max_tokens)),
            "stream": False,
        },
        timeout=(30, 300),
    )
    if response.status_code >= 400:
        raise ValueError("model_upstream_error")
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("model_invalid_response") from exc
    if not content:
        raise ValueError("model_empty_response")
    usage = body.get("usage") or {}
    return content, {
        "prompt_tokens": max(0, int(usage.get("prompt_tokens") or 0)),
        "completion_tokens": max(0, int(usage.get("completion_tokens") or 0)),
        "total_tokens": max(0, int(usage.get("total_tokens") or 0)),
    }
