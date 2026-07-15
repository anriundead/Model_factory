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


def call_research_model(service, messages: list[dict], *, post=requests.post) -> str:
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
            "max_tokens": 1024,
            "stream": False,
        },
        timeout=(30, 300),
    )
    if response.status_code >= 400:
        raise ValueError("model_upstream_error")
    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("model_invalid_response") from exc
    if not content:
        raise ValueError("model_empty_response")
    return content
