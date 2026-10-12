# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Normalize Anthropic content without losing images or tool results."""

from __future__ import annotations

import base64
import io
from contextlib import closing
from typing import Any, Dict, List, Optional, Set, Union


def _image(block: Dict[str, Any]) -> Dict[str, Any]:
    source = block.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError("Anthropic image source must be an object")
    if source.get("type") == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if media_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
            raise ValueError("Unsupported Anthropic image media_type")
        if not isinstance(data, str):
            raise ValueError("Anthropic images require base64 data")
        from ...model.llm.media import decode_data_uri

        url = f"data:{media_type};base64,{data}"
        decode_data_uri(url)
    elif source.get("type") == "url":
        candidate = source.get("url")
        if not isinstance(candidate, str) or not candidate.startswith(
            ("https://", "http://")
        ):
            raise ValueError("Anthropic image URLs must use http or https")
        url = candidate
    else:
        raise ValueError("Anthropic images require a base64 or URL source")
    return {"type": "image_url", "image_url": {"url": url}}


def _document(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    source = block.get("source") or {}
    if not isinstance(source, dict):
        raise ValueError("Anthropic document source must be an object")
    kind = source.get("type")
    if kind == "text" and source.get("media_type") == "text/plain":
        data = source.get("data")
        if not isinstance(data, str):
            raise ValueError("Text documents require string data")
        return [{"type": "text", "text": data}]
    if kind == "content":
        content = source.get("content")
        result = normalize_content(content)
        return [{"type": "text", "text": result}] if isinstance(result, str) else result
    if kind not in {"base64", "url"}:
        raise ValueError("Unsupported Anthropic document source")
    if kind == "base64":
        if source.get("media_type") != "application/pdf":
            raise ValueError("Base64 documents must use application/pdf")
        data = source.get("data")
        if not isinstance(data, str):
            raise ValueError("PDF documents require base64 data")
        url = f"data:application/pdf;base64,{data}"
    else:
        candidate = source.get("url")
        if not isinstance(candidate, str) or not candidate.startswith(
            ("https://", "http://")
        ):
            raise ValueError("Document URLs must use http or https")
        url = candidate
    from ...model.llm.media import load_media_bytes

    raw = load_media_bytes(url)
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError(
            "PDF input requires pypdfium2; install xinference[image]"
        ) from exc
    parts: List[Dict[str, Any]] = []
    with closing(pdfium.PdfDocument(raw)) as pdf:
        if len(pdf) > 20:
            raise ValueError("PDF input supports at most 20 pages per document")
        for page in pdf:
            with closing(page):
                width, height = page.get_size()
                scale = min(1.5, 2048 / max(width, height))
                with closing(page.render(scale=scale)) as bitmap:
                    output = io.BytesIO()
                    bitmap.to_pil().save(output, format="PNG")
                    encoded = base64.b64encode(output.getvalue()).decode("ascii")
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        }
                    )
    return parts


def normalize_content(
    content: Any, tool_names: Optional[Set[str]] = None
) -> Union[str, List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be a string or an array of blocks")
    parts: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Anthropic content blocks must be JSON objects")
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Anthropic text blocks require a string 'text' field")
            if not text.startswith("x-anthropic-billing-header"):
                parts.append({"type": "text", "text": text})
        elif kind == "image":
            parts.append(_image(block))
        elif kind == "document":
            if block.get("citations", {}).get("enabled"):
                raise ValueError("Document citations are not supported")
            prefix = "\n".join(
                str(block[key]) for key in ("title", "context") if block.get(key)
            )
            if prefix:
                parts.append({"type": "text", "text": prefix})
            parts.extend(_document(block))
        elif kind == "tool_reference":
            name = block.get("tool_name")
            if (
                not isinstance(name, str)
                or tool_names is None
                or name not in tool_names
            ):
                raise ValueError(f"Unknown tool reference: {name}")
            parts.append({"type": "text", "text": f"Tool available: {name}"})
        else:
            raise ValueError(f"Unsupported Anthropic content block type: {kind}")
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts
