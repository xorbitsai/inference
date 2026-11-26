# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Normalize Anthropic content without losing images or tool results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from .pdf_content import PDFRenderBudget, decode_pdf_base64, render_pdf_pages


@dataclass
class ContentBudget:
    """Share PDF resource accounting and decoded blocks across one request."""

    pdf: PDFRenderBudget = field(default_factory=PDFRenderBudget)
    documents: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    direct_images: int = 0
    tool_result_images: int = 0
    counted_direct_images: Set[int] = field(default_factory=set)
    counted_tool_result_images: Set[int] = field(default_factory=set)

    def metrics(self) -> Dict[str, int]:
        return {
            "direct_images": self.direct_images,
            "tool_result_images": self.tool_result_images,
            "document_pages": self.pdf.pages,
            "document_count": len(self.documents),
            "total_images": (
                self.direct_images + self.tool_result_images + self.pdf.pages
            ),
        }


def _image(
    block: Dict[str, Any], budget: ContentBudget, image_origin: str
) -> Dict[str, Any]:
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
    block_id = id(block)
    if image_origin == "tool_result":
        if block_id not in budget.counted_tool_result_images:
            budget.counted_tool_result_images.add(block_id)
            budget.tool_result_images += 1
    elif block_id not in budget.counted_direct_images:
        budget.counted_direct_images.add(block_id)
        budget.direct_images += 1
    return {"type": "image_url", "image_url": {"url": url}}


def _document(block: Dict[str, Any], budget: ContentBudget) -> List[Dict[str, Any]]:
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
        result = normalize_content(content, budget=budget)
        return [{"type": "text", "text": result}] if isinstance(result, str) else result
    if kind not in {"base64", "url"}:
        raise ValueError("Unsupported Anthropic document source")
    if kind == "base64":
        if source.get("media_type") != "application/pdf":
            raise ValueError("Base64 documents must use application/pdf")
        data = source.get("data")
        if not isinstance(data, str):
            raise ValueError("PDF documents require base64 data")
        return render_pdf_pages(decode_pdf_base64(data, budget.pdf), budget.pdf)

    url = source.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("Document URLs must use http or https")
    from ...model.llm.media import load_media_bytes

    return render_pdf_pages(load_media_bytes(url), budget.pdf)


def normalize_content(
    content: Any,
    tool_names: Optional[Set[str]] = None,
    budget: Optional[ContentBudget] = None,
    image_origin: str = "direct",
) -> Union[str, List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be a string or an array of blocks")
    if budget is None:
        budget = ContentBudget()
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
            parts.append(_image(block, budget, image_origin))
        elif kind == "document":
            citations = block.get("citations")
            if citations is not None and not isinstance(citations, dict):
                raise ValueError("Document citations must be an object or null")
            if citations and citations.get("enabled"):
                raise ValueError("Document citations are not supported")
            prefix = "\n".join(
                str(block[key]) for key in ("title", "context") if block.get(key)
            )
            if prefix:
                parts.append({"type": "text", "text": prefix})
            key = id(block)
            if key not in budget.documents:
                budget.documents[key] = _document(block, budget)
            parts.extend(budget.documents[key])
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
