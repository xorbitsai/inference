# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Normalize inline OpenAI Chat Completions PDF attachments."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .pdf_content import (
    PDF_DATA_URL_PREFIX,
    PDFRenderBudget,
    decode_pdf_base64,
    render_pdf_pages,
)


def _is_file_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "file"


async def expand_chat_pdfs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    attachments = {
        index: content
        for index, message in enumerate(messages)
        if isinstance(content := message.get("content"), list)
        and any(_is_file_part(part) for part in content)
    }
    if not attachments:
        return messages
    return await asyncio.to_thread(_expand_pdfs, messages, attachments)


def _expand_pdfs(
    messages: List[Dict[str, Any]], attachments: Dict[int, List[Any]]
) -> List[Dict[str, Any]]:
    budget = PDFRenderBudget()
    expanded = list(messages)
    for index, content in attachments.items():
        message = messages[index]
        if message.get("role") != "user":
            raise ValueError("PDF file parts are supported only in user messages.")
        parts: List[Any] = []
        for part in content:
            if not _is_file_part(part):
                parts.append(part)
                continue
            file = part.get("file")
            if not isinstance(file, dict) or file.get("file_id") is not None:
                raise ValueError("File parts require an inline PDF in file.file_data.")
            data = file.get("file_data")
            if not isinstance(data, str) or not data:
                raise ValueError("File parts require non-empty Base64 PDF data.")
            if data.startswith("data:"):
                if not data.startswith(PDF_DATA_URL_PREFIX):
                    raise ValueError(
                        "PDF data URLs must use application/pdf and Base64."
                    )
                data = data[len(PDF_DATA_URL_PREFIX) :]
            parts.extend(render_pdf_pages(decode_pdf_base64(data, budget), budget))
        expanded[index] = {**message, "content": parts}
    return expanded
