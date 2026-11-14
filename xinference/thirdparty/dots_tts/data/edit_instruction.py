"""Shared edit-instruction construction and target-text rendering helpers."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

_INS_TAG_RE = re.compile(r"<\s*(/?)\s*ins\s*>", re.IGNORECASE)
_BOUNDARY_CLOSING_INS_RE = re.compile(r"^<\s*/\s*ins\s*>", re.IGNORECASE)
_TRANSFER_TAG_RE = re.compile(r"<\s*spk_transfer\s*/\s*>", re.IGNORECASE)
_ANY_TEXT_EDIT_TAG_RE = re.compile(r"</?(?:del|ins|sub)(?:\s+[^>]*)?>")
_QUERY_TAG_RE = re.compile(r"</?q>")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?，。？！；：])")
_APOSTROPHE_SUFFIX_RE = re.compile(
    r"(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE
)
_TAG_TOKEN_RE = re.compile(r"<[^>]*>")
_OPEN_TAG_RE = re.compile(r"^<\s*([A-Za-z_][\w-]*)(?:\s|,|>)", re.DOTALL)
_CLOSE_TAG_RE = re.compile(r"^<\s*/\s*([A-Za-z_][\w-]*)\s*>$", re.DOTALL)
_SELF_CLOSING_TAG_RE = re.compile(
    r"^<\s*([A-Za-z_][\w-]*)(?:(?:\s|,)[^>]*)?/\s*>$", re.DOTALL
)
_TARGET_ATTR_RE = re.compile(
    r"\btarg\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL
)
_KNOWN_CONTAINER_TAGS = frozenset(
    {"del", "ins", "sub", "emo", "pitch", "rate", "enhance", "bg"}
)
_KNOWN_EMPTY_TAGS = frozenset({"pause", "spk_transfer"})
_AUTO_DISABLE_XVECTOR_TAGS = frozenset({"emo", "bg", "enhance"})

EditXVectorMode = bool | Literal["auto"]


class TaggedTextContractError(ValueError):
    """Raised when literal text would be ambiguous with structural edit tags."""


@dataclass(slots=True)
class _InstructionNode:
    tag: str
    raw_opening_tag: str = ""
    children: list["_InstructionNode | str"] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _RemovedInsertion:
    """Boundary marker retained while rendering a source transcript."""


def escape_text(text: str) -> str:
    """Escape text and attribute values for the structured edit instruction."""

    return html.escape(text, quote=True)


def unescape_text(text: str) -> str:
    """Decode an escaped instruction fragment for target-text rendering."""

    return html.unescape(text)


def validate_literal_text(text: str, *, label: str) -> None:
    """Reject literal text that would be parsed as a structural text-edit tag."""

    if _ANY_TEXT_EDIT_TAG_RE.search(text):
        raise TaggedTextContractError(f"{label} contains unsupported markup.")


def _contains_ascii_word_char(text: str) -> bool:
    return bool(_ASCII_WORD_RE.search(text))


def _visible_text(text: str) -> str:
    return _QUERY_TAG_RE.sub("", text)


def _visible_edge(text: str, *, left: bool) -> str:
    stripped = _visible_text(text)
    if not stripped:
        return ""
    return stripped[0] if left else stripped[-1]


def _is_ascii_word_char(char: str) -> bool:
    return bool(char and _ASCII_WORD_RE.fullmatch(char))


def _is_word_internal_join(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if right[0] in {"'", "-", "’"} and len(right) > 1:
        return _is_ascii_word_char(left[-1]) and _is_ascii_word_char(right[1])
    if left[-1] in {"'", "-", "’"} and len(left) > 1:
        if not (_is_ascii_word_char(left[-2]) and _is_ascii_word_char(right[0])):
            return False
        if left[-1] in {"'", "’"}:
            return bool(_APOSTROPHE_SUFFIX_RE.match(right))
        return True
    return False


def _attach_connector_to_left(part_visible: str, left: str, right: str) -> bool:
    if not left or not right or not _is_ascii_word_char(left[-1]):
        return False
    if part_visible == "-":
        return _is_ascii_word_char(right[0])
    if part_visible in {"'", "’"}:
        return bool(_APOSTROPHE_SUFFIX_RE.match(right)) or left[-1] in {"s", "S"}
    return False


def normalize_target_parts(parts: Iterable[str]) -> str:
    """Render target-side parts without trusting tag-boundary whitespace."""

    clean_parts = [raw_part.strip() for raw_part in parts]
    rendered: list[str] = []
    for index, part in enumerate(clean_parts):
        if not part:
            continue
        left_visible = _visible_text(rendered[-1]) if rendered else ""
        right_visible = _visible_text(part)
        next_visible = ""
        for next_part in clean_parts[index + 1 :]:
            next_visible = _visible_text(next_part)
            if next_visible:
                break
        if rendered and _attach_connector_to_left(
            right_visible, left_visible, next_visible
        ):
            rendered[-1] += part
            continue
        left_edge = _visible_edge(left_visible, left=False)
        right_edge = _visible_edge(right_visible, left=True)
        needs_space = (
            bool(rendered)
            and not _is_word_internal_join(left_visible, right_visible)
            and (
                _contains_ascii_word_char(left_edge)
                or _contains_ascii_word_char(right_edge)
            )
        )
        if needs_space:
            rendered.append(" ")
        rendered.append(part)
    text = "".join(rendered)
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def _parse_instruction(instruction: str) -> _InstructionNode:
    if not isinstance(instruction, str) or not instruction:
        raise TaggedTextContractError("Edit instruction must be a non-empty string.")

    root = _InstructionNode(tag="__root__")
    stack = [root]
    cursor = 0
    for match in _TAG_TOKEN_RE.finditer(instruction):
        if match.start() > cursor:
            stack[-1].children.append(instruction[cursor : match.start()])
        token = match.group(0)
        close_match = _CLOSE_TAG_RE.fullmatch(token)
        if close_match is not None:
            tag = close_match.group(1).lower()
            if len(stack) == 1 or stack[-1].tag != tag:
                expected = stack[-1].tag if len(stack) > 1 else "no closing tag"
                raise TaggedTextContractError(
                    f"Unexpected </{tag}> tag; expected {expected}."
                )
            stack.pop()
            cursor = match.end()
            continue

        self_closing_match = _SELF_CLOSING_TAG_RE.fullmatch(token)
        if self_closing_match is not None:
            tag = self_closing_match.group(1).lower()
            if tag not in _KNOWN_EMPTY_TAGS:
                raise TaggedTextContractError(
                    f"Unsupported self-closing edit tag: <{tag}/> ."
                )
            stack[-1].children.append(_InstructionNode(tag=tag, raw_opening_tag=token))
            cursor = match.end()
            continue

        open_match = _OPEN_TAG_RE.match(token)
        if open_match is None:
            raise TaggedTextContractError(f"Malformed edit tag: {token!r}.")
        tag = open_match.group(1).lower()
        if tag not in _KNOWN_CONTAINER_TAGS:
            raise TaggedTextContractError(f"Unsupported edit tag: <{tag}>.")
        node = _InstructionNode(tag=tag, raw_opening_tag=token)
        stack[-1].children.append(node)
        stack.append(node)
        cursor = match.end()

    if cursor < len(instruction):
        stack[-1].children.append(instruction[cursor:])
    if len(stack) != 1:
        raise TaggedTextContractError(f"Unclosed <{stack[-1].tag}> edit tag.")
    if "<" in _TAG_TOKEN_RE.sub("", instruction) or ">" in _TAG_TOKEN_RE.sub("", instruction):
        raise TaggedTextContractError("Malformed angle bracket in edit instruction.")
    return root


def instruction_operation_tags(instruction: str) -> frozenset[str]:
    """Return the structural operation tags present in an edit instruction."""

    root = _parse_instruction(instruction)
    tags: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            if isinstance(child, _InstructionNode):
                tags.add(child.tag)
                stack.append(child)
    return frozenset(tags)


def normalize_edit_xvector_mode(value: object) -> EditXVectorMode:
    """Validate the public bool-or-auto Edit X-vector request contract."""

    if isinstance(value, bool):
        return value
    if value == "auto":
        return "auto"
    raise ValueError('use_xvector must be a boolean or "auto"')


def resolve_edit_use_xvector(
    mode: EditXVectorMode,
    instruction: str,
) -> bool:
    """Resolve an Edit X-vector mode from parsed instruction operations."""

    normalized = normalize_edit_xvector_mode(mode)
    if isinstance(normalized, bool):
        return normalized
    operation_tags = instruction_operation_tags(instruction)
    return not (
        operation_tags
        and operation_tags.issubset(_AUTO_DISABLE_XVECTOR_TAGS)
    )


def _sub_target(node: _InstructionNode) -> str:
    match = _TARGET_ATTR_RE.search(node.raw_opening_tag)
    if match is None:
        raise TaggedTextContractError("<sub> requires a quoted targ attribute.")
    return unescape_text(match.group(2))


def _render_target_node(node: _InstructionNode) -> tuple[list[tuple[str, bool]], bool]:
    child_segments: list[tuple[str, bool]] = []
    has_text_edit = False
    for child in node.children:
        if isinstance(child, str):
            child_segments.append((unescape_text(child), False))
            continue
        rendered, child_has_text_edit = _render_target_node(child)
        child_segments.extend(rendered)
        has_text_edit = has_text_edit or child_has_text_edit

    if node.tag == "del":
        # Retain an empty edit boundary.  A deletion can be the only separator
        # between two otherwise unchanged English words; dropping the boundary
        # here would merge the neighbouring plain-text segments before
        # ``normalize_target_parts`` has a chance to restore their word break.
        return [("", True)], True
    if node.tag == "sub":
        return [(_sub_target(node), True)], True
    if node.tag == "ins":
        inserted = "".join(text for text, _ in child_segments)
        return [(inserted, True)], True
    if node.tag in _KNOWN_EMPTY_TAGS:
        return [], has_text_edit
    return child_segments, has_text_edit


def render_target_text(instruction: str) -> str:
    """Render the plain target transcript from a structured edit instruction."""

    segments, has_text_edit = _render_target_node(_parse_instruction(instruction))
    if not has_text_edit:
        return "".join(text for text, _ in segments)

    parts: list[str] = []
    unchanged: list[str] = []
    for text, is_edit in segments:
        if is_edit:
            if unchanged:
                parts.append("".join(unchanged))
                unchanged.clear()
            parts.append(text)
        else:
            unchanged.append(text)
    if unchanged:
        parts.append("".join(unchanged))
    return normalize_target_parts(parts)


def _render_source_node(node: _InstructionNode) -> list[str | _RemovedInsertion]:
    rendered: list[str | _RemovedInsertion] = []
    for child in node.children:
        if isinstance(child, str):
            rendered.append(unescape_text(child))
        else:
            rendered.extend(_render_source_node(child))

    if node.tag == "ins":
        return [_RemovedInsertion()]
    if node.tag == "sub":
        _sub_target(node)
    if node.tag in _KNOWN_EMPTY_TAGS:
        return []
    return rendered


def _source_insertion_needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_edge = left[-1]
    right_edge = right[0]
    return all(
        not char.isspace()
        and not unicodedata.category(char).startswith("P")
        and not _is_cjk_char(char)
        for char in (left_edge, right_edge)
    )


def render_source_text(instruction: str) -> str:
    """Render the plain source transcript from a structured edit instruction.

    Inserted content is absent in the source.  Removing an insertion opens a
    word boundary only when both neighbouring visible characters are ordinary
    non-CJK, non-punctuation, non-whitespace characters.
    """

    parts = _render_source_node(_parse_instruction(instruction))
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if isinstance(part, str):
            rendered.append(part)
            continue
        left = rendered[-1] if rendered else ""
        right = ""
        for following in parts[index + 1 :]:
            if isinstance(following, str) and following:
                right = following
                break
        if _source_insertion_needs_space(left, right):
            rendered.append(" ")
    return "".join(rendered)


def _is_cjk_char(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
    )


def _insertion_depth(text: str, *, require_closed: bool = False) -> int:
    depth = 0
    for match in _INS_TAG_RE.finditer(text):
        if match.group(1):
            if depth != 1:
                raise ValueError("Unbalanced </ins> tag in edit instruction.")
            depth = 0
        else:
            if depth != 0:
                raise ValueError("Nested <ins> tag in edit instruction.")
            depth = 1
    if require_closed and depth != 0:
        raise ValueError("Unclosed <ins> tag in edit instruction.")
    return depth


def _needs_speaker_transfer(left: str, right: str) -> bool:
    left_boundary_is_insertion = bool(
        re.search(r"</\s*ins\s*>$", left, re.IGNORECASE)
    )
    right_boundary_is_insertion = bool(
        re.match(r"<\s*ins\s*>", right, re.IGNORECASE)
    )
    already_marked = bool(
        re.search(r"<\s*spk_transfer\s*/\s*>$", left, re.IGNORECASE)
    )
    return (left_boundary_is_insertion or right_boundary_is_insertion) and not already_marked


def concat_edit_instruction_values(
    values: Iterable[Any],
    *,
    separators: Sequence[str] | None = None,
) -> str:
    """Join instruction blocks and mark insertion-driven speaker boundaries.

    ``separators`` is used by inference when transcript boundaries already decide
    spacing.  The training pipeline omits it to preserve its historical behavior.
    """

    parts = [str(value).strip() for value in values]
    if separators is not None and len(separators) != max(0, len(parts) - 1):
        raise ValueError("instruction separators must match instruction boundaries")

    result = ""
    for index, text in enumerate(parts):
        if not text:
            continue
        if not result:
            result = text
            continue
        if _insertion_depth(result) == 1:
            closing_tag = _BOUNDARY_CLOSING_INS_RE.match(text)
            if closing_tag is None:
                raise ValueError(
                    "An <ins> tag spanning an edit-concat boundary must close "
                    "at the beginning of the next instruction."
                )
            result = f"{result}</ins><spk_transfer/>"
            text = text[closing_tag.end() :].lstrip()
            if not text:
                continue
        separator = (
            separators[index - 1]
            if separators is not None
            else "" if _is_cjk_char(result[-1]) and _is_cjk_char(text[0]) else " "
        )
        if _needs_speaker_transfer(result, text):
            result = f"{result}<spk_transfer/>"
        result = f"{result}{separator}{text}"
        _insertion_depth(result)
    if result:
        _insertion_depth(result, require_closed=True)
    return result
