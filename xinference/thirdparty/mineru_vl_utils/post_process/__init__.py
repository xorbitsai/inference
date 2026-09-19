from ..structs import ContentBlock
from .equation_block import do_handle_equation_block
from .equation_double_subscript import try_fix_equation_double_subscript
from .equation_fix_eqqcolon import try_fix_equation_eqqcolon
from .equation_big import try_fix_equation_big
from .equation_leq import try_fix_equation_leq
from .equation_left_right import try_match_equation_left_right
from .otsl2html import convert_otsl_to_html

PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}


def _process_equation(content: str, debug: bool) -> str:
    content = try_match_equation_left_right(content, debug=debug)
    content = try_fix_equation_double_subscript(content, debug=debug)
    content = try_fix_equation_eqqcolon(content, debug=debug)
    content = try_fix_equation_big(content, debug=debug)
    content = try_fix_equation_leq(content, debug=debug)
    return content


def _add_equation_brackets(content: str) -> str:
    content = content.strip()
    if not content.startswith("\\["):
        content = f"\\[\n{content}"
    if not content.endswith("\\]"):
        content = f"{content}\n\\]"
    return content


def post_process(
    blocks: list[ContentBlock],
    handle_equation_block: bool,
    abandon_list: bool,
    abandon_paratext: bool,
    debug: bool = False,
) -> list[ContentBlock]:
    for block in blocks:
        if block.type == "table" and block.content:
            block.content = convert_otsl_to_html(block.content)
        if block.type == "equation" and block.content:
            block.content = _process_equation(block.content, debug=debug)

    if handle_equation_block:
        blocks = do_handle_equation_block(blocks, debug=debug)

    for block in blocks:
        if block.type == "equation" and block.content:
            block.content = _add_equation_brackets(block.content)

    out_blocks: list[ContentBlock] = []
    for block in blocks:
        if block.type == "equation_block":  # drop equation_block anyway
            continue
        if abandon_list and block.type == "list":
            continue
        if abandon_paratext and block.type in PARATEXT_TYPES:
            continue
        out_blocks.append(block)

    return out_blocks
