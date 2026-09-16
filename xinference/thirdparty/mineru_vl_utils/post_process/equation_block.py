import re

from ..structs import ContentBlock


def _bbox_cover_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if areaB == 0:
        return 0.0
    ratio = interArea / areaB
    return ratio


def _combined_equations(equation_contents):

    total_tags = 0
    for equation_content in equation_contents:
        total_tags += len(re.findall(r"\\tag\s*\{[^}]*\}", equation_content))

    # multiple "\tags" not allowed
    if total_tags > 1:
        processed_contents = []
        for equation_content in equation_contents:
            # replace \tag{...} to (...)
            processed_content = re.sub(r"\\tag\s*\{([^}]*)\}", r"(\1)", equation_content)
            processed_contents.append(processed_content)
        equation_contents = processed_contents

    combined_content = "\\begin{array}{l} "
    for equation_content in equation_contents:
        combined_content += equation_content + " \\\\ "
    combined_content += "\\end{array}"
    return combined_content


def do_handle_equation_block(
    blocks: list[ContentBlock],
    debug: bool = False,
) -> list[ContentBlock]:
    sem_equation_indices: list[int] = []
    span_equation_indices: list[int] = []
    for idx, block in enumerate(blocks):
        if block.type == "equation_block":
            sem_equation_indices.append(idx)
        elif block.type == "equation":
            span_equation_indices.append(idx)

    sem_equation_spans: dict[int, list[int]] = {}
    for sem_idx in sem_equation_indices:
        covered_span_indices = [
            span_idx
            for span_idx in span_equation_indices
            if _bbox_cover_ratio(
                blocks[sem_idx].bbox,
                blocks[span_idx].bbox,
            )
            > 0.9
        ]
        if len(covered_span_indices) > 1:
            sem_equation_spans[sem_idx] = covered_span_indices

    if debug:
        for idx, span_indices in sem_equation_spans.items():
            print(f"Combined equation_block at idx {idx} with spans at {span_indices}")

    out_blocks: list[ContentBlock] = []
    for idx in range(len(blocks)):
        block = blocks[idx]
        if any(idx in span_indices for span_indices in sem_equation_spans.values()):
            continue
        if idx in sem_equation_spans:
            span_indices = sem_equation_spans[idx]
            span_equation_contents = [blocks[span_idx].content for span_idx in span_indices]
            sem_equation_content = _combined_equations(span_equation_contents)
            out_blocks.append(
                ContentBlock(
                    type="equation",
                    bbox=block.bbox,
                    angle=block.angle,
                    content=sem_equation_content,
                )
            )
            continue
        if block.type == "equation_block":
            continue
        out_blocks.append(block)
    return out_blocks
