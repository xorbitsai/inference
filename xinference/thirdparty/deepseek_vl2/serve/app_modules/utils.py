# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# -*- coding:utf-8 -*-
from __future__ import annotations

import html
import logging
import io
import os
import re
import base64
import time
from PIL import Image, ImageDraw, ImageFont

import mdtex2html
from markdown import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import ClassNotFound, get_lexer_by_name, guess_lexer

from deepseek_vl2.serve.app_modules.presets import (
    ALREADY_CONVERTED_MARK,
    BOX2COLOR,
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE
)

logger = logging.getLogger("gradio_logger")


def configure_logger():
    logger = logging.getLogger("gradio_logger")
    logger.setLevel(logging.DEBUG)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("deepseek_vl2/serve/logs", exist_ok=True)
    file_handler = logging.FileHandler(
        f"deepseek_vl2/serve/logs/{timestr}_gradio_log.log"
    )
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def strip_stop_words(x, stop_words):
    for w in stop_words:
        if w in x:
            return x[: x.index(w)].strip()
    return x.strip()


def format_output(history, text, x):
    updated_history = history + [[text, x]]
    a = [[y[0], convert_to_markdown(y[1])] for y in updated_history]
    return a, updated_history


def markdown_to_html_with_syntax_highlight(md_str):  # deprecated
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)

        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("text", stripall=True)

        formatter = HtmlFormatter()
        highlighted_code = highlight(code, lexer, formatter)

        return f'<pre><code class="{lang}">{highlighted_code}</code></pre>'

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:  # deprecated
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(
                r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()
            ):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    result += ALREADY_CONVERTED_MARK
    return result


def convert_asis(userinput):
    return f'<p style="white-space:pre-wrap;">{html.escape(userinput)}</p>{ALREADY_CONVERTED_MARK}'


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    return any(s.endswith(stop_word) for stop_word in stop_words)


def detect_converted_mark(userinput):
    return bool(userinput.endswith(ALREADY_CONVERTED_MARK))


def detect_language(code):
    first_line = "" if code.startswith("\n") else code.strip().split("\n", 1)[0]
    language = first_line.lower() if first_line else ""
    code_without_language = code[len(first_line) :].lstrip() if first_line else code
    return language, code_without_language


def convert_to_markdown(text):
    text = text.replace("$", "&#36;")
    text = text.replace("\r\n", "\n")

    def replace_leading_tabs_and_spaces(line):
        new_line = []

        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line) :]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += f"{line}\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += f"{line}\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text


def add_language_tag(text):
    def detect_language(code_block):
        try:
            lexer = guess_lexer(code_block)
            return lexer.name.lower()
        except ClassNotFound:
            return ""

    code_block_pattern = re.compile(r"(```)(\w*\n[^`]+```)", re.MULTILINE)

    def replacement(match):
        code_block = match.group(2)
        if match.group(2).startswith("\n"):
            language = detect_language(code_block)
            return (
                f"```{language}{code_block}```" if language else f"```\n{code_block}```"
            )
        else:
            return match.group(1) + code_block + "```"

    text2 = code_block_pattern.sub(replacement, text)
    return text2


def is_variable_assigned(var_name: str) -> bool:
    return var_name in locals()


def pil_to_base64(
    image: Image.Image,
    alt: str = "user upload image",
    resize: bool = True,
    max_size: int = MAX_IMAGE_SIZE,
    min_size: int = MIN_IMAGE_SIZE,
    format: str = "JPEG",
    quality: int = 95
) -> str:

    if resize:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_size / aspect_ratio, min_size, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))

    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{alt}" />'

    return img_str


def parse_ref_bbox(response, image: Image.Image):
    try:
        image = image.copy()
        image_w, image_h = image.size
        draw = ImageDraw.Draw(image)

        ref = re.findall(r'<\|ref\|>.*?<\|/ref\|>', response)
        bbox = re.findall(r'<\|det\|>.*?<\|/det\|>', response)
        assert len(ref) == len(bbox)

        if len(ref) == 0:
            return None

        boxes, labels = [], []
        for box, label in zip(bbox, ref):
            box = box.replace('<|det|>', '').replace('<|/det|>', '')
            label = label.replace('<|ref|>', '').replace('<|/ref|>', '')
            box = box[1:-1]
            for onebox in re.findall(r'\[.*?\]', box):
                boxes.append(eval(onebox))
                labels.append(label)

        for indice, (box, label) in enumerate(zip(boxes, labels)):
            box = (
                int(box[0] / 999 * image_w),
                int(box[1] / 999 * image_h),
                int(box[2] / 999 * image_w),
                int(box[3] / 999 * image_h),
            )

            box_color = BOX2COLOR[indice % len(BOX2COLOR.keys())]
            box_width = 3
            draw.rectangle(box, outline=box_color, width=box_width)

            text_x = box[0]
            text_y = box[1] - 20
            text_color = box_color
            font = ImageFont.truetype("deepseek_vl2/serve/assets/simsun.ttc", size=20)
            draw.text((text_x, text_y), label, font=font, fill=text_color)

        # print(f"boxes = {boxes}, labels = {labels}, re-render = {image}")
        return image
    except:
        return None


def display_example(image_list):
    images_html = ""
    for i, img_path in enumerate(image_list):
        image = Image.open(img_path)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=100)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{img_path}" style="height:80px; margin-right: 10px;" />'
        images_html += img_str

    result_html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <div style="flex: 1; margin-right: 10px;">{images_html}</div>
    </div>
    """

    return result_html

