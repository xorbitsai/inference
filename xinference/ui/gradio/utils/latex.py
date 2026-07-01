# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LaTeX processing utilities for OCR text formatting.

This module provides functions to process LaTeX formulas in OCR text,
making them compatible with different output formats like Markdown,
HTML, and pure LaTeX.
"""

import re
from typing import Any, Dict, Union


def process_latex_formulas(text: str, output_format: str = "markdown") -> str:
    """
    Process LaTeX formulas in OCR text to make them compatible with different output formats.

    Args:
        text: The OCR text containing LaTeX formulas
        output_format: Target format ("markdown", "html", "latex", "gradio")

    Returns:
        Processed text with formulas converted to the target format
    """
    if not text:
        return text

    processed_text = text

    if output_format == "markdown":
        # Convert \[ ... \] to $$ ... $$ for block math in Markdown
        processed_text = re.sub(
            r"\\\[\s*(.*?)\s*\\\]",
            lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
            processed_text,
            flags=re.DOTALL,
        )

        # Convert \( ... \) to $ ... $ for inline math in Markdown
        processed_text = re.sub(
            r"\\\((.*?)\\\)",
            lambda m: f"${m.group(1).strip()}$",
            processed_text,
            flags=re.DOTALL,
        )

        # Handle common LaTeX math environments
        # Convert \begin{equation} ... \end{equation} to $$ ... $$
        processed_text = re.sub(
            r"\\begin\{equation\}(.*?)\\end\{equation\}",
            lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
            processed_text,
            flags=re.DOTALL,
        )

        # Convert \begin{align} ... \end{align} to $$ ... $$
        processed_text = re.sub(
            r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
            lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
            processed_text,
            flags=re.DOTALL,
        )

        # Convert \begin{gather} ... \end{gather} to $$ ... $$
        processed_text = re.sub(
            r"\\begin\{gather\}(.*?)\\end\{gather\}",
            lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
            processed_text,
            flags=re.DOTALL,
        )

    elif output_format == "html":
        # Convert \[ ... \] to <div>...</div> with proper formatting
        processed_text = re.sub(
            r"\\\[(.*?)\\\]",
            lambda m: f'<div class="math-display">\\[{m.group(1).strip()}\\]</div>',
            processed_text,
            flags=re.DOTALL,
        )

        # Convert \( ... \) to <span>...</span> for inline math
        processed_text = re.sub(
            r"\\\((.*?)\\\)",
            lambda m: f'<span class="math-inline">\\({m.group(1).strip()}\\)</span>',
            processed_text,
            flags=re.DOTALL,
        )

    elif output_format == "latex":
        # Keep original LaTeX format, just clean up spacing
        processed_text = re.sub(
            r"\\\[(.*?)\\\]",
            lambda m: f"\\[\n{m.group(1).strip()}\n\\]",
            processed_text,
            flags=re.DOTALL,
        )

        processed_text = re.sub(
            r"\\\((.*?)\\\)",
            lambda m: f"\\({m.group(1).strip()}\\)",
            processed_text,
            flags=re.DOTALL,
        )

    return processed_text


def clean_latex_syntax(text: str) -> str:
    """
    Clean up common LaTeX syntax issues in OCR text.

    Args:
        text: The OCR text containing LaTeX

    Returns:
        Cleaned LaTeX text
    """
    if not text:
        return text

    # Fix common OCR errors in LaTeX
    cleaned = text

    # Fix spacing around operators
    cleaned = re.sub(r"(\w)([+\-=*/])(\w)", r"\1 \2 \3", cleaned)

    # Fix double backslashes that might be mangled, but preserve LaTeX delimiters
    # Only fix excessive backslashes (3+), not double backslashes which are valid LaTeX
    cleaned = re.sub(r"\\\\{3,}", r"\\\\", cleaned)

    # Fix fractions that might be incorrectly spaced
    cleaned = re.sub(
        r"\\frac\s*\{\s*(\w+)\s*\}\s*\{\s*(\w+)\s*\}", r"\\frac{\1}{\2}", cleaned
    )

    # Fix superscripts and subscripts
    cleaned = re.sub(r"\^\s*\{\s*(\w+)\s*\}", r"^{\1}", cleaned)
    cleaned = re.sub(r"_\s*\{\s*(\w+)\s*\}", r"_{\1}", cleaned)

    # Fix common misrecognized Greek letters
    greek_corrections = {
        r"\\alpha\s": r"\\alpha ",
        r"\\beta\s": r"\\beta ",
        r"\\gamma\s": r"\\gamma ",
        r"\\delta\s": r"\\delta ",
        r"\\epsilon\s": r"\\epsilon ",
        r"\\theta\s": r"\\theta ",
        r"\\lambda\s": r"\\lambda ",
        r"\\mu\s": r"\\mu ",
        r"\\pi\s": r"\\pi ",
        r"\\sigma\s": r"\\sigma ",
        r"\\phi\s": r"\\phi ",
        r"\\omega\s": r"\\omega ",
        r"\\Delta\s": r"\\Delta ",
        r"\\Sigma\s": r"\\Sigma ",
        r"\\Pi\s": r"\\Pi ",
        r"\\infty\s": r"\\infty ",
        r"\\pm\s": r"\\pm ",
        r"\\times\s": r"\\times ",
        r"\\div\s": r"\\div ",
        r"\\neq\s": r"\\neq ",
        r"\\leq\s": r"\\leq ",
        r"\\geq\s": r"\\geq ",
        r"\\approx\s": r"\\approx ",
        r"\\in\s": r"\\in ",
        r"\\subset\s": r"\\subset ",
        r"\\supset\s": r"\\supset ",
        r"\\int\s": r"\\int ",
        r"\\sum\s": r"\\sum ",
        r"\\prod\s": r"\\prod ",
        r"\\partial\s": r"\\partial ",
        r"\\nabla\s": r"\\nabla ",
        r"\\sqrt\s": r"\\sqrt ",
        r"^\{2\}": r"²",
        r"^\{3\}": r"³",
    }

    for latex, unicode in greek_corrections.items():
        cleaned = re.sub(latex, unicode, cleaned)

    return cleaned


def process_ocr_latex(text: str, output_format: str = "markdown") -> str:
    """
    Convenience function to process OCR text containing LaTeX formulas.

    This function combines LaTeX syntax cleaning and formula formatting
    in a single call, which is useful for OCR post-processing.

    Args:
        text: The OCR text containing LaTeX formulas
        output_format: Target format ("markdown", "html", "latex", "gradio")

    Returns:
        Processed text with cleaned LaTeX syntax and properly formatted formulas
    """
    if not text:
        return text

    # First clean up LaTeX syntax issues
    cleaned_text = clean_latex_syntax(text)

    # Then convert formulas to the desired format
    processed_text = process_latex_formulas(cleaned_text, output_format)

    return processed_text


def contains_latex(text: str) -> bool:
    """
    Detect if text contains LaTeX formulas.

    Args:
        text: The text to check for LaTeX formulas

    Returns:
        True if the text contains LaTeX formulas, False otherwise
    """
    if not text:
        return False

    # Check for common LaTeX indicators
    latex_indicators = [
        r"\\\[.*?\\\]",  # Block formulas \[...\]
        r"\\\(.*?\\\)",  # Inline formulas \(...\)
        r"\$.*?\$",  # Inline formulas $...$
        r"\$\$.*?\$\$",  # Block formulas $$...$$
        r"\\begin\{.*?\}",  # LaTeX environments
        r"\\[a-zA-Z]+\{",  # LaTeX commands with arguments
        r"\\[a-zA-Z]+",  # Simple LaTeX commands
    ]

    for pattern in latex_indicators:
        if re.search(pattern, text):
            return True

    return False


def process_ocr_result_with_latex(
    result: Union[str, Dict[str, Any]],
    output_format: str = "markdown",
    debug_info: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Process OCR results, applying LaTeX formatting if needed.

    This is a unified function to handle both LaTeX detection and processing
    for OCR results, supporting both string and dict formats.

    Args:
        result: OCR result, either as string or dict with 'text' key
        output_format: Target format ("markdown", "html", "latex", "gradio")
        debug_info: Whether to print debug information about LaTeX processing

    Returns:
        Processed result with LaTeX formulas formatted appropriately
    """
    # Handle dict format (from visualize_ocr and other methods)
    if isinstance(result, dict):
        if "text" not in result:
            return result

        text = result["text"]
        if not contains_latex(text):
            return result

        # Process LaTeX
        processed_text = process_ocr_latex(text, output_format)

        # Create new result dict with processed text
        processed_result = result.copy()
        processed_result["text"] = processed_text
        processed_result["latex_processing"] = {
            "latex_detected": True,
            "output_format": output_format,
        }

        if debug_info:
            print("🧮 LaTeX Formula Processing:")
            original_block_formulas = len(re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL))
            original_inline_dollar = len(re.findall(r"\$(.*?)\$", text, re.DOTALL))
            original_inline_paren = len(re.findall(r"\\\((.*?)\\\)", text, re.DOTALL))
            converted_inline = len(re.findall(r"\$(.*?)\$", processed_text, re.DOTALL))

            print(f"  - Original block formulas (\\[...\\]): {original_block_formulas}")
            print(f"  - Original inline formulas ($...$): {original_inline_dollar}")
            print(f"  - Original inline formulas (\\(...\\)): {original_inline_paren}")
            print(f"  - Final inline formulas ($...$): {converted_inline}")
            print(
                f"  - Format: Markdown-compatible ($...$ for inline, $$...$$ for block)"
            )

        return processed_result

    # Handle string format
    else:
        # Ensure result is treated as string
        text_result = str(result)

        if not contains_latex(text_result):
            return result

        if debug_info:
            print("🧮 LaTeX Formula Processing:")
            original_block_formulas = len(
                re.findall(r"\\\[(.*?)\\\]", text_result, re.DOTALL)
            )
            original_inline_dollar = len(
                re.findall(r"\$(.*?)\$", text_result, re.DOTALL)
            )
            original_inline_paren = len(
                re.findall(r"\\\((.*?)\\\)", text_result, re.DOTALL)
            )

        processed_text = process_ocr_latex(text_result, output_format)

        if debug_info:
            converted_inline = len(re.findall(r"\$(.*?)\$", processed_text, re.DOTALL))
            print(f"  - Original block formulas (\\[...\\]): {original_block_formulas}")
            print(f"  - Original inline formulas ($...$): {original_inline_dollar}")
            print(f"  - Original inline formulas (\\(...\\)): {original_inline_paren}")
            print(f"  - Final inline formulas ($...$): {converted_inline}")
            print(
                f"  - Format: Markdown-compatible ($...$ for inline, $$...$$ for block)"
            )

        return processed_text
