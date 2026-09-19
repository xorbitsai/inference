import re


def try_fix_equation_double_subscript(latex: str, debug: bool = False) -> str:
    pattern = r"_\s*\{([^{}]|\{[^{}]*\})*\}\s*_\s*\{([^{}]|\{[^{}]*\})*\}"
    if not re.search(pattern, latex):
        return latex
    new_latex = re.sub(pattern, "", latex)
    if debug:
        print(f"Fixed equation double-subscript from: {latex} to: {new_latex}")
    return new_latex
