import re


def try_fix_equation_leq(latex: str, debug: bool = False) -> str:
    latex = re.sub(r"<", r"< ", latex)
    return latex
