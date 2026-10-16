import re


def try_fix_equation_eqqcolon(latex: str, debug: bool = False) -> str:
    new_latex = re.sub(r"\\eqqcolon", "=:", latex)
    new_latex = re.sub(r"\\coloneqq", ":=", new_latex)
    if debug and new_latex != latex:
        print(f"Fixed equation eq-colon from: {latex} to: {new_latex}")
    return new_latex


if __name__ == "__main__":
    latex = r"a \\coloneqq b"
    print(try_fix_equation_eqqcolon(latex))
