from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

__all__ = ["ensure_cjk_font", "make_comparison", "save_compare"]

# Tokens for pixel-aware wrap: ASCII word, whitespace run, or a single CJK char.
_WRAP_TOKEN_RE = re.compile(r"\s+|[\u4e00-\u9fff]|[^\s\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Font search order: CJK-capable first so Chinese prompts render properly.
_CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk-vf-fonts/NotoSansCJK-VF.otf.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "./fonts/Noto_Sans_SC/static/NotoSansSC-Regular.ttf",
)
_LATIN_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)
_CJK_FONT_CACHE_DIR = Path.home() / ".cache" / "sensenova_u1" / "fonts"
_CJK_FONT_CACHE_PATH = _CJK_FONT_CACHE_DIR / "NotoSansCJKsc-Regular.otf"
_CJK_FONT_URLS = (
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
)

_warned_missing_cjk = False
_warned_download_cjk = False


def _try_load_truetype(path: str | Path, size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return None


def ensure_cjk_font(*, timeout: float = 30.0) -> Path | None:
    """Return a local CJK font path, downloading Noto Sans CJK SC if needed.

    The font is stored under the user's cache directory so this works in
    non-root containers and does not mutate system font directories.
    """
    for path in _CJK_FONTS:
        if _try_load_truetype(path, size=16) is not None:
            return Path(path)

    if _try_load_truetype(_CJK_FONT_CACHE_PATH, size=16) is not None:
        return _CJK_FONT_CACHE_PATH

    tmp_path = _CJK_FONT_CACHE_PATH.with_suffix(".tmp")
    _CJK_FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for url in _CJK_FONT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                tmp_path.write_bytes(response.read())
            if _try_load_truetype(tmp_path, size=16) is None:
                tmp_path.unlink(missing_ok=True)
                continue
            tmp_path.replace(_CJK_FONT_CACHE_PATH)
            return _CJK_FONT_CACHE_PATH
        except (OSError, urllib.error.URLError):
            tmp_path.unlink(missing_ok=True)
            continue
    return None


def _load_font(size: int, *, need_cjk: bool = False) -> tuple[ImageFont.ImageFont | ImageFont.FreeTypeFont, bool]:
    """Return (font, has_cjk_coverage). Falls back to PIL default if nothing usable."""
    for path in _CJK_FONTS:
        font = _try_load_truetype(path, size)
        if font is not None:
            return font, True
    if need_cjk:
        path = ensure_cjk_font()
        if path is not None:
            font = _try_load_truetype(path, size)
            if font is not None:
                return font, True
    for path in _LATIN_FONTS:
        font = _try_load_truetype(path, size)
        if font is not None:
            return font, False
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:
        return ImageFont.load_default(), False


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    """Greedy pixel-aware wrap. Keeps ASCII words intact, splits CJK per-char."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        cur = ""
        for tok in _WRAP_TOKEN_RE.findall(paragraph):
            candidate = cur + tok
            if font.getlength(candidate.rstrip()) <= max_width:
                cur = candidate
                continue
            if cur.strip():
                lines.append(cur.rstrip())
            cur = "" if tok.isspace() else tok
        if cur.rstrip():
            lines.append(cur.rstrip())
    return lines or [""]


def make_comparison(
    inputs: Sequence[Image.Image],
    output: Image.Image,
    prompt: str,
    *,
    pad: int = 16,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Return ``[inputs... | output]`` stacked horizontally with ``prompt`` below.

    Inputs are letterboxed to match the output's height so the row lines up
    cleanly regardless of aspect ratio.
    """
    row_h = output.size[1]
    row_imgs: list[Image.Image] = []
    for im in inputs:
        if im.size[1] != row_h:
            new_w = max(1, round(im.size[0] * row_h / im.size[1]))
            im = im.resize((new_w, row_h), Image.LANCZOS)
        row_imgs.append(im)
    row_imgs.append(output)
    row_w = sum(im.size[0] for im in row_imgs) + pad * (len(row_imgs) + 1)

    prompt_has_cjk = bool(_CJK_RE.search(prompt))
    font, has_cjk = _load_font(max(18, row_h // 30), need_cjk=prompt_has_cjk)
    global _warned_download_cjk, _warned_missing_cjk
    if prompt_has_cjk and has_cjk and not _warned_download_cjk:
        print("[compare] using a CJK-capable font for prompt rendering.")
        _warned_download_cjk = True
    if prompt_has_cjk and not has_cjk and not _warned_missing_cjk:
        print(
            "[compare] prompt contains CJK but no CJK-capable font was found; "
            "automatic font download failed and Chinese characters will render as tofu. "
            "Install e.g. `fonts-noto-cjk` (Debian/Ubuntu), `google-noto-cjk-fonts` "
            "(RHEL-family) for proper rendering."
        )
        _warned_missing_cjk = True

    lines = _wrap_text(prompt, font, row_w - pad * 2)
    bbox = font.getbbox("Ag中")
    line_h = max(1, int((bbox[3] - bbox[1]) * 1.3))
    text_h = line_h * len(lines) + pad * 2

    canvas = Image.new("RGB", (row_w, row_h + pad * 2 + text_h), bg)
    x = pad
    for im in row_imgs:
        canvas.paste(im, (x, pad))
        x += im.size[0] + pad
    draw = ImageDraw.Draw(canvas)
    y = row_h + pad * 2
    for line in lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    return canvas


def save_compare(
    out_path: Path,
    inputs: Sequence[Image.Image],
    output: Image.Image,
    prompt: str,
) -> None:
    """Save a comparison next to ``out_path`` as ``<stem>_compare<suffix>``."""
    cmp_path = out_path.with_name(f"{out_path.stem}_compare{out_path.suffix}")
    make_comparison(inputs, output, prompt).save(cmp_path)
    print(f"[saved] {cmp_path}")
