# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
Helpers for PDF input on the ``/v1/images/ocr`` endpoint.

OCR models consume a single PIL image, so PDF uploads are rasterized
page by page and the per-page OCR results are merged back into one
JSON-serializable response body.
"""

import json
import logging
import os
import threading
from typing import Any, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# PDFium is not thread-safe: pypdfium2 documents that concurrent calls,
# even on different documents, may crash or corrupt the process. Every
# PDFium operation in this module (document open, page access, rendering,
# close) must hold this lock; model OCR calls happen outside it.
_pdfium_lock = threading.Lock()

PDF_MAGIC = b"%PDF"
DEFAULT_PDF_OCR_DPI = 200
# Rasterizing above this resolution rarely helps OCR quality but can
# exhaust memory on large pages.
MAX_PDF_OCR_DPI = 600
# One request OCRs at most this many pages.
MAX_PDF_OCR_PAGES = 200
# A page whose raster would exceed this many pixels (~240 MB of RGB
# pixels; an A3 page at 600 DPI is ~70 MP) is rejected so a single
# oversized page cannot exhaust the API process.
MAX_PDF_OCR_PAGE_PIXELS = 80_000_000
# OCR tasks that consume the whole PDF instead of one rasterized page at a
# time, and therefore receive the uploaded bytes unchanged.
WHOLE_DOCUMENT_OCR_TASKS = frozenset({"parse"})
# Whole-document parsers render at ``72 * zoomin`` DPI, i.e. a scale of
# ``zoomin`` over the page's point size. DeepDoc additionally re-renders at
# ``zoomin * 3`` when a pass finds no OCR boxes *anywhere in the document* --
# its recovery path for a render too coarse to read -- but only while
# ``zoomin < 9``, so a parse started at or above that never amplifies.
#
# The guard tests the pre-multiplication value, so the ladder overshoots the
# limit for any zoomin that is not a power-of-three divisor of 9: 2 escalates
# 2 -> 6 -> 18, and 6 goes straight to 18. That makes the reachable scale
# non-monotonic in zoomin (2 and 6 reach 18, but 3 stops at 9), which is a
# defect in deepdoc-lib rather than here; see ``worst_case_parse_zoom``.
PDF_PARSE_RETRY_ZOOM_FACTOR = 3
PDF_PARSE_RETRY_ZOOM_LIMIT = 9


def worst_case_parse_zoom(zoomin: int) -> int:
    """The largest scale a whole-document parse can actually render at.

    The retry is recursive: each pass that still finds no boxes triples the
    zoom again, and only the ``< limit`` guard stops it. So zoomin 1 renders
    at 1, 3 and finally 9, not just 3.

    Because deepdoc-lib checks ``zoomin < 9`` *before* multiplying, the last
    step can overshoot: 2 reaches 18 and 6 reaches 18, while 3 stops at 9.
    This models that faithfully rather than clamping, so the number used for
    budgeting is the one the parser can really allocate. The consequence is
    that this is not monotonic in ``zoomin`` -- which is why the per-page
    limit, the only budget still enforced at this scale, is checked against
    every candidate zoom before one is recommended to the caller.
    """
    scale = zoomin
    while scale < PDF_PARSE_RETRY_ZOOM_LIMIT:
        scale *= PDF_PARSE_RETRY_ZOOM_FACTOR
    return scale


def worst_case_parse_peak_pixels(width: float, height: float, zoomin: int) -> int:
    """Peak pixels one page can occupy during a whole-document parse.

    ``self.page_images = [...]`` builds the new list before rebinding the
    name, so the render that a retry replaces is still referenced while the
    replacement is being allocated. Peak is therefore the final render plus
    the one immediately before it.
    """
    final = worst_case_parse_zoom(zoomin)
    previous = final // PDF_PARSE_RETRY_ZOOM_FACTOR
    peak = int(width * final) * int(height * final)
    if previous >= zoomin:
        peak += int(width * previous) * int(height * previous)
    return peak


# Budgets for whole-document parsing. Measured against pdfplumber, a
# rendered page costs ~4 bytes per pixel once the PIL object is accounted
# for; deepdoc then adds crops, characters and a deepcopy of the boxes.
#
# The per-page ceiling is enforced at the worst-case (retry) scale, and is
# separate from MAX_PDF_OCR_PAGE_PIXELS because the two paths render
# differently -- reusing that limit here would reject an ordinary A3 page at
# the default zoom once the retry is accounted for. Sized to admit A4 and A3
# at the default zoom (41 and 81 MP worst-case) while still rejecting an
# outsized MediaBox: the 14400x14400 pt maximum is 1.9 G px at zoomin 3.
MAX_PDF_PARSE_PAGE_PIXELS = 200_000_000
# Unlike the per-page path, which rasterizes lazily and holds one page at a
# time, whole-document parsers render every page up front and keep them all
# alive at once, so the pages are budgeted together as well.
#
# This one is enforced at the *requested* scale. Charging every document for
# the retry is what stopped an ordinary 31-page A4 text PDF parsing at every
# zoomin (#5307): it was budgeted at 45 MP per page where it really renders
# at 4.5. The retry is not a state a text document reaches -- deepdoc's guard
# is ``len(self.boxes) == 0`` and ``boxes`` accumulates over every page, so it
# only fires when not one page in the whole document produced a single box.
#
# At 1 G px and ~4 bytes per rendered pixel this is ~4 GB of page images at
# the requested zoom: ~221 A4 pages at the default zoomin 3, 55 at zoomin 6,
# with the 200-page ceiling capping it from the other side.
MAX_PDF_PARSE_TOTAL_PIXELS = 1_000_000_000
# The retry is still real when it does fire, and it re-renders the *whole*
# document, so the requested-scale budget alone would leave a hole: 200 A4
# pages admitted at zoomin 3 would escalate to 8 G px (~32 GB). The escalated
# document therefore gets its own ceiling, checked at the worst-case scale.
#
# It is deliberately looser than the requested-scale budget rather than equal
# to it. The escalated render *replaces* the first one -- the retry re-enters
# ``__images__``, which rebinds ``self.page_images`` -- so the two are not
# summed; and reaching it at all requires a document that yielded no text
# anywhere, which is the rare case rather than the one being priced.
#
# At 3 G px this holds an escalated document to ~12 GB of page images, which
# admits 73 A4 pages at the default zoom. That is the binding constraint on
# long documents now, and it is a real one: a 200-page A4 document really
# would need ~32 GB if it escalated. Deployments whose parse workers are
# sized for more can raise both ceilings via the environment rather than
# being held to a default chosen for a modest worker.
MAX_PDF_PARSE_RETRY_TOTAL_PIXELS = 3_000_000_000


def _pixel_budget_from_env(name: str, default: int) -> int:
    """Read a pixel ceiling from the environment, falling back to ``default``.

    A malformed or non-positive value is ignored rather than raising: this
    runs at import time, and a typo in a deployment's environment should not
    take the API process down.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer", name, raw)
        return default
    if value <= 0:
        logger.warning("Ignoring %s=%r: must be positive", name, raw)
        return default
    return value


MAX_PDF_PARSE_TOTAL_PIXELS = _pixel_budget_from_env(
    "XINFERENCE_MAX_PDF_PARSE_TOTAL_PIXELS", MAX_PDF_PARSE_TOTAL_PIXELS
)
MAX_PDF_PARSE_RETRY_TOTAL_PIXELS = _pixel_budget_from_env(
    "XINFERENCE_MAX_PDF_PARSE_RETRY_TOTAL_PIXELS", MAX_PDF_PARSE_RETRY_TOTAL_PIXELS
)


def _parse_budget_error(sizes: List[Tuple[float, float]], zoomin: int) -> Optional[str]:
    """Why a document does not fit at ``zoomin``, or ``None`` if it does.

    Both budgets are applied: the per-page ceiling at the worst-case scale,
    since one oversized MediaBox must not be admitted on the strength of a
    retry that may still fire, and the two document-wide ceilings at the
    requested and worst-case scales respectively.
    """
    worst_case = worst_case_parse_zoom(zoomin)
    requested_total = 0
    retry_total = 0
    for index, (width, height) in enumerate(sizes):
        peak_pixels = worst_case_parse_peak_pixels(width, height, zoomin)
        if peak_pixels > MAX_PDF_PARSE_PAGE_PIXELS:
            return (
                f"Page {index + 1} would rasterize to {peak_pixels} pixels at "
                f"zoomin {zoomin:g} (the parser re-renders at up to "
                f"{worst_case:g}x when a document yields no text at all), "
                f"exceeding the per-page limit of {MAX_PDF_PARSE_PAGE_PIXELS}"
            )
        requested_total += int(width * zoomin) * int(height * zoomin)
        if requested_total > MAX_PDF_PARSE_TOTAL_PIXELS:
            return (
                f"The uploaded PDF would rasterize to more than "
                f"{requested_total} pixels in total at zoomin {zoomin:g}, "
                f"exceeding the whole-document limit of "
                f"{MAX_PDF_PARSE_TOTAL_PIXELS}"
            )
        retry_total += int(width * worst_case) * int(height * worst_case)
        if retry_total > MAX_PDF_PARSE_RETRY_TOTAL_PIXELS:
            return (
                f"The uploaded PDF would rasterize to more than {retry_total} "
                f"pixels in total if the parser re-rendered it at "
                f"{worst_case:g}x, which it does when a document yields no "
                f"text at all, exceeding the retry limit of "
                f"{MAX_PDF_PARSE_RETRY_TOTAL_PIXELS}"
            )
    return None


def largest_fitting_parse_zoom(
    sizes: List[Tuple[float, float]], upper_bound: int
) -> Optional[int]:
    """The largest zoom in ``1..upper_bound`` this document fits at.

    Not simply ``upper_bound`` counted down until something fits: because the
    retry ladder overshoots for zoom values that are not power-of-three
    divisors of 9, a *lower* zoomin can have a *higher* worst case (2 and 6
    both reach 18x, while 3 stops at 9x). Every candidate is therefore tested
    rather than assuming the budget shrinks as the zoom does.

    The search covers the whole range, not just values below the request, for
    the same reason: a document rejected at zoomin 2 (18x) may well fit at 3
    (9x), and sending the caller down to 1 would cost quality for nothing.
    """
    for candidate in range(upper_bound, 0, -1):
        if _parse_budget_error(sizes, candidate) is None:
            return candidate
    return None


def is_pdf_upload(content_type: Optional[str], head: bytes) -> bool:
    """Detect a PDF upload by content type or ``%PDF`` magic bytes."""
    if content_type and content_type.split(";")[0].strip() == "application/pdf":
        return True
    return head.startswith(PDF_MAGIC)


def validate_pdf_for_parse(
    data: bytes, zoomin: int, max_zoomin: Optional[int] = None
) -> int:
    """Check a PDF that will be handed to a whole-document parser.

    Whole-document parsing tasks (e.g. DeepDoc's ``task="parse"``) render the
    PDF themselves, so the bytes are passed straight through instead of being
    rasterized here. That skips everything ``rasterize_pdf`` would have
    validated, and parsers tend to fail unhelpfully: DeepDoc swallows load
    errors and returns an empty result, then re-renders at three times the
    zoom when it finds no boxes.

    Page geometry therefore has to be checked here too, in two ways. A single
    valid page with an outsized MediaBox — 14400x14400 points is legal —
    rasterizes to billions of pixels on its own. And because every page is
    rendered up front and held together, a document whose pages are each
    comfortably under the per-page limit can still exhaust the worker in
    aggregate, so the pages are budgeted as a whole too.

    The per-page ceiling is applied at the worst-case scale, since a single
    oversized MediaBox must not be admitted on the strength of a retry that
    may still fire, and it counts the render being replaced alongside its
    replacement (see ``worst_case_parse_peak_pixels``). The document-wide
    ceiling is applied at the requested scale, with a separate, looser one
    bounding what a retry would re-render the document at; see
    ``MAX_PDF_PARSE_TOTAL_PIXELS`` for why the two differ.

    Returns the page count. Raises ``ValueError`` if the document cannot be
    opened, has no pages, has too many pages, or would rasterize to too many
    pixels on any single page or across the document. The message names a
    ``zoomin`` that would fit whenever one exists, rather than advising the
    caller to lower it -- the retry ladder is not monotonic, so lowering it
    can make the budget larger.

    ``max_zoomin`` bounds the search for that recommendation; it should be the
    largest value the endpoint accepts. It defaults to ``zoomin``, which only
    searches downwards -- pass the real ceiling so a request rejected at a
    zoom with an overshooting ladder (2 or 6, which reach 18x) can be pointed
    at a higher one with a shorter ladder (3, which stops at 9x).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        error_message = "Failed to import module 'pypdfium2' required for PDF OCR"
        installation_guide = [
            "Please make sure 'pypdfium2' is installed, e.g. with `pip install pypdfium2`\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    with _pdfium_lock:
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as e:
            raise ValueError(f"Could not read the uploaded PDF: {e}") from e
        try:
            page_count = len(pdf)
            if page_count < 1:
                raise ValueError("The uploaded PDF has no pages")
            if page_count > MAX_PDF_OCR_PAGES:
                raise ValueError(
                    f"The uploaded PDF has {page_count} pages, at most "
                    f"{MAX_PDF_OCR_PAGES} pages can be parsed per request"
                )
            sizes = []
            for page_number in range(1, page_count + 1):
                page = pdf[page_number - 1]
                try:
                    sizes.append(page.get_size())
                finally:
                    page.close()
        finally:
            pdf.close()

    reason = _parse_budget_error(sizes, zoomin)
    if reason is not None:
        # "Lower `zoomin`" was the old advice and it is not sound: the retry
        # ladder is not monotonic, so a lower zoomin can be budgeted *higher*
        # (2 and 6 both escalate to 18x where 3 stops at 9x). Name a zoom that
        # actually fits, and only fall back to splitting when none does.
        # Search the whole permitted range, not just below the request: with a
        # non-monotonic ladder a higher zoom can be the one that fits.
        fitting = largest_fitting_parse_zoom(sizes, max(max_zoomin or zoomin, zoomin))
        if fitting is None:
            raise ValueError(f"{reason}; split the document into smaller parts")
        raise ValueError(f"{reason}; retry with `zoomin` {fitting}")

    return page_count


def normalize_pages(
    pages: Optional[Union[int, List[int]]], page_count: int
) -> List[int]:
    """Validate the ``pages`` kwarg and return 1-based page numbers.

    ``None`` selects all pages.
    """
    if pages is None:
        return list(range(1, page_count + 1))
    if isinstance(pages, int) and not isinstance(pages, bool):
        pages = [pages]
    if (
        not isinstance(pages, (list, tuple))
        or not pages
        or not all(isinstance(p, int) and not isinstance(p, bool) for p in pages)
    ):
        raise ValueError(
            "`pages` must be a 1-based page number or a non-empty list of "
            f"1-based page numbers, got: {pages!r}"
        )
    for p in pages:
        if p < 1 or p > page_count:
            raise ValueError(
                f"Page {p} is out of range, the PDF has {page_count} page(s)"
            )
    return list(pages)


def rasterize_pdf(
    data: bytes,
    pages: Optional[Union[int, List[int]]] = None,
    dpi: Union[int, float] = DEFAULT_PDF_OCR_DPI,
) -> Generator[Tuple[int, Any], None, None]:
    """Rasterize a PDF into PIL images, one page at a time.

    Returns an iterator of ``(page_number, PIL.Image)`` tuples, page
    numbers being 1-based. Page selection and page sizes are validated
    eagerly (raising ``ValueError``) so invalid requests fail before any
    OCR runs; rendering itself is lazy so only one page's pixels are
    alive at a time. The caller must exhaust or ``close()`` the iterator
    to release the underlying document.

    Safe to call from multiple threads: all PDFium operations are
    serialized on a module-level lock (PDFium itself is not
    thread-safe), which is released between pages while OCR runs.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        error_message = "Failed to import module 'pypdfium2' required for PDF OCR"
        installation_guide = [
            "Please make sure 'pypdfium2' is installed, e.g. with `pip install pypdfium2`\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    if not isinstance(dpi, (int, float)) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError(f"`dpi` must be a positive number, got: {dpi!r}")
    dpi = min(float(dpi), float(MAX_PDF_OCR_DPI))
    scale = dpi / 72.0

    with _pdfium_lock:
        pdf = pdfium.PdfDocument(data)
        try:
            page_numbers = normalize_pages(pages, len(pdf))
            if len(page_numbers) > MAX_PDF_OCR_PAGES:
                raise ValueError(
                    f"{len(page_numbers)} pages selected, at most "
                    f"{MAX_PDF_OCR_PAGES} pages can be OCRed per request; "
                    "use `pages` to select a subset"
                )
            for page_number in page_numbers:
                page = pdf[page_number - 1]
                try:
                    width, height = page.get_size()
                finally:
                    page.close()
                pixels = int(width * scale) * int(height * scale)
                if pixels > MAX_PDF_OCR_PAGE_PIXELS:
                    raise ValueError(
                        f"Page {page_number} would rasterize to {pixels} pixels "
                        f"at {dpi:g} DPI, exceeding the limit of "
                        f"{MAX_PDF_OCR_PAGE_PIXELS}; lower `dpi`"
                    )
        except Exception:
            pdf.close()
            raise

    def _render() -> Generator[Tuple[int, Any], None, None]:
        try:
            for page_number in page_numbers:
                # hold the lock only while PDFium renders; it is released
                # at the yield so OCR on this page can overlap with other
                # requests' PDFium work
                with _pdfium_lock:
                    page = pdf[page_number - 1]
                    try:
                        bitmap = page.render(scale=scale)
                        try:
                            image = bitmap.to_pil()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                yield page_number, image
        finally:
            with _pdfium_lock:
                pdf.close()

    return _render()


def merge_ocr_page_results(page_results: List[Tuple[int, Any]]) -> str:
    """Merge per-page OCR results into one JSON response body.

    When every page yields plain text, the texts are joined so the
    response stays a JSON string like the single-image contract.
    Otherwise (e.g. ``return_dict=True`` style models) a per-page
    structure is returned. Either way the body parses with
    ``response.json()``.
    """
    if all(isinstance(result, str) for _, result in page_results):
        joined = "\n\n".join(result for _, result in page_results)
        return json.dumps(joined, ensure_ascii=False)
    merged = {
        "pages": [
            {"page": page_number, "result": result}
            for page_number, result in page_results
        ]
    }
    return json.dumps(merged, ensure_ascii=False)
