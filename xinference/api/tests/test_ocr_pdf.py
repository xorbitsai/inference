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

"""Tests for PDF input support on the ``/v1/images/ocr`` endpoint helpers."""

import json
import os
from unittest import mock

import pytest

from ..pdf_ocr import (
    DEFAULT_PDF_OCR_DPI,
    MAX_PDF_OCR_PAGES,
    MAX_PDF_PARSE_PAGE_PIXELS,
    MAX_PDF_PARSE_RETRY_TOTAL_PIXELS,
    MAX_PDF_PARSE_TOTAL_PIXELS,
    PDF_PARSE_RETRY_ZOOM_LIMIT,
    WHOLE_DOCUMENT_OCR_TASKS,
    _parse_budget_error,
    is_pdf_upload,
    largest_fitting_parse_zoom,
    merge_ocr_page_results,
    normalize_pages,
    rasterize_pdf,
    validate_pdf_for_parse,
    worst_case_parse_peak_pixels,
    worst_case_parse_zoom,
)

A4 = "0 0 595 842"


def make_pdf(page_count: int = 1, media_box: str = "0 0 200 100") -> bytes:
    """Build a minimal valid PDF with ``page_count`` blank pages."""
    kids = " ".join(f"{3 + i} 0 R" for i in range(page_count))
    objects = [
        b"<</Type /Catalog /Pages 2 0 R>>",
        f"<</Type /Pages /Kids [{kids}] /Count {page_count}>>".encode(),
    ]
    objects.extend(
        f"<</Type /Page /Parent 2 0 R /MediaBox [{media_box}]>>".encode()
        for _ in range(page_count)
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objects) + 1} /Root 1 0 R>>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


class TestIsPdfUpload:
    def test_by_content_type(self):
        assert is_pdf_upload("application/pdf", b"")
        assert is_pdf_upload("application/pdf; charset=binary", b"")

    def test_by_magic_bytes(self):
        assert is_pdf_upload(None, b"%PDF-1.4\n...")
        assert is_pdf_upload("application/octet-stream", b"%PDF-1.7")

    def test_negative(self):
        assert not is_pdf_upload("image/png", b"\x89PNG\r\n\x1a\n")
        assert not is_pdf_upload(None, b"\xff\xd8\xff\xe0")
        assert not is_pdf_upload("application/octet-stream", b"")


class TestNormalizePages:
    def test_none_selects_all(self):
        assert normalize_pages(None, 3) == [1, 2, 3]

    def test_single_int(self):
        assert normalize_pages(2, 3) == [2]

    def test_list(self):
        assert normalize_pages([3, 1], 3) == [3, 1]

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            normalize_pages([4], 3)
        with pytest.raises(ValueError, match="out of range"):
            normalize_pages([0], 3)

    def test_invalid_types(self):
        with pytest.raises(ValueError):
            normalize_pages("1", 3)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            normalize_pages([1.5], 3)  # type: ignore[list-item]
        with pytest.raises(ValueError):
            normalize_pages([], 3)
        with pytest.raises(ValueError):
            normalize_pages(True, 3)  # type: ignore[arg-type]


class TestMergeOcrPageResults:
    def test_all_text_joined(self):
        body = merge_ocr_page_results([(1, "hello"), (2, "world")])
        assert json.loads(body) == "hello\n\nworld"

    def test_single_text_page_matches_image_contract(self):
        body = merge_ocr_page_results([(1, "hello")])
        assert json.loads(body) == "hello"

    def test_dict_results_keep_page_structure(self):
        results = [(1, {"blocks": ["a"]}), (3, {"blocks": ["b"]})]
        body = merge_ocr_page_results(results)
        assert json.loads(body) == {
            "pages": [
                {"page": 1, "result": {"blocks": ["a"]}},
                {"page": 3, "result": {"blocks": ["b"]}},
            ]
        }

    def test_mixed_results_keep_page_structure(self):
        body = merge_ocr_page_results([(1, "text"), (2, {"k": "v"})])
        assert json.loads(body)["pages"][0] == {"page": 1, "result": "text"}


class TestRasterizePdf:
    def test_all_pages(self):
        pytest.importorskip("pypdfium2")
        images = list(rasterize_pdf(make_pdf(page_count=2)))
        assert [page_number for page_number, _ in images] == [1, 2]
        for _, image in images:
            # 200x100pt page at the default 200 DPI
            expected = (
                round(200 * DEFAULT_PDF_OCR_DPI / 72),
                round(100 * DEFAULT_PDF_OCR_DPI / 72),
            )
            assert (image.width, image.height) == expected

    def test_page_selection_and_dpi(self):
        pytest.importorskip("pypdfium2")
        images = list(rasterize_pdf(make_pdf(page_count=3), pages=[2], dpi=72))
        assert len(images) == 1
        page_number, image = images[0]
        assert page_number == 2
        assert (image.width, image.height) == (200, 100)

    def test_dpi_capped(self):
        pytest.importorskip("pypdfium2")
        images = list(rasterize_pdf(make_pdf(), dpi=100000))
        _, image = images[0]
        assert image.width == round(200 * 600 / 72)

    def test_invalid_dpi(self):
        pytest.importorskip("pypdfium2")
        with pytest.raises(ValueError, match="dpi"):
            rasterize_pdf(make_pdf(), dpi=0)

    def test_invalid_pages_raise_before_rendering(self):
        pytest.importorskip("pypdfium2")
        # validation is eager: the error surfaces at call time, not on
        # first iteration
        with pytest.raises(ValueError, match="out of range"):
            rasterize_pdf(make_pdf(page_count=2), pages=[3])

    def test_rendering_is_lazy(self):
        pytest.importorskip("pypdfium2")
        page_iter = rasterize_pdf(make_pdf(page_count=3))
        assert not isinstance(page_iter, (list, tuple))
        page_number, image = next(page_iter)
        assert page_number == 1
        image.close()
        page_iter.close()

    def test_page_count_limit(self):
        pytest.importorskip("pypdfium2")
        with pytest.raises(ValueError, match="at most"):
            rasterize_pdf(make_pdf(page_count=MAX_PDF_OCR_PAGES + 1))
        # selecting a subset of a large document is fine
        images = list(
            rasterize_pdf(make_pdf(page_count=MAX_PDF_OCR_PAGES + 1), pages=[1])
        )
        assert len(images) == 1

    def test_page_pixel_limit(self):
        pytest.importorskip("pypdfium2")
        # 14400x14400pt (the PDF spec maximum) is ~1.6e9 pixels at 200 DPI
        huge = make_pdf(media_box="0 0 14400 14400")
        with pytest.raises(ValueError, match="lower `dpi`"):
            rasterize_pdf(huge)
        # the same page is fine at a low enough DPI
        images = list(rasterize_pdf(huge, dpi=20))
        assert len(images) == 1
        images[0][1].close()

    def test_concurrent_rasterization(self):
        """PDFium is not thread-safe; the module-level lock must let
        concurrent callers interleave safely without deadlocking."""
        pytest.importorskip("pypdfium2")
        from concurrent.futures import ThreadPoolExecutor

        def worker(_):
            pages = []
            for page_number, image in rasterize_pdf(make_pdf(page_count=3)):
                pages.append(page_number)
                image.close()
            return pages

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(worker, range(16)))
        assert results == [[1, 2, 3]] * 16

    def test_pdf_magic_detected_on_generated_pdf(self):
        assert is_pdf_upload(None, make_pdf()[:8])


class TestValidatePdfForParse:
    """Whole-document tasks get the uploaded bytes without going through
    ``rasterize_pdf``, so this is the only place a bad PDF is caught before
    the parser sees it."""

    def test_accepts_a_valid_pdf_and_returns_the_page_count(self):
        pytest.importorskip("pypdfium2")
        assert validate_pdf_for_parse(make_pdf(page_count=3), 3) == 3

    def test_rejects_non_pdf_bytes(self):
        pytest.importorskip("pypdfium2")
        with pytest.raises(ValueError, match="Could not read the uploaded PDF"):
            validate_pdf_for_parse(b"\x89PNG\r\n\x1a\n not a pdf", 3)

    def test_rejects_empty_input(self):
        pytest.importorskip("pypdfium2")
        with pytest.raises(ValueError):
            validate_pdf_for_parse(b"", 3)

    def test_rejects_too_many_pages(self):
        pytest.importorskip("pypdfium2")
        oversized = make_pdf(page_count=MAX_PDF_OCR_PAGES + 1)
        with pytest.raises(ValueError, match="at most"):
            validate_pdf_for_parse(oversized, 3)

    def test_page_count_ceiling_is_shared_with_the_per_page_path(self):
        # The page *count* ceiling is the same for both tasks. The pixel
        # budgets are not -- parse enforces its own, against the retry scale
        # -- so this uses the small default page size to isolate the count.
        pytest.importorskip("pypdfium2")
        assert validate_pdf_for_parse(make_pdf(page_count=MAX_PDF_OCR_PAGES), 3) == (
            MAX_PDF_OCR_PAGES
        )

    def test_rejects_an_oversized_page(self):
        # A single page with a huge (but legal) MediaBox rasterizes to
        # billions of pixels long before the page limit is relevant, so the
        # page-pixel budget has to be enforced here too.
        pytest.importorskip("pypdfium2")
        oversized = make_pdf(media_box="0 0 14400 14400")
        with pytest.raises(ValueError, match="per-page limit"):
            validate_pdf_for_parse(oversized, 3)

    def test_page_budget_accounts_for_the_retry_scale(self):
        # deepdoc re-renders at 3x when a page yields no text, and that
        # retry is left intact, so the per-page budget is enforced against
        # the scale a run can actually reach.
        pytest.importorskip("pypdfium2")
        assert worst_case_parse_zoom(3) == 9
        # 1700x1700 pt: 26 MP at zoomin 3, but 234 MP at the 9x it may
        # actually render at.
        borderline = make_pdf(media_box="0 0 1700 1700")
        assert int(1700 * 3) ** 2 < MAX_PDF_PARSE_PAGE_PIXELS
        assert worst_case_parse_peak_pixels(1700, 1700, 3) > MAX_PDF_PARSE_PAGE_PIXELS
        with pytest.raises(ValueError, match="per-page limit"):
            validate_pdf_for_parse(borderline, zoomin=3)

    def test_common_page_sizes_pass_at_the_default_zoom(self):
        # The budget must not reject ordinary documents. A4, Letter and A3
        # come to 41, 39 and 81 MP at the zoomin-3 worst case.
        pytest.importorskip("pypdfium2")
        for media_box in ("0 0 595 842", "0 0 612 792", "0 0 842 1191"):
            assert validate_pdf_for_parse(make_pdf(media_box=media_box), 3) == 1

    def test_budget_scales_with_zoomin(self):
        pytest.importorskip("pypdfium2")
        # A3 is 81 MP at the zoomin-3 worst case but 325 MP at zoomin 6.
        a3 = make_pdf(media_box="0 0 842 1191")
        assert validate_pdf_for_parse(a3, zoomin=3) == 1
        with pytest.raises(ValueError, match="per-page limit"):
            validate_pdf_for_parse(a3, zoomin=6)

    def test_rejects_an_oversized_document_of_individually_small_pages(self):
        # Whole-document parsers render every page up front and hold them all
        # at once, so a document whose pages each pass the per-page limit can
        # still exhaust the worker in aggregate.
        pytest.importorskip("pypdfium2")
        # 1000x1000 pt at zoomin 4: 16 MP per page, well under the per-page
        # cap even at the 12x worst case (144 MP), but 200 of them are
        # 3.2 G px together at the requested scale alone.
        per_page = int(1000 * 4) ** 2
        assert int(1000 * 12) ** 2 < MAX_PDF_PARSE_PAGE_PIXELS
        assert per_page * MAX_PDF_OCR_PAGES > MAX_PDF_PARSE_TOTAL_PIXELS
        doc = make_pdf(page_count=MAX_PDF_OCR_PAGES, media_box="0 0 1000 1000")
        with pytest.raises(ValueError, match="whole-document limit|retry limit"):
            validate_pdf_for_parse(doc, zoomin=4)

    def test_the_requested_scale_is_what_shapes_ordinary_acceptance(self):
        # deepdoc's `__ocr` appends to `boxes` on every page, including
        # `append([])` for a page that yields nothing, so `len(boxes) == 0`
        # cannot hold for a document that rendered any pages -- the 9x retry
        # is unreachable in deepdoc-lib 0.2.2. So the requested scale, not the
        # retry scale, is what an ordinary document is charged for: 100 A4
        # pages are 0.45 G px at zoomin 3 and are admitted, where budgeting
        # every page at 9x would have rejected them. The retry ceiling is a
        # backstop for a later release and is exercised separately.
        pytest.importorskip("pypdfium2")
        a4 = make_pdf(page_count=100, media_box=A4)
        assert validate_pdf_for_parse(a4, zoomin=3) == 100


class TestParseAdmitsRealDocuments:
    """#5307: an ordinary 31-page A4 text PDF was rejected at every zoomin.

    It really renders to ~140 MP at the default zoom, but was budgeted as if
    every page would trigger the 9x retry, which put it 40% over the ceiling.
    """

    PAGES = 31

    def test_thirty_one_page_a4_parses_at_the_default_zoom(self):
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=self.PAGES, media_box=A4)
        assert validate_pdf_for_parse(doc, zoomin=3) == self.PAGES

    def test_the_document_that_was_rejected_everywhere_now_has_options(self):
        # The bug was not just that the default failed -- no zoomin in 1..6
        # worked at all, so the document could not be parsed by any request.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=self.PAGES, media_box=A4)
        accepted = []
        for zoomin in range(1, 7):
            try:
                validate_pdf_for_parse(doc, zoomin)
            except ValueError:
                continue
            accepted.append(zoomin)
        assert 3 in accepted, "the default zoom must work"
        assert len(accepted) > 1

    def test_the_default_zoom_allowance_is_materially_larger_than_before(self):
        # The old aggregate budget capped the default zoom at ~22 A4 pages,
        # which is short for the reports and papers this feature targets.
        # Budgeting at the requested scale lifts that to ~130, where the
        # retry ceiling takes over as the binding constraint.
        pytest.importorskip("pypdfium2")
        for page_count in (73, 74, 100):
            doc = make_pdf(page_count=page_count, media_box=A4)
            assert validate_pdf_for_parse(doc, zoomin=3) == page_count


class TestParseRejectionAdviceIsActionable:
    """The 400 must never recommend something that makes matters worse."""

    def test_never_advises_lowering_zoomin(self):
        # "lower `zoomin`" was the old advice and it is unsound: the retry
        # ladder is not monotonic, so a lower zoomin can be budgeted higher.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=MAX_PDF_OCR_PAGES, media_box=A4)
        for zoomin in range(1, 7):
            try:
                validate_pdf_for_parse(doc, zoomin)
            except ValueError as e:
                assert "lower `zoomin`" not in str(e)

    def test_the_recommended_zoom_actually_fits(self):
        # Whatever zoom the message names must itself be accepted, otherwise
        # the caller is sent round the loop again.
        pytest.importorskip("pypdfium2")
        import re

        # 100 A4 pages fit at zoomin 1 and 3 but not the rest, so the higher
        # zooms exercise the recommendation path.
        doc = make_pdf(page_count=100, media_box=A4)
        for zoomin in range(1, 7):
            try:
                validate_pdf_for_parse(doc, zoomin, max_zoomin=6)
            except ValueError as e:
                match = re.search(r"retry with `zoomin` (\d+)", str(e))
                assert match, f"no actionable advice at zoomin {zoomin}: {e}"
                recommended = int(match.group(1))
                assert validate_pdf_for_parse(doc, recommended) == 100

    def test_advises_splitting_when_no_zoom_fits(self):
        # A single outsized MediaBox blows the per-page ceiling at every zoom,
        # so there is genuinely nothing to recommend.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(media_box="0 0 14400 14400")
        with pytest.raises(ValueError, match="split the document"):
            validate_pdf_for_parse(doc, zoomin=3, max_zoomin=6)

    def test_the_named_zoom_is_the_largest_that_fits(self):
        # Not merely *a* zoom that works -- the best one, so the caller keeps
        # as much render quality as the budget allows.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=100, media_box=A4)
        with pytest.raises(ValueError, match=r"retry with `zoomin` 3"):
            validate_pdf_for_parse(doc, zoomin=6, max_zoomin=6)
        assert validate_pdf_for_parse(doc, zoomin=3) == 100
        with pytest.raises(ValueError):
            validate_pdf_for_parse(doc, zoomin=4)


class TestLargestFittingParseZoom:
    def test_finds_the_largest_zoom_within_the_budget(self):
        # 100 A4 pages: 4-6 exceed a ceiling, 3 fits.
        sizes = [(595.0, 842.0)] * 100
        assert largest_fitting_parse_zoom(sizes, 6) == 3

    def test_every_candidate_is_tested_not_assumed_monotonic(self):
        # The per-page ceiling is still checked at the non-monotonic
        # worst-case scale, so the budget genuinely is not monotonic in
        # zoomin. A 1000x1000 pt page fits at 1, 3 and 4 but *not* at 2,
        # which escalates to 18x. A search that stopped at the first failure
        # counting down from 6 would report 4 correctly, but one that assumed
        # "lower is always safer" would wrongly offer 2.
        sizes = [(1000.0, 1000.0)]
        fits = [z for z in range(1, 7) if _parse_budget_error(sizes, z) is None]
        assert fits == [1, 3, 4], "guard: this size must be non-monotonic"
        assert largest_fitting_parse_zoom(sizes, 6) == 4
        assert largest_fitting_parse_zoom(sizes, 2) == 1

    def test_returns_the_request_itself_when_it_already_fits(self):
        sizes = [(595.0, 842.0)] * 5
        assert largest_fitting_parse_zoom(sizes, 3) == 3

    def test_returns_none_when_nothing_fits(self):
        # A single outsized MediaBox blows the per-page ceiling at every zoom.
        sizes = [(14400.0, 14400.0)]
        assert largest_fitting_parse_zoom(sizes, 6) is None


class TestParseBudgetMonotonicity:
    """A request must never be rejected in a way a lower zoom cannot fix.

    The underlying ladder is not monotonic -- that lives in deepdoc-lib --
    so the property that has to hold here is the weaker, useful one: for any
    document and any zoom, if the request is rejected then either some zoom
    is accepted and named, or no zoom works and splitting is advised.
    """

    @pytest.mark.parametrize("page_count", [1, 5, 31, 100, 200])
    def test_rejection_always_carries_a_true_way_forward(self, page_count):
        pytest.importorskip("pypdfium2")
        import re

        doc = make_pdf(page_count=page_count, media_box=A4)
        for zoomin in range(1, 7):
            try:
                validate_pdf_for_parse(doc, zoomin)
            except ValueError as e:
                message = str(e)
                match = re.search(r"retry with `zoomin` (\d+)", message)
                if match:
                    assert validate_pdf_for_parse(doc, int(match.group(1))) == (
                        page_count
                    )
                else:
                    assert "split the document" in message

    def test_a_zoom_is_only_named_when_it_genuinely_fits(self):
        # The counterpart to the above: when a page busts the per-page
        # ceiling at every zoom, no value helps. The message must not name
        # one anyway -- that is the original bug in a new costume.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(media_box="0 0 14400 14400")
        for zoomin in range(1, 7):
            with pytest.raises(ValueError) as excinfo:
                validate_pdf_for_parse(doc, zoomin, max_zoomin=6)
            assert "retry with `zoomin`" not in str(excinfo.value)
            assert "split the document" in str(excinfo.value)


class TestParseBudgetConfiguration:
    """The whole-document ceiling is deployment-tunable."""

    def test_per_page_ceiling_still_covers_the_retry_scale(self):
        # The document-wide budget no longer prices the retry, but the
        # per-page one still does: that guards a single outsized MediaBox and
        # does not depend on the `len(boxes) == 0` reasoning holding for
        # every deepdoc-lib version.
        for zoomin in range(1, 7):
            worst = worst_case_parse_zoom(zoomin)
            peak = worst_case_parse_peak_pixels(595, 842, zoomin)
            assert peak >= int(595 * worst) * int(842 * worst)

    def test_ceilings_are_configurable(self):
        # Deployments whose parse workers have headroom can raise these
        # rather than being held to a default sized for a modest worker.
        from ..pdf_ocr import _pixel_budget_from_env

        assert _pixel_budget_from_env("XINFERENCE_TEST_ABSENT_BUDGET", 7) == 7
        with mock.patch.dict(os.environ, {"XINFERENCE_TEST_BUDGET": "5000"}):
            assert _pixel_budget_from_env("XINFERENCE_TEST_BUDGET", 7) == 5000

    @pytest.mark.parametrize("bad", ["", "not-a-number", "0", "-1"])
    def test_a_malformed_ceiling_falls_back_instead_of_raising(self, bad):
        # This runs at import time; a typo in the environment must not take
        # the API process down.
        from ..pdf_ocr import _pixel_budget_from_env

        with mock.patch.dict(os.environ, {"XINFERENCE_TEST_BUDGET": bad}):
            assert _pixel_budget_from_env("XINFERENCE_TEST_BUDGET", 7) == 7

    def test_the_retry_ceiling_is_an_effective_memory_bound(self):
        # The per-page ceiling times the page ceiling is 40 G px -- ~160 GB of
        # page images. That is a mathematical bound, not a memory-safety one,
        # so the escalated total is capped separately and much lower.
        assert MAX_PDF_PARSE_PAGE_PIXELS * MAX_PDF_OCR_PAGES == 40_000_000_000
        assert MAX_PDF_PARSE_RETRY_TOTAL_PIXELS < 10_000_000_000

    def test_the_retry_ceiling_rejects_the_case_that_would_oom_a_worker(self):
        # 200 A4 pages at zoomin 3 is only 0.902 G px as requested, so the
        # main budget admits it, but it escalates to ~9 G px (~36 GB) -- past
        # what an ordinary parse worker survives.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=MAX_PDF_OCR_PAGES, media_box=A4)
        assert int(595 * 3) * int(842 * 3) * MAX_PDF_OCR_PAGES < (
            MAX_PDF_PARSE_TOTAL_PIXELS
        )
        with pytest.raises(ValueError, match="retry limit"):
            validate_pdf_for_parse(doc, zoomin=3)

    def test_the_retry_ceiling_still_admits_the_reported_document(self):
        # The ceiling must not undo the fix: the 31-page A4 document from
        # #5307 peaks at 1.4 G px even at 9x, so it clears the limit at every
        # zoomin -- rejecting it is what the previous 3 G px value was wrongly
        # believed to do.
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=31, media_box=A4)
        for zoomin in range(1, 7):
            assert validate_pdf_for_parse(doc, zoomin, max_zoomin=6) == 31

    def test_the_configured_ceiling_is_what_validation_uses(self):
        pytest.importorskip("pypdfium2")
        doc = make_pdf(page_count=100, media_box=A4)
        assert validate_pdf_for_parse(doc, zoomin=3) == 100
        with mock.patch(
            "xinference.api.pdf_ocr.MAX_PDF_PARSE_TOTAL_PIXELS", 10_000_000
        ):
            with pytest.raises(ValueError, match="whole-document limit"):
                validate_pdf_for_parse(doc, zoomin=3)


class TestWorstCaseParseZoom:
    """The retry is recursive, so a low zoomin chains up several times."""

    def test_chain_is_followed_to_the_ceiling(self):
        # zoomin 1 renders at 1, then 3, then 9 -- not just 3.
        assert worst_case_parse_zoom(1) == 9
        assert worst_case_parse_zoom(2) == 18
        assert worst_case_parse_zoom(3) == 9
        assert worst_case_parse_zoom(4) == 12
        assert worst_case_parse_zoom(5) == 15
        assert worst_case_parse_zoom(6) == 18

    def test_result_always_reaches_the_ceiling(self):
        # Whatever the input, the chain must terminate at or above the limit,
        # otherwise a further retry would still be possible.
        for zoomin in range(1, 7):
            assert worst_case_parse_zoom(zoomin) >= PDF_PARSE_RETRY_ZOOM_LIMIT

    def test_never_below_the_requested_scale(self):
        for zoomin in range(1, 7):
            assert worst_case_parse_zoom(zoomin) >= zoomin

    def test_at_or_above_the_ceiling_does_not_amplify(self):
        assert worst_case_parse_zoom(9) == 9
        assert worst_case_parse_zoom(12) == 12


class TestWorstCaseParsePeakPixels:
    def test_counts_the_coexisting_previous_render(self):
        # `page_images = [...]` builds the new list before rebinding, so the
        # render being replaced is still alive. Peak is 9x + 1x, not 9x.
        peak = worst_case_parse_peak_pixels(100, 100, 3)
        assert peak == (900 * 900) + (300 * 300)

    def test_no_previous_render_when_no_retry_can_fire(self):
        assert worst_case_parse_peak_pixels(100, 100, 9) == 900 * 900

    def test_every_accepted_zoomin_is_bounded(self):
        for zoomin in range(1, 7):
            peak = worst_case_parse_peak_pixels(595, 842, zoomin)
            final = worst_case_parse_zoom(zoomin)
            assert peak >= int(595 * final) * int(842 * final)


class TestWholeDocumentOcrTasks:
    def test_parse_is_a_whole_document_task(self):
        assert "parse" in WHOLE_DOCUMENT_OCR_TASKS

    def test_per_page_tasks_are_not(self):
        for task in ("ocr", "layout", "table"):
            assert task not in WHOLE_DOCUMENT_OCR_TASKS
