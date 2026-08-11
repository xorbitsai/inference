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

import pytest

from ..pdf_ocr import (
    DEFAULT_PDF_OCR_DPI,
    MAX_PDF_OCR_PAGES,
    WHOLE_DOCUMENT_OCR_TASKS,
    is_pdf_upload,
    merge_ocr_page_results,
    normalize_pages,
    rasterize_pdf,
    validate_pdf_for_parse,
)


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
        assert validate_pdf_for_parse(make_pdf(page_count=3)) == 3

    def test_rejects_non_pdf_bytes(self):
        pytest.importorskip("pypdfium2")
        with pytest.raises(ValueError, match="Could not read the uploaded PDF"):
            validate_pdf_for_parse(b"\x89PNG\r\n\x1a\n not a pdf")

    def test_rejects_empty_input(self):
        pytest.importorskip("pypdfium2")
        with pytest.raises(ValueError):
            validate_pdf_for_parse(b"")

    def test_rejects_too_many_pages(self):
        pytest.importorskip("pypdfium2")
        oversized = make_pdf(page_count=MAX_PDF_OCR_PAGES + 1)
        with pytest.raises(ValueError, match="at most"):
            validate_pdf_for_parse(oversized)

    def test_page_limit_is_shared_with_the_per_page_path(self):
        # Users see one page ceiling for the OCR endpoint, whichever task
        # they pick.
        pytest.importorskip("pypdfium2")
        assert validate_pdf_for_parse(make_pdf(page_count=MAX_PDF_OCR_PAGES)) == (
            MAX_PDF_OCR_PAGES
        )

    def test_rejects_an_oversized_page(self):
        # A single page with a huge (but legal) MediaBox rasterizes to
        # billions of pixels long before the page limit is relevant, so the
        # page-pixel budget has to be enforced here too.
        pytest.importorskip("pypdfium2")
        oversized = make_pdf(media_box="0 0 14400 14400")
        with pytest.raises(ValueError, match="exceeding the limit"):
            validate_pdf_for_parse(oversized)

    def test_common_page_sizes_pass_at_every_allowed_zoom(self):
        # The budget must not reject ordinary documents: A4, Letter and even
        # A0 are all legitimate parse inputs.
        pytest.importorskip("pypdfium2")
        for media_box in ("0 0 595 842", "0 0 612 792", "0 0 842 1191"):
            for zoomin in (3, 6):
                assert (
                    validate_pdf_for_parse(make_pdf(media_box=media_box), zoomin) == 1
                )

    def test_budget_scales_with_zoomin(self):
        pytest.importorskip("pypdfium2")
        # 2384x3370 pt (A0): 72 MP at zoomin 3, 289 MP at zoomin 6.
        a0 = make_pdf(media_box="0 0 2384 3370")
        assert validate_pdf_for_parse(a0, zoomin=3) == 1
        with pytest.raises(ValueError, match="exceeding the limit"):
            validate_pdf_for_parse(a0, zoomin=6)


class TestWholeDocumentOcrTasks:
    def test_parse_is_a_whole_document_task(self):
        assert "parse" in WHOLE_DOCUMENT_OCR_TASKS

    def test_per_page_tasks_are_not(self):
        for task in ("ocr", "layout", "table"):
            assert task not in WHOLE_DOCUMENT_OCR_TASKS
