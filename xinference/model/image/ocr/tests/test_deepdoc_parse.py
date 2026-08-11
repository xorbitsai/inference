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
Unit tests for DeepDoc's ``task="parse"`` serialization and kwargs.

These cover the pure parts of the parse task — element serialization and
kwarg validation — without loading the ONNX models, so they run on CPU
without the ``deepdoc`` package installed. The recognizer reuse and the
output parity against a local ``parse_into_bboxes`` run need real models
and are verified out of band.

The element dicts used here mirror what ``parse_into_bboxes`` really
returns, including the detail that the table/figure elements it
re-inserts into the text flow carry no ``col_id`` and no
``position_tag``.
"""

import base64
import io

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")

import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402

from ..deepdoc import (  # noqa: E402
    DEFAULT_IMAGE_SCOPE,
    DEFAULT_PARSE_ZOOMIN,
    IMAGE_SCOPES,
    MAX_PARSE_ZOOMIN,
    DeepDocModel,
    _element_to_json,
    _parse_image_scope,
    parse_zoomin,
)


def make_image(color: str = "red") -> PIL.Image.Image:
    return PIL.Image.new("RGB", (4, 4), color=color)


def text_element(**overrides):
    """An element shaped like a text/title box from ``parse_into_bboxes``."""
    element = {
        "page_number": 1,
        "x0": 71.66,
        "x1": 240.25,
        "top": 124.1,
        "bottom": 301.6,
        "layout_type": "text",
        "layoutno": "text-0",
        "col_id": 0,
        "positions": [[1, 71, 240, 124, 301]],
        "text": "hello",
        "position_tag": "@@1\t71\t240\t124\t301##",
        "image": make_image(),
    }
    element.update(overrides)
    return element


def table_element(**overrides):
    """A re-inserted table element: no ``col_id``, no ``position_tag``."""
    element = {
        "page_number": 2,
        "x0": 20.0,
        "x1": 400.0,
        "top": 50.0,
        "bottom": 200.0,
        "layout_type": "table",
        "positions": [[2, 20, 400, 50, 200]],
        "text": "<table><caption>c</caption><tr><th>h</th></tr></table>",
        "image": make_image("blue"),
    }
    element.update(overrides)
    return element


class TestElementToJson:
    def test_shape(self):
        result = _element_to_json(text_element(), "none")
        assert set(result) == {"type", "text", "metadata"}
        assert result["type"] == "text"
        assert result["text"] == "hello"
        assert result["metadata"]["page_number"] == 1
        assert result["metadata"]["layout_type"] == "text"
        assert result["metadata"]["positions"] == [[1, 71, 240, 124, 301]]

    def test_drops_internal_keys(self):
        # `position_tag` is a deepdoc-internal marker and `image` is
        # superseded by `image_base64`; neither may reach the client.
        result = _element_to_json(text_element(), "all")
        assert "position_tag" not in result["metadata"]
        assert "image" not in result["metadata"]
        # `text` is promoted to the top level rather than duplicated.
        assert "text" not in result["metadata"]

    def test_table_without_col_id_or_position_tag(self):
        # The re-inserted table/figure elements lack these keys entirely.
        # Serialization must neither crash nor invent a value for them.
        result = _element_to_json(table_element(), "none")
        assert result["type"] == "table"
        assert result["text"].startswith("<table>")
        assert "col_id" not in result["metadata"]
        assert "position_tag" not in result["metadata"]

    def test_col_id_preserved_when_present(self):
        result = _element_to_json(text_element(col_id=3), "none")
        assert result["metadata"]["col_id"] == 3

    def test_numpy_scalars_coerced(self):
        result = _element_to_json(
            text_element(
                x0=np.float32(1.5),
                page_number=np.int64(2),
                positions=np.array([[1, 2, 3, 4, 5]]),
            ),
            "none",
        )
        metadata = result["metadata"]
        assert isinstance(metadata["x0"], float) and metadata["x0"] == 1.5
        assert isinstance(metadata["page_number"], int)
        assert metadata["positions"] == [[1, 2, 3, 4, 5]]
        for value in metadata.values():
            assert not isinstance(value, (np.generic, np.ndarray))

    def test_unknown_extra_keys_pass_through(self):
        result = _element_to_json(text_element(some_new_key="v"), "none")
        assert result["metadata"]["some_new_key"] == "v"


class TestImageScope:
    def test_table_figure_encodes_only_table_and_figure(self):
        assert "image_base64" not in _element_to_json(text_element(), "table_figure")
        assert "image_base64" in _element_to_json(table_element(), "table_figure")
        figure = table_element(layout_type="figure")
        assert "image_base64" in _element_to_json(figure, "table_figure")

    def test_all_encodes_every_element(self):
        assert "image_base64" in _element_to_json(text_element(), "all")
        assert "image_base64" in _element_to_json(table_element(), "all")

    def test_none_encodes_nothing(self):
        assert "image_base64" not in _element_to_json(text_element(), "none")
        assert "image_base64" not in _element_to_json(table_element(), "none")

    def test_encoded_payload_is_a_png(self):
        result = _element_to_json(table_element(), "table_figure")
        raw = base64.b64decode(result["image_base64"])
        assert PIL.Image.open(io.BytesIO(raw)).format == "PNG"

    def test_missing_image_omits_the_field(self):
        # The field is omitted rather than set to null, so clients can rely
        # on its presence meaning "there is a crop".
        result = _element_to_json(table_element(image=None), "table_figure")
        assert "image_base64" not in result

    def test_unencodable_image_does_not_fail_the_document(self):
        class Broken:
            def save(self, *args, **kwargs):
                raise OSError("cannot encode")

        result = _element_to_json(table_element(image=Broken()), "table_figure")
        assert "image_base64" not in result
        assert result["text"].startswith("<table>")


class TestParseZoomin:
    def test_default(self):
        assert parse_zoomin(None) == DEFAULT_PARSE_ZOOMIN

    def test_none_falls_back_to_default(self):
        # An explicit JSON null from the HTTP API must not crash.
        assert parse_zoomin(None) == DEFAULT_PARSE_ZOOMIN

    def test_valid_value(self):
        assert parse_zoomin(1) == 1
        assert parse_zoomin(MAX_PARSE_ZOOMIN) == MAX_PARSE_ZOOMIN

    @pytest.mark.parametrize(
        "value", [0, -1, MAX_PARSE_ZOOMIN + 1, "3", 3.5, True, [3]]
    )
    def test_invalid_values_rejected(self, value):
        with pytest.raises(ValueError, match="zoomin"):
            parse_zoomin(value)


class TestParseImageScope:
    def test_default(self):
        assert _parse_image_scope({}) == DEFAULT_IMAGE_SCOPE

    def test_none_falls_back_to_default(self):
        assert _parse_image_scope({"image_scope": None}) == DEFAULT_IMAGE_SCOPE

    @pytest.mark.parametrize("scope", IMAGE_SCOPES)
    def test_supported_scopes(self, scope):
        assert _parse_image_scope({"image_scope": scope}) == scope

    def test_unknown_scope_lists_the_supported_ones(self):
        with pytest.raises(ValueError) as excinfo:
            _parse_image_scope({"image_scope": "tables"})
        message = str(excinfo.value)
        for scope in IMAGE_SCOPES:
            assert scope in message


class FakePdfParser:
    """Stands in for ``RAGFlowPdfParser``, recording how it was called.

    The per-document attributes mirror the state a real parse run leaves
    behind on the instance.
    """

    def __init__(self, elements):
        self._elements = elements
        self.calls = []
        self.boxes = []
        self.page_images = []
        self.page_chars = []
        self.pdf = None

    def parse_into_bboxes(self, fnm, zoomin=3):
        self.calls.append((fnm, zoomin))
        # A real run leaves the document it just processed on the instance.
        self.boxes = list(self._elements or [])
        self.page_images = [make_image(), make_image()]
        self.page_chars = [[{"text": "a"}]]
        self.pdf = object()
        return self._elements


class FakeLayoutRecognizer:
    def forward(self, images, thr=0.2):
        return [[{"type": "title"}]]


class FakeTableRecognizer:
    def __call__(self, images, thr=0.2):
        return [[{"label": "table"}]]


def make_model(parser=None):
    """A ``DeepDocModel`` with the ONNX loading side-stepped."""
    model = DeepDocModel.__new__(DeepDocModel)
    model._ocr = object()
    model._layout_recognizer = None
    model._table_recognizer = None
    model._pdf_parser = parser
    return model


class TestParseTaskRouting:
    def test_parse_returns_serialized_elements(self):
        parser = FakePdfParser([text_element(), table_element()])
        model = make_model(parser)
        result = model.ocr(b"%PDF-1.4 ...", task="parse")
        assert result["task"] == "parse"
        assert [e["type"] for e in result["elements"]] == ["text", "table"]
        # Bytes reach the parser unchanged: the whole document is needed for
        # cross-page merging, so nothing rasterizes or splits it per page.
        assert parser.calls == [(b"%PDF-1.4 ...", DEFAULT_PARSE_ZOOMIN)]

    def test_zoomin_and_image_scope_forwarded(self):
        parser = FakePdfParser([table_element()])
        model = make_model(parser)
        result = model.ocr(b"%PDF", task="parse", zoomin=2, image_scope="none")
        assert parser.calls == [(b"%PDF", 2)]
        assert "image_base64" not in result["elements"][0]

    @pytest.mark.parametrize("payload", [bytearray(b"%PDF"), memoryview(b"%PDF")])
    def test_bytes_like_payloads_accepted(self, payload):
        parser = FakePdfParser([])
        model = make_model(parser)
        assert model.ocr(payload, task="parse") == {"task": "parse", "elements": []}
        assert parser.calls == [(b"%PDF", DEFAULT_PARSE_ZOOMIN)]

    def test_empty_result_is_an_empty_element_list(self):
        model = make_model(FakePdfParser(None))
        assert model.ocr(b"%PDF", task="parse") == {"task": "parse", "elements": []}

    @pytest.mark.parametrize("payload", [make_image(), None, "/tmp/doc.pdf"])
    def test_non_bytes_payload_rejected(self, payload):
        model = make_model(FakePdfParser([]))
        with pytest.raises(ValueError, match="requires the raw bytes of a PDF"):
            model.ocr(payload, task="parse")

    def test_invalid_kwargs_reported_before_parsing(self):
        parser = FakePdfParser([])
        model = make_model(parser)
        with pytest.raises(ValueError, match="image_scope"):
            model.ocr(b"%PDF", task="parse", image_scope="bogus")
        assert parser.calls == []

    def test_unloaded_model_still_raises(self):
        model = make_model(FakePdfParser([]))
        model._ocr = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.ocr(b"%PDF", task="parse")

    def test_other_tasks_never_reach_the_pdf_parser(self):
        # The three per-page tasks go through the real `_process_single`
        # dispatch with stub recognizers, so the parse branch cannot
        # accidentally capture them.
        parser = FakePdfParser([])
        model = make_model(parser)
        model._ocr = lambda img: [([0, 0, 1, 1], ("hi", 0.9))]
        model._layout_recognizer = FakeLayoutRecognizer()
        model._table_recognizer = FakeTableRecognizer()

        assert model.ocr(make_image(), task="ocr") == "hi"
        assert model.ocr(make_image(), task="layout")["layouts"] == [{"type": "title"}]
        assert model.ocr(make_image(), task="table")["structures"] == [
            {"label": "table"}
        ]
        assert parser.calls == []

    def test_unsupported_task_lists_parse(self):
        model = make_model(FakePdfParser([]))
        with pytest.raises(ValueError, match="'parse'"):
            model.ocr(make_image(), task="nope")


class TestParserStateIsReleased:
    """The parser is cached to keep one set of ONNX models loaded, so the
    document it just processed must not stay attached to it."""

    def test_document_state_cleared_after_a_successful_parse(self):
        parser = FakePdfParser([text_element(), table_element()])
        model = make_model(parser)
        result = model.ocr(b"%PDF", task="parse")
        assert len(result["elements"]) == 2
        # Rasterized pages and crops would otherwise stay resident until the
        # next request replaced them.
        assert parser.page_images == []
        assert parser.page_chars == []
        assert parser.boxes == []
        assert parser.pdf is None

    def test_document_state_cleared_when_parsing_raises(self):
        class Failing(FakePdfParser):
            def parse_into_bboxes(self, fnm, zoomin=3):
                super().parse_into_bboxes(fnm, zoomin)
                raise RuntimeError("boom")

        parser = Failing([text_element()])
        model = make_model(parser)
        with pytest.raises(RuntimeError, match="boom"):
            model.ocr(b"%PDF", task="parse")
        assert parser.page_images == []
        assert parser.boxes == []

    def test_zoom_retry_is_suppressed(self):
        # deepdoc re-renders at 3x the zoom (9x the pixels) when a first pass
        # finds no boxes, which the API layer's page-size budget does not
        # account for -- and which leaves later stages working at a zoom that
        # no longer matches the pages.
        renders = []

        class Retrying(FakePdfParser):
            def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, cb=None):
                renders.append(zoomin)
                # what deepdoc does when it finds nothing
                if zoomin < 9:
                    self.__images__(fnm, zoomin * 3, page_from, page_to, cb)

            def parse_into_bboxes(self, fnm, zoomin=3):
                self.calls.append((fnm, zoomin))
                self.__images__(fnm, zoomin)
                return self._elements

        parser = Retrying([])
        model = make_model(parser)
        model.ocr(b"%PDF", task="parse")
        assert renders == [3]

    def test_parser_without_images_still_parses(self):
        # The guard must not fail a request on a parser shape it does not
        # recognise.
        parser = FakePdfParser([text_element()])
        model = make_model(parser)
        assert len(model.ocr(b"%PDF", task="parse")["elements"]) == 1

    def test_stale_pages_cannot_leak_into_a_later_request(self):
        # deepdoc swallows document-load failures, so without the reset a
        # PDF it cannot open would be parsed against the previous request's
        # pages. After the reset such a run sees an empty document.
        parser = FakePdfParser([text_element()])
        model = make_model(parser)
        model.ocr(b"%PDF first", task="parse")
        assert parser.page_images == []
        assert parser.page_chars == []
