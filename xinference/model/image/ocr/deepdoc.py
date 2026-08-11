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

import base64
import io
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)

# ``parse_into_bboxes`` attaches a cropped image to every element, but only
# tables and figures are worth shipping back; encoding all of them would
# inflate the response by an order of magnitude.
IMAGE_SCOPES = ("table_figure", "all", "none")
DEFAULT_IMAGE_SCOPE = "table_figure"
_IMAGE_SCOPE_TYPES = ("table", "figure")
DEFAULT_PARSE_ZOOMIN = 3
# Above this, ``parse_into_bboxes`` renders pages so large that a single
# document can exhaust host memory.
MAX_PARSE_ZOOMIN = 6
# Keys that must not appear in ``metadata``: ``position_tag`` is a
# deepdoc-internal marker string, ``image`` is replaced by ``image_base64``,
# and ``text`` is promoted to a top-level field.
_METADATA_EXCLUDED_KEYS = frozenset({"position_tag", "image", "text"})


def _jsonable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _parse_threshold(kwargs: Dict[str, Any], default: float = 0.2) -> float:
    # None (e.g. an explicit JSON null from the HTTP API) falls back to the
    # default instead of crashing float().
    value = kwargs.get("threshold")
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid threshold: {value!r}, expected a number") from e


def parse_zoomin(value: Any, default: int = DEFAULT_PARSE_ZOOMIN) -> int:
    """Validate the ``zoomin`` kwarg for task ``parse``.

    Public because the API layer validates it too: it needs the value to
    budget the raster before handing the PDF over, and both layers must
    reject exactly the same inputs.
    """
    # None (e.g. an explicit JSON null from the HTTP API) falls back to the
    # default, like `threshold` does.
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid zoomin: {value!r}, expected an integer")
    if value < 1 or value > MAX_PARSE_ZOOMIN:
        raise ValueError(
            f"Invalid zoomin: {value!r}, expected an integer between 1 "
            f"and {MAX_PARSE_ZOOMIN}"
        )
    return value


def _parse_image_scope(
    kwargs: Dict[str, Any], default: str = DEFAULT_IMAGE_SCOPE
) -> str:
    value = kwargs.get("image_scope")
    if value is None:
        return default
    if value not in IMAGE_SCOPES:
        raise ValueError(
            f"Invalid image_scope: {value!r}. "
            f"Supported values: {', '.join(repr(s) for s in IMAGE_SCOPES)}."
        )
    return value


def _wants_image(layout_type: Any, image_scope: str) -> bool:
    if image_scope == "all":
        return True
    if image_scope == "none":
        return False
    return layout_type in _IMAGE_SCOPE_TYPES


def _encode_image(image: Any) -> Optional[str]:
    """PNG-encode a crop to base64, or return None if it cannot be encoded.

    Crops are line art (tables, figures), so PNG is used rather than JPEG:
    lossy artifacts would degrade downstream re-reading of the crop.
    """
    if image is None:
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        # A crop that fails to encode must not fail the whole document.
        logger.warning("Failed to encode a DeepDoc crop image", exc_info=True)
        return None


def _build_parallel_limiter() -> Optional[List[Any]]:
    """Mirror the per-device semaphores ``RAGFlowPdfParser.__init__`` builds.

    Bypassing that constructor to reuse the loaded recognizers means this has
    to be reproduced, otherwise multi-GPU workers would silently serialize
    page recognition onto a single device.
    """
    import asyncio

    from deepdoc.common import check_and_install_torch, settings

    try:
        check_and_install_torch()
        if settings.PARALLEL_DEVICES > 1:
            return [asyncio.Semaphore(1) for _ in range(settings.PARALLEL_DEVICES)]
    except Exception:
        logger.debug("Could not size the DeepDoc parallel limiter", exc_info=True)
    return None


_ZOOM_RETRY_DISABLED = "_xinference_no_zoom_retry"


def _disable_zoom_retry(parser: Any) -> None:
    """Stop ``__images__`` from re-rendering the document at three times the zoom.

    When a first pass finds no boxes at all -- a blank or image-only PDF --
    deepdoc recurses into ``__images__`` at ``zoomin * 3``, i.e. nine times
    the pixels, and does so for a document that by definition produced no
    output. It is also inconsistent: ``parse_into_bboxes`` runs every later
    stage at the *original* zoom, so the pages it then works on no longer
    match the scale the coordinates are computed against.

    Wrapping the bound method to swallow the recursive call keeps one render
    per request, which is what the API layer's page-size budget assumes.
    """
    if getattr(parser, _ZOOM_RETRY_DISABLED, False):
        return
    original = getattr(parser, "__images__", None)
    if original is None:
        # A parser shape we do not recognise; leave it alone rather than
        # fail the request over a defensive guard.
        logger.debug("Parser has no __images__ to guard against zoom retries")
        return

    def __images__(fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        # The recursion is the last statement of `__images__`, so a guard
        # that makes the inner call a no-op leaves the first render intact.
        if getattr(parser, "_xinference_in_images", False):
            return None
        parser._xinference_in_images = True
        try:
            return original(fnm, zoomin, page_from, page_to, callback)
        finally:
            parser._xinference_in_images = False

    parser.__images__ = __images__
    setattr(parser, _ZOOM_RETRY_DISABLED, True)


def _reset_parser_document_state(parser: Any) -> None:
    """Drop the per-document state a parse run leaves on the parser.

    The parser is cached and reused across requests to keep a single set of
    ONNX models loaded, but ``parse_into_bboxes`` stores the document it just
    processed on the instance and never clears it. Two problems follow:

    * the rasterized pages, the per-box crops and the extracted characters
      stay reachable after the response is sent -- at the page limit and the
      largest render scale that is gigabytes of resident memory held idle;
    * ``__images__`` resets ``boxes`` before it loads the new document but
      swallows load failures, so a document it cannot open would leave the
      *previous* request's pages attached and silently parse those instead.

    Clearing the state after every run bounds the memory and makes a failed
    load produce an empty result rather than the last document's content.
    """
    for attribute, empty in (
        ("boxes", []),
        ("page_images", []),
        ("page_chars", []),
        ("page_layout", []),
        ("page_cum_height", [0]),
        ("garbages", {}),
        ("lefted_chars", []),
        ("mean_height", []),
        ("mean_width", []),
        ("outlines", []),
        ("tb_cpns", []),
        ("pdf", None),
        ("total_page", 0),
    ):
        try:
            setattr(parser, attribute, empty)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not reset parser attribute %s", attribute)


def _element_to_json(element: Dict[str, Any], image_scope: str) -> Dict[str, Any]:
    """Turn one ``parse_into_bboxes`` element into a JSON-serializable dict.

    ``parse_into_bboxes`` returns numpy scalars and PIL images, and the
    table/figure elements it re-inserts into the text flow carry a smaller
    key set than the text ones (no ``col_id``, no ``position_tag``), so
    every key is read defensively. Absent keys are omitted rather than
    defaulted, so the response never claims information deepdoc did not
    produce.
    """
    metadata = {
        k: _jsonable(v) for k, v in element.items() if k not in _METADATA_EXCLUDED_KEYS
    }
    result: Dict[str, Any] = {
        "type": _jsonable(element.get("layout_type")),
        "text": element.get("text"),
        "metadata": metadata,
    }
    if _wants_image(element.get("layout_type"), image_scope):
        encoded = _encode_image(element.get("image"))
        if encoded is not None:
            result["image_base64"] = encoded
    return result


class DeepDocModel(OCRModel):
    """RAGFlow's DeepDoc ONNX models for document parsing.

    Inference is provided by the ``deepdoc-lib`` package
    (https://github.com/xorbitsai/deepdoc-lib). Four tasks are supported,
    selected via the ``task`` kwarg:
    - ``ocr`` (default): text detection + recognition, returns plain text
    - ``layout``: page layout analysis, returns layout blocks as a dict
    - ``table``: table structure recognition, returns structures as a dict
    - ``parse``: the full document-parsing pipeline over a whole PDF,
      returning ordered elements with table HTML, figures and cross-page
      coordinates. Unlike the other three it consumes PDF bytes rather
      than a page image.
    """

    required_libs = ("deepdoc",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "DeepDoc"

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # model info when loading
        self._ocr = None
        self._layout_recognizer = None
        self._table_recognizer = None
        self._pdf_parser = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def _model_dir(self) -> str:
        # HuggingFace InfiniFlow/deepdoc lays the onnx files flat, while
        # ModelScope Xorbits/deepdoc keeps them under a vision/ subdirectory.
        model_dir = self._model_path or ""
        if not os.path.exists(os.path.join(model_dir, "det.onnx")):
            candidate = os.path.join(model_dir, "vision")
            if os.path.exists(os.path.join(candidate, "det.onnx")):
                model_dir = candidate
        return model_dir

    def load(self):
        from deepdoc.vision import OCR

        logger.info(f"Loading DeepDoc models from {self._model_path}")
        # Text detection/recognition is the default task; layout and table
        # recognizers are loaded lazily on first use.
        self._ocr = OCR(model_dir=self._model_dir())

    def _get_layout_recognizer(self):
        if self._layout_recognizer is None:
            from deepdoc.vision import LayoutRecognizer

            self._layout_recognizer = LayoutRecognizer(
                "layout", model_dir=self._model_dir()
            )
        return self._layout_recognizer

    def _get_table_recognizer(self):
        if self._table_recognizer is None:
            from deepdoc.vision import TableStructureRecognizer

            self._table_recognizer = TableStructureRecognizer(
                model_dir=self._model_dir()
            )
        return self._table_recognizer

    def _xgb_model_dir(self) -> str:
        """Locate the directory holding ``updown_concat_xgb.model``.

        The paragraph-merging booster lives outside the vision bundle: the
        ModelScope layout keeps ``vision/`` and ``xgb/`` as siblings, while a
        flat checkout keeps the booster next to the onnx files.
        """
        model_name = "updown_concat_xgb.model"
        vision_dir = self._model_dir()
        candidates = [
            os.path.join(os.path.dirname(vision_dir), "xgb"),
            os.path.join(vision_dir, "xgb"),
            vision_dir,
            self._model_path or "",
        ]
        for candidate in candidates:
            if candidate and os.path.exists(os.path.join(candidate, model_name)):
                return candidate
        raise RuntimeError(
            f"Could not find {model_name} for DeepDoc task 'parse' under "
            f"{self._model_path!r}. Re-download the model files."
        )

    def _build_pdf_parser(self):
        """Build a ``PdfParser`` that reuses the already-loaded recognizers.

        ``RAGFlowPdfParser.__init__`` would construct its own OCR, layout and
        table recognizers, doubling the ONNX sessions (and the VRAM) this
        model already holds. So the constructor is bypassed and only the
        attributes it would have set are assembled, wiring in the existing
        recognizers. If a future ``deepdoc-lib`` needs attributes we do not
        know about, fall back to the real constructor and inject afterwards
        -- slower and briefly twice the memory, but correct.
        """
        import xgboost as xgb
        from deepdoc import PdfModelConfig, PdfParser, TokenizerConfig
        from deepdoc.depend.rag_tokenizer import RagTokenizer

        vision_dir = self._model_dir()
        xgb_dir = self._xgb_model_dir()
        # `model_provider="local"` keeps the onnx/booster side offline, but the
        # tokenizer config must stay online-capable: RagTokenizer raises when
        # its nltk data is missing and offline mode is on.
        model_cfg = PdfModelConfig(
            vision_model_dir=vision_dir,
            xgb_model_dir=xgb_dir,
            model_provider="local",
        )
        tokenizer_cfg = TokenizerConfig()

        booster = xgb.Booster()
        # Upstream keys this purely off CUDA availability, which already
        # honours the CUDA_VISIBLE_DEVICES that `gpu_idx` sets. An explicit
        # non-CUDA `device` is respected on top of that, so a model pinned to
        # CPU does not pull the booster onto a GPU.
        if not self._device or "cuda" in self._device:
            try:
                import torch

                if torch.cuda.is_available():
                    booster.set_param({"device": "cuda"})
            except Exception:
                logger.debug("torch unavailable, running the xgb booster on CPU")
        booster.load_model(os.path.join(xgb_dir, "updown_concat_xgb.model"))

        parser = PdfParser.__new__(PdfParser)
        parser.model_cfg = model_cfg
        parser.tokenizer_cfg = tokenizer_cfg
        # Load-bearing: `_concat_downward` tokenizes and tags line text.
        parser.tokenizer = RagTokenizer(
            dict_prefix=tokenizer_cfg.resolve_dict_prefix(),
            offline=tokenizer_cfg.offline,
            nltk_data_dir=tokenizer_cfg.nltk_data_dir,
        )
        parser.ocr = self._ocr
        parser.layouter = self._get_layout_recognizer()
        parser.tbl_det = self._get_table_recognizer()
        parser.updown_cnt_mdl = booster
        parser.parallel_limiter = _build_parallel_limiter()
        parser.page_from = 0
        parser.column_num = 1
        return parser

    def _get_pdf_parser(self):
        if self._pdf_parser is None:
            try:
                self._pdf_parser = self._build_pdf_parser()
            except Exception:
                logger.warning(
                    "Could not assemble a DeepDoc PdfParser around the loaded "
                    "recognizers; falling back to constructing a new one, "
                    "which loads a second set of models.",
                    exc_info=True,
                )
                self._pdf_parser = self._build_pdf_parser_fallback()
        return self._pdf_parser

    def _build_pdf_parser_fallback(self):
        from deepdoc import PdfModelConfig, PdfParser

        parser = PdfParser(
            model_cfg=PdfModelConfig(
                vision_model_dir=self._model_dir(),
                xgb_model_dir=self._xgb_model_dir(),
                model_provider="local",
            )
        )
        # Drop the freshly built recognizers in favour of the loaded ones so
        # steady-state memory still holds a single set.
        parser.ocr = self._ocr
        parser.layouter = self._get_layout_recognizer()
        parser.tbl_det = self._get_table_recognizer()
        return parser

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], bytes],
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run DeepDoc on one image, a list of images, or a whole PDF.

        Args:
            image: PIL Image or list of PIL Images. For task 'parse' this is
                instead the raw bytes of a PDF document.
            **kwargs: Additional parameters including:
                - task: 'ocr' (default), 'layout', 'table' or 'parse'
                - threshold: score threshold for 'table' (default 0.2). The
                  YOLOv10 layout model uses a fixed threshold in its
                  upstream postprocess, so 'layout' ignores this value.
                - return_dict: for task 'ocr', return a dict with boxes
                  and scores instead of plain text (default False)
                - zoomin: render scale for 'parse' (default 3)
                - image_scope: which 'parse' elements carry a base64 crop,
                  'table_figure' (default), 'all' or 'none'

        Returns:
            Plain text for task 'ocr', otherwise a JSON-serializable dict.
            Lists of images return a list of texts (or a list of dicts).
            The REST layer serializes the return value to JSON once, so
            structured results must be returned as objects, not pre-encoded
            JSON strings.
        """
        logger.info("DeepDoc kwargs: %s", kwargs)

        if self._ocr is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        task = kwargs.get("task", "ocr")
        return_dict = kwargs.get("return_dict", False)

        # `parse` runs the whole-document pipeline, which renders the PDF
        # itself and merges across pages, so it takes the original bytes
        # rather than a rasterized page and must branch before any
        # image normalization below.
        if task == "parse":
            return self._process_parse(image, kwargs)

        if image is None:
            raise ValueError("Input image cannot be None.")
        single = isinstance(image, PIL.Image.Image)
        images = [image] if single else list(image)
        if any(img is None for img in images):
            raise ValueError("Input image list cannot contain None.")
        results = [self._process_single(img, task, kwargs) for img in images]

        if task == "ocr" and not return_dict:
            texts = [
                "\n".join(line["text"] for line in res["lines"]) for res in results
            ]
            return texts[0] if single else texts

        payload = results[0] if single else results
        return _jsonable(payload)

    def _process_single(
        self, image: PIL.Image.Image, task: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        import numpy as np

        if image.mode != "RGB":
            image = image.convert("RGB")
        img = np.array(image)

        if task == "ocr":
            assert self._ocr is not None
            res = self._ocr(img)
            # OCR.__call__ returns a (None, None, timing) tuple when no text
            # box is detected, and a list of (box, (text, score)) otherwise.
            if not isinstance(res, list):
                res = []
            lines = [
                {"box": box, "text": text, "score": float(score)}
                for box, (text, score) in res
            ]
            return {"task": task, "lines": lines}
        elif task == "layout":
            recognizer = self._get_layout_recognizer()
            threshold = _parse_threshold(kwargs)
            layouts = recognizer.forward([img], thr=threshold)
            return {"task": task, "layouts": layouts[0] if layouts else []}
        elif task == "table":
            recognizer = self._get_table_recognizer()
            threshold = _parse_threshold(kwargs)
            structures = recognizer([img], thr=threshold)
            return {"task": task, "structures": structures[0] if structures else []}
        else:
            raise ValueError(
                f"Unsupported task for DeepDoc: {task}. "
                "Supported tasks: 'ocr', 'layout', 'table', 'parse'."
            )

    def _process_parse(self, data: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full document-parsing pipeline over a PDF.

        ``parse_into_bboxes`` does its own rendering, layout and table
        recognition, paragraph merging and reading-order reconstruction, and
        needs the whole document to accumulate cross-page coordinates -- so
        unlike the other tasks it consumes PDF bytes, not a page image.
        """
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise ValueError(
                "DeepDoc task 'parse' requires the raw bytes of a PDF "
                f"document, got {type(data).__name__}. Upload a PDF to the "
                "OCR endpoint instead of an image."
            )

        zoomin = parse_zoomin(kwargs.get("zoomin"))
        image_scope = _parse_image_scope(kwargs)

        parser = self._get_pdf_parser()
        _disable_zoom_retry(parser)
        try:
            # `__images__` accepts bytes directly (it wraps them in a BytesIO),
            # so no temporary file is needed.
            elements = parser.parse_into_bboxes(data, zoomin=zoomin)
            return {
                "task": "parse",
                "elements": [_element_to_json(e, image_scope) for e in elements or []],
            }
        finally:
            _reset_parser_document_state(parser)
