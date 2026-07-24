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

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


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


class DeepDocModel(OCRModel):
    """RAGFlow's DeepDoc ONNX models for document parsing.

    Inference is provided by the ``deepdoc-lib`` package
    (https://github.com/xorbitsai/deepdoc-lib). Three tasks are supported,
    selected via the ``task`` kwarg:
    - ``ocr`` (default): text detection + recognition, returns plain text
    - ``layout``: page layout analysis, returns layout blocks as JSON
    - ``table``: table structure recognition, returns structures as JSON
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

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Run DeepDoc on one image or a list of images.

        Args:
            image: PIL Image or list of PIL Images
            **kwargs: Additional parameters including:
                - task: 'ocr' (default), 'layout' or 'table'
                - threshold: score threshold for 'table' (default 0.2). The
                  YOLOv10 layout model uses a fixed threshold in its
                  upstream postprocess, so 'layout' ignores this value.
                - return_dict: for task 'ocr', return a JSON string with boxes
                  and scores instead of plain text (default False)

        Returns:
            Plain text for task 'ocr', otherwise a JSON string. Lists of
            images return a list of texts (or a JSON array string).
        """
        logger.info("DeepDoc kwargs: %s", kwargs)

        if self._ocr is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        task = kwargs.get("task", "ocr")
        return_dict = kwargs.get("return_dict", False)

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
        return json.dumps(_jsonable(payload), ensure_ascii=False)

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
                "Supported tasks: 'ocr', 'layout', 'table'."
            )
