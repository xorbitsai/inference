# Copyright 2022-2023 XProbe Inc.
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

import argparse
import logging
import os.path

from xinference.client import Client

logger = logging.getLogger(__name__)


def _prompt(text):
    return f"Translate the english text to chinese: {text}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="Xinference endpoint, required",
        required=True,
    )
    parser.add_argument("-i", "--input", type=str, help="Input text", required=True)

    args = parser.parse_args()
    endpoint = args.endpoint
    logger.info("Connect to xinference server: %s", endpoint)
    client = Client(endpoint)

    logger.info("Launch model.")
    model_uid = client.launch_model(
        model_name="OpenBuddy",
        model_format="ggmlv3",
        model_size_in_billions=13,
        quantization="Q4_1",
        n_ctx=2048,
    )
    translator_model = client.get_model(model_uid)

    logger.info("Read %s", args.input)
    with open(args.input, "r") as f:
        eng = f.read()

    paragraphs = eng.split("\n\n")
    logger.info("%s contains %s lines.", args.input, len(paragraphs))
    input, ext = os.path.splitext(args.input)
    output = f"{input}_translated{ext}"
    logger.info("Translated output: %s", output)
    with open(output, "w") as f:
        for idx, text_string in enumerate(paragraphs, 1):
            logger.info(
                "[%s/%s] Translate: %.10s...", idx, len(paragraphs), text_string
            )
            completion = translator_model.chat(
                _prompt(text_string), generate_config={"temperature": 0.23}
            )
            content = completion["choices"][0]["message"]["content"]
            stripped_content = content.split("\n")[0]
            logger.info("%s", stripped_content)
            f.write(stripped_content + "\n")
