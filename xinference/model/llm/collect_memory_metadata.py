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

"""Extract memory estimation metadata from a local model config.json."""

import argparse
import json

from .memory_metadata import ModelMemoryMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="Local config.json from the exact model revision"
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        metadata = ModelMemoryMetadata.from_config(json.load(f))
    print(metadata.json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
