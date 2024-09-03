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


def _install():
    from .audio import _install as audio_install
    from .embedding import _install as embedding_install
    from .flexible import _install as flexible_install
    from .image import _install as image_install
    from .llm import _install as llm_install
    from .rerank import _install as rerank_install
    from .video import _install as video_install

    llm_install()
    audio_install()
    embedding_install()
    flexible_install()
    image_install()
    rerank_install()
    video_install()
