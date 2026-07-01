# Copyright 2022-2026 XProbe Inc.
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

import os


def test_megatts(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="MegaTTS3",
        model_type="audio",
        compile=False,
        download_hub="huggingface",
    )
    model = client.get_model(model_uid)

    # Test copy voice
    prompt_speech_path = os.path.join(os.path.dirname(__file__), "bbc_news.wav")
    with open(prompt_speech_path, "rb") as f:
        prompt_speech = f.read()
    prompt_latent_path = os.path.join(os.path.dirname(__file__), "bbc_news.npy")
    with open(prompt_latent_path, "rb") as f:
        prompt_latent = f.read()
    response = model.speech(
        "His death in this conjuncture was a public misfortune.",
        prompt_speech=prompt_speech,
        prompt_latent=prompt_latent,
    )
    assert type(response) is bytes
    assert len(response) > 0
