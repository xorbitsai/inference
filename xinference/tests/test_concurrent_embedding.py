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


"""
Simple test for multithreaded embedding creation
"""

import threading
import time

from xinference.client import RESTfulClient


def embedding_thread(model, text):
    model.create_embedding(text)


def nonconcurrent_embedding(model, texts):
    for text in texts:
        model.create_embedding(text)


def main():
    client = RESTfulClient("http://127.0.0.1:35819")
    model_uid = client.launch_model(model_name="orca", quantization="q4_0")
    model = client.get_model(model_uid)

    texts = ["Once upon a time", "Hello, world!", "Hi"]

    start_time = time.time()

    threads = []
    for text in texts:
        thread = threading.Thread(target=embedding_thread, args=(model, text))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    print(f"Concurrent Time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    nonconcurrent_embedding(model, texts)
    end_time = time.time()
    print(f"Nonconcurrent Time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
