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

lock = threading.Lock()
concurrent_results = {}
nonconcurrent_results = {}


def embedding_thread(model, text):
    global concurrent_results
    embedding = model.create_embedding(text)
    with lock:
        concurrent_results[text] = embedding


def nonconcurrent_embedding(model, texts):
    global nonconcurrent_results
    for text in texts:
        embedding = model.create_embedding(text)
        nonconcurrent_results[text] = embedding


def main():
    client = RESTfulClient("http://127.0.0.1:60282")
    model_uid = client.launch_model(
        model_name="opt",
        model_size_in_billions=1,
        model_format="pytorch",
        quantization="8-bit",
    )
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

    print("Comparing embeddings...")

    for text in texts:
        if concurrent_results[text] == nonconcurrent_results[text]:
            print(f"Embedding for '{text}' matches.")
        else:
            print(f"Embedding for '{text}' does not match.")


if __name__ == "__main__":
    main()
