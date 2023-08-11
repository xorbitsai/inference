# import pytest

# from ..client import ChatModelHandle, Client, RESTfulChatModelHandle, RESTfulClient


# def test_concurrent_embedding(setup):
#     endpoint, _ = setup
#     client = Client(endpoint)
#     assert len(client.list_models()) == 0

#     model_uid = client.launch_model(
#         model_name="orca", model_size_in_billions=3, quantization="q4_0"
#     )
#     assert len(client.list_models()) == 1

#     model = client.get_model(model_uid=model_uid)
#     assert isinstance(model, ChatModelHandle)

#     completion = model.chat("write a poem.")
#     assert "content" in completion["choices"][0]["message"]

#     client.terminate_model(model_uid=model_uid)
#     assert len(client.list_models()) == 0

#     # concurrent embedding for pytorch models


#     model_uid = client.launch_model(
#         model_name="orca",
#         model_size_in_billions=3,
#         quantization="q4_0",
#     )

import threading
import time

from xinference.client import RESTfulClient


def embedding_thread(model, text):
    model.create_embedding(text)
    # print(f"Embedding: {embedding}")


def nonconcurrent_embedding(model, texts):
    for text in texts:
        model.create_embedding(text)
        # print(f"Embedding: {embedding}")


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
