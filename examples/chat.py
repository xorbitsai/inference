from typing import List

from xinference.client import Client
from xinference.types import ChatCompletionMessage

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--endpoint", type=str, help="Xinference endpoint, required")
    parser.add_argument("--model_name", type=str, help="Name of the model, required")
    parser.add_argument(
        "--model_size_in_billions", type=int, required=False, help="Size of the model in billions", )
    parser.add_argument("--model_format", type=str, required=False, help="Format of the model", )
    parser.add_argument("--quantization", type=str, required=False, help="Quantization")

    args = parser.parse_args()

    endpoint = args.endpoint
    model_name = args.model_name
    model_size_in_billions = args.model_size_in_billions
    model_format = args.model_format
    quantization = args.quantization

    print(f"Xinference endpoint: {endpoint}")
    print(f"Model Name: {model_name}")
    print(f"Model Size (in billions): {model_size_in_billions}")
    print(f"Model Format: {model_format}")
    print(f"Quantization: {quantization}")

    client = Client(endpoint)
    model_uid = client.launch_model(model_name, n_ctx=2048)
    model = client.get_model(model_uid)

    chat_history: List["ChatCompletionMessage"] = []
    while True:
        prompt = input("you: ")
        completion = model.chat(
            prompt=prompt,
            chat_history=chat_history,
            generate_config={"max_tokens": 1024}
        )
        content = completion["choices"][0]["message"]["content"]
        print(f"{model_name}: {content}")
        chat_history.append(
            ChatCompletionMessage(role="user", content=prompt)
        )
        chat_history.append(
            ChatCompletionMessage(role="assistant", content=content)
        )
