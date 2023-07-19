import gradio as gr
from xinference.client import Client
from typing import List, Dict

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
             instructions to run:
                 1. Install Xinference and Llama-cpp-python
                 2. Run 'xinference --host "localhost" --port 9997' in terminal
                 3. Run this python file in new terminal window
                 
                 e.g. (feel free to copy)
                 python gradio_chatinterface.py \\
                 --endpoint http://localhost:9997 \\
                 --model_name vicuna-v1.3 \\
                 --model_size_in_billions 7 \\
                 --model_format ggmlv3 \\
                 --quantization q2_K
                 
                 If you decide to change the port number in step 2,
                 please also change the endpoint in the arguments
             ''')
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Xinference endpoint, required"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model, required"
    )
    parser.add_argument(
        "--model_size_in_billions",
        type=int,
        required=False,
        help="Size of the model in billions",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        required=False,
        help="Format of the model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=False,
        help="Quantization of the model"
    )

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
    model_uid = client.launch_model(
        model_name,
        model_size_in_billions=model_size_in_billions,
        model_format=model_format,
        quantization=quantization,
        n_ctx=2048,
    )
    model = client.get_model(model_uid)

    def flatten(matrix: List[List[str]]) -> List[str]:
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list


    def to_chat(lst: List[str]) -> List[Dict[str, str]]:
        res = []
        for i in range(len(lst)):
            role = "assistant" if i % 2 == 1 else "user"
            res.append(
                {
                    "role": role,
                    "content": lst[i],
                }
            )
        print(res)
        return res


    def generate_wrapper(message: str, history: List[List[str]]) -> str:
        output = model.chat(
            prompt=message,
            chat_history=to_chat(flatten(history)),
            generate_config={'max_tokens': 512, 'stream': False}
        )
        return output["choices"][0]["message"]["content"]


    demo = gr.ChatInterface(
        fn=generate_wrapper,
        examples=[
            "Show me a two sentence horror story with a plot twist",
            "Generate a Haiku poem using trignometry as the central theme",
            "Write three sentences of scholarly description regarding a supernatural beast",
            "Prove there does not exist a largest integer"
        ],
        title="Xinference Chat Bot"
    )
    demo.launch()
