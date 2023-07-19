import gradio as gr
from xinference.client import Client
from typing import Any, List, Dict

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
             Instructions to run:
                 1. Install Xinference and Llama-cpp-python
                 2. Run 'xinference --host "localhost" --port 9997' in terminal
                 3. Run this python file in new terminal window
                 
                 If you decide to change the port number in step 2,
                 please also change the definition of client in the code so the two numbers match
             '''))

    args = parser.parse_args()

    client = Client("http://localhost:9997")
    model_uid = client.launch_model(model_name="vicuna-v1.3")
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
            "Give me a one sentence horror story.",
            "Write a haiku using the word trignometry.",
            "What is a good story opener?"
        ],
        title="Xinference Chat Bot"
    )
    demo.launch()
