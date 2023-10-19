.. _examples_chatbot:

=======================
Example: CLI chatbot 🤖️
=======================

**Description**:

Demonstrate how to interact with Xinference to play with LLM chat functionality with an AI agent in command line💻

**Used Technology**:

    @ `ggerganov <https://twitter.com/ggerganov>`_ 's `ggml <https://github.com/ggerganov/ggml>`_

    @ `Xinference <https://github.com/xorbitsai/inference>`_ as a launcher

    @ All LLaMA and Chatglm models supported by `Xorbitsio inference <https://github.com/xorbitsai/inference>`_

**Detailed Explanation on the Demo Functionality** :

1. Take the user command line input in the terminal and grab the required parameters for model launching.

2. Launch the Xinference frameworks and automatically deploy the model user demanded into the cluster.

3. Initialize an empty chat history to store all the context in the chatroom.

4. Recursively ask for user's input as prompt and let the model to generate response based on the prompt and the
   chat history. Show the Output of the response in the terminal.

5. Store the user's input and agent's response into the chat history as context for the upcoming rounds.

**Source Code** :
    * `chat <https://github.com/RayJi01/Xprobe_inference/blob/main/examples/chat.py>`_