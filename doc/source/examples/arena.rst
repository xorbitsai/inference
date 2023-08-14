.. _dual_model_chatbot:

==========================
Example: Chatbot Arena 🤼️
==========================

**Description**:

Experience the thrill of conversing with two AI models simultaneously! This interface allows you to launch two models side by side and engage them in a chat. Enter a prompt, and watch as both models respond to your inquiry 🎙️

**Notice**:

Please do not try to open both models together. Patiently wait until one is finished launching before starting the next. Similarly, wait until you've received responses from both models before moving on to the next question. Thank you for understanding! 🚦

**Used Technology**:

    @ `ggerganov <https://twitter.com/ggerganov>`_ 's `ggml <https://github.com/ggerganov/ggml>`_

    @ `Xinference <https://github.com/xorbitsai/inference>`_ as a launcher

    @ All LLaMA and Chatglm models supported by `Xorbitsio inference <https://github.com/xorbitsai/inference>`_

**Detailed Explanation on the Demo Functionality** :

1. Launch the two models with all required parameters selected by the user.

2. Initialize two separate chat histories to store the context for both models.

3. Prompt the user for input and simultaneously pass it to both models, generating individual responses.

5. Show the outputs of both models in the interface. Respective chat histories serve as context for upcoming rounds.

**Source Code** :
    * `arena <https://github.com/xorbitsai/inference/blob/main/examples/gradio_arena.py>`_
