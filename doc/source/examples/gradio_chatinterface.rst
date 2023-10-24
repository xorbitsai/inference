.. _examples_gradio_chatinterface:

==============================
Example: Gradio ChatInterface🤗
==============================

**Description**:

This example showcases how to build a chatbot with 120 lines of code with Gradio ChatInterface and Xinference local LLM

**Used Technology**:

    @ `Xinference <https://github.com/xorbitsai/inference>`_ as a LLM model hosting service

    @ `Gradio <https://github.com/gradio-app/gradio>`_ as a web interface for the chatbot

**Detailed Explanation on the Demo Functionality** :

* Parse user-provided command line arguments to capture essential model parameters such as model name, size, format, and quantization.

* Establish a connection to the Xinference framework and deploy the specified model, ensuring it's ready for real-time interactions.

* Implement helper functions (flatten and to_chat) to efficiently handle and store chat interactions, ensuring the model has context for generating relevant responses.

* Set up an interactive chat interface using Gradio, allowing users to communicate with the model in a user-friendly environment.

* Activate the Gradio web interface, enabling users to start their chat sessions and receive model-generated responses based on their queries.

**Source Code** :
    * `Gradio ChatInterface <https://github.com/xorbitsai/inference/blob/main/examples/gradio_chatinterface.py>`_