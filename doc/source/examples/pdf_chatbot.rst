.. _examples_pdf_chatbot:

======================
Example: PDF Chatbot📚
======================

**Description**:

This example showcases how to build a PDF chatbot with local LLM and Embedding models

**Used Technology**:

    @ `Xinference <https://github.com/xorbitsai/inference>`_ as a LLM model hosting service

    @ `LlamaIndex <https://github.com/run-llama/llama_index>`_ for orchestrating the entire RAG pipeline 

    @ `Streamlit <https://streamlit.io/>`_ for interactive UI

**Detailed Explanation on the Demo Functionality** :

* Crafted a Dockerfile to simplify the process and ensure easy reproducibility.

* Set up models with Xinference and expose two ports for accessing them.

* Leverage Streamlit for seamless file uploads and interactive communication with the chat engine.

* 5x faster doc embedding than OpenAI's API.

* Leveraging the power of GGML to offload models to the GPU, ensuring swift acceleration. Less long waits for returns.

**Source Code** :
    * `PDF Chatbot <https://github.com/onesuper/PDF-Chatbot-Local-LLM-Embeddings>`_