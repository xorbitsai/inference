.. _examples_langchain_streamlit_doc_chat:

======================================
Example: LangChain Streamlit Doc Chat📄
======================================

**Description**:

This Streamlit-based application demonstrates a AI chatbot powered by local LLM and embedding models

**Used Technology**:

    @ `Xinference <https://github.com/xorbitsai/inference>`_: as the LLM and embedding model hosting service

    @ `LangChain <https://github.com/run-llama/llama_index>`_: orchestrates the entire document processing and query answering pipeline

    @ `Streamlit <https://streamlit.io/>`_: for interactive user interface

**Detailed Explanation on the Demo Functionality** :

* Streamlit UI for uploading text files, enhancing user interaction.

* Texts are split into chunks and embedded using Xinference for efficient processing.

* Executes similarity searches on embedded texts to pinpoint relevant sections for user queries.

* Utilizes a structured prompt template for focused LLM interactions.

* Xinference's LLM processes queries within the context of relevant document parts, providing accurate responses.

* The system facilitates effective and context-sensitive document exploration, aiding users in information retrieval.

**Source Code** :
    * `LangChain Streamlit Doc Chat <https://github.com/xorbitsai/inference/blob/main/examples/LangChain_Streamlit_Doc_Chat.py>`_