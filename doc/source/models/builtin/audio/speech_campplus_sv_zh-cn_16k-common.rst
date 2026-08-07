.. _models_builtin_speech_campplus_sv_zh-cn_16k-common:

========================================
speech_campplus_sv_zh-cn_16k-common
========================================

- **Model Name:** speech_campplus_sv_zh-cn_16k-common
- **Model Family:** campplus
- **Abilities:** ['speaker_embedding']
- **Multilingual:** False

This CAMPPlus speaker-verification model converts a speech sample into a
fixed-length representation of speaker identity. It is intended for Chinese
speech and can be used as the embedding stage in speaker verification and
speaker identification systems. It does not transcribe the spoken content.

Specifications
^^^^^^^^^^^^^^

- **Model ID:** iic/speech_campplus_sv_zh-cn_16k-common
- **Model Hub:** `ModelScope <https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common>`__
- **Embedding Dimensions:** 192
- **Sample Rate:** 16 kHz

Output and comparison
^^^^^^^^^^^^^^^^^^^^^

Each request returns one 192-dimensional vector. Store the vector in your
application and use cosine similarity to compare samples. Select a similarity
threshold using representative recordings from the microphones, speakers, and
acoustic conditions expected in production.

Execute the following command to launch the model::

   xinference launch --model-name speech_campplus_sv_zh-cn_16k-common --model-type audio

See :ref:`audio` for Web UI, cURL, and Python examples for the
``/v1/audio/embeddings`` endpoint.
