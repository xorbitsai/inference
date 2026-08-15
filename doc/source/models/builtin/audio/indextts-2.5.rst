.. _models_builtin_indextts-2.5:

============
IndexTTS-2.5
============

- **Model Name:** IndexTTS-2.5
- **Model Family:** IndexTTS2
- **Abilities:** ['text2audio', 'text2audio_zero_shot', 'text2audio_voice_cloning', 'text2audio_emotion_control']
- **Multilingual:** True (Chinese, English, Japanese, Spanish, and Arabic)

Specifications
^^^^^^^^^^^^^^

- **Model ID:** IndexTeam/IndexTTS-2.5

Execute the following command to launch the model::

   xinference launch --model-name IndexTTS-2.5 --model-type audio

Upstream officially supports Python 3.10 and 3.11. A GPU is strongly
recommended; enable ``use_bf16`` at launch to reduce GPU memory use. Text-based
emotion guidance additionally requires ``use_qwen_emo=True``.
