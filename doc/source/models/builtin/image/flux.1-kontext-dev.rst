.. _models_builtin_flux.1-kontext-dev:

==================
FLUX.1-Kontext-dev
==================

- **Model Name:** FLUX.1-Kontext-dev
- **Model Family:** stable_diffusion
- **Abilities:** image2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** black-forest-labs/FLUX.1-Kontext-dev
- **GGUF Model ID**: bullerwins/FLUX.1-Kontext-dev-GGUF
- **GGUF Quantizations**: BF16, Q2_K, Q3_K_S, Q4_K_M, Q4_K_S, Q4_K_S, Q5_K_M, Q5_K_S, Q5_K_S, Q6_K, Q8_0


Execute the following command to launch the model::

   xinference launch --model-name FLUX.1-Kontext-dev --model-type image


For GGUF quantization, using below command:

    xinference launch --model-name FLUX.1-Kontext-dev --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True
