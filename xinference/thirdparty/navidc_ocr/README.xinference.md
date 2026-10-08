# NaviDC-OCR vendored vLLM runtime

This directory vendors `NaviOCR-vllm/NaviOCR_vllm/qwen2_5_vl.py` from:

- Repository: https://github.com/caipeng328/NaviDC-OCR
- Commit: `2e79d29bf32d4e8997b7cbd2ee619a12bfc8d616`
- Source SHA-256: `f0f12a0c8809777c440b8cfc9fb67ee4c0e900919c63a54481dd68fdf797eba3`
- License: Apache License 2.0, as declared by the vendored source header

The model source is unchanged. Xinference provides its own registration module
so the implementation uses the explicit `NaviOCRForConditionalGeneration`
architecture without replacing vLLM's global
`Qwen2_5_VLForConditionalGeneration` registration.

The upstream implementation targets `vllm==0.11.0` and
`transformers==4.57.1`.
