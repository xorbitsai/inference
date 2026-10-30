# CI: linux
def test_unsloth_gguf_registry():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("DeepSeek-V4.1-Flash-1017")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    for spec in specs:
        assert spec.model_id == "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
        files, first, merge = generate_model_file_names_with_quantization_parts(spec)
        assert merge
        assert len(files) == int(files[-1].split("-of-")[-1].split(".")[0])
        assert spec.multimodal_projectors == ["mmproj-BF16.gguf", "mmproj-F16.gguf"]
