import pytest

from ..supervisor import SupervisorActor


@pytest.mark.asyncio
async def test_get_llm_spec():
    supervisor = SupervisorActor()
    llm_family = await supervisor.get_llm_spec(
        "TheBloke/Llama-2-7B-Chat-GGML", "ggmlv3", "huggingface"
    )
    assert llm_family is not None
    assert len(llm_family.model_specs) == 1
    assert llm_family.model_specs[0].model_id == "TheBloke/Llama-2-7B-Chat-GGML"
    assert llm_family.model_specs[0].model_size_in_billions == 7
    assert llm_family.model_specs[0].model_hub == "huggingface"
    assert len(llm_family.model_specs[0].quantizations) == 14
    assert (
        llm_family.model_specs[0].model_file_name_template
        == "llama-2-7b-chat.ggmlv3.{quantization}.bin"
    )
    assert llm_family.model_specs[0].model_file_name_split_template is None
    assert llm_family.model_specs[0].quantization_parts is None

    assert {
        "q2_K",
        "q3_K_L",
        "q3_K_M",
        "q3_K_S",
        "q4_0",
        "q4_1",
        "q4_K_M",
        "q4_K_S",
        "q5_0",
        "q5_1",
        "q5_K_M",
        "q5_K_S",
        "q6_K",
        "q8_0",
    }.intersection(set(llm_family.model_specs[0].quantizations)) == set(
        llm_family.model_specs[0].quantizations
    )

    llm_family = await supervisor.get_llm_spec(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "ggufv2", "huggingface"
    )
    assert llm_family is not None
    assert len(llm_family.model_specs) == 1
    assert llm_family.model_specs[0].model_id == "TheBloke/KafkaLM-70B-German-V0.1-GGUF"
    assert llm_family.model_specs[0].model_size_in_billions == 70
    assert llm_family.model_specs[0].model_hub == "huggingface"
    qs = llm_family.model_specs[0].quantizations
    assert len(qs) == 12
    assert (
        llm_family.model_specs[0].model_file_name_template
        == "kafkalm-70b-german-v0.1.{quantization}.gguf"
    )
    assert (
        llm_family.model_specs[0].model_file_name_split_template
        == "kafkalm-70b-german-v0.1.{quantization}.gguf-split-{part}"
    )
    parts = llm_family.model_specs[0].quantization_parts
    assert parts is not None
    assert len(parts) == 2
    assert len(parts["Q8_0"]) == 2

    assert {
        "Q2_K",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_0",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
    }.intersection(set(qs)) == set(qs)
