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
