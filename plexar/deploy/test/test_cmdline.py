import pytest
from click.testing import CliRunner

from plexar.client import Client
from plexar.deploy.cmdline import model_generate


@pytest.mark.asyncio
async def test_generate(setup):
    pool = setup
    address = pool.external_address
    client = Client(address)
    model_uid = client.launch_model("wizardlm-v1.0", quantization="q4_0")
    assert model_uid is not None

    runner = CliRunner()
    result = runner.invoke(
        model_generate,
        [
            # "model",
            # "generate",
            "--model_uid",
            model_uid,
            "--prompt",
            "You are a helpful AI assistant. USER: write a poem. ASSISTANT:",
        ],
    )

    assert len(result.stdout) != 0
    assert type(result.stdout) == str
    assert result.exit_code == 0
    # check whether it's really have comma inside.
    assert "," in str(result.stdout)
