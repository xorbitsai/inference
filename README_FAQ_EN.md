[查看中文FAQ](./README_FAQ_CN.md)

# Xinference Frequently Asked Questions (FAQ) Troubleshooting

### Access is possible through `http://localhost:9997`, but not through `ip+9997` of the local machine.

Add `H 0.0.0.0` when starting, as in

```bash
bashxinference -H 0.0.0.0
```

If using docker (official docker is recommended), add `p 9998:9997` when starting, then access is available through `ip+9998` of the local machine.

### Can multiple models be loaded together?

A single GPU can only support loading one LLM model at a time, but it is possible to load an embedding model and a rerank model simultaneously. With multiple GPUs, you can load multiple LLM models.

### Issues with loading or slow downloads of the built-in models in xinference

Xinference by default uses huggiface as the source for models. If your machines are in Mainland China, there might be accessibility issues when using built-in models. 

To address this, add `XINFERENCE_MODEL_SRC=modelscope` when starting the service to change the model source to ModelScope, which is optimized for Mainland China. 

If you're starting xinference with Docker, include `e XINFERENCE_MODEL_SRC=modelscope` during the docker run command. For more environment variable configurations, please refer to the official [[Environment Variables](https://inference.readthedocs.io/zh-cn/latest/getting_started/environments.html)] documentation.

### How to upgrade xinference

```bash
bashpip install --upgrade xinference
```

### Installation of xinference dependencies is slow

We are recommended to use the official docker image for installation. There is a nightly-main version based on the main branch updated daily. For stable versions, see GitHub.

```bash
docker pull xprobe/xinference
```

### Does xinference support configuring LoRA?

It is currently not supported; it requires manual integration with the main model.

### Can't find a custom registration entry point for rerank models in xinference

Upgrade inference to the latest version, versions `0.7.3` and below are not supported.

### Does xinference support running on Huawei Ascend 310 or 910 hardware?

Yes, it does.

### Does xinference support an API that is compatible with OpenAI?

Yes, xinference not only supports an API compatible with OpenAI but also has a client API available for use. For more details, please visit the official website [Client API](https://inference.readthedocs.io/zh-cn/latest/user_guide/client_api.html).

### When using xinference to load models, multi-GPU support is not functioning, and it only loads onto one card.

- If you are using Docker for vLLM multi-GPU inference, you need to specify `-shm-size`. 

- If the vLLM backend is in use, you should disable vLLM before performing the inference.

### Does Xinference support setting up a chat model for embeddings?

It used to, but since the embedding performance of LLMs was poor, the feature has been removed to prevent misuse.
