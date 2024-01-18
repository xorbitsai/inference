[English FAQ](./README_FAQ_EN.md)

# Xinference常见问题汇总及排查

### 通过`http://localhost:9997`可以访问，但是通过本机`ip+9997`不能访问

起动时增加`-H 0.0.0.0`，即

```bash
xinference -H 0.0.0.0
```

如果是通过docker（建议使用官方docker），启动时增加 `-p 9998:9997`，然后通过本机`ip+9998`即可访问

### 是否可以加截多个模型

一张gpu只支持加载一个LLM模型，可以同时加载embedding模型、rerank模型，有多张显卡可以加载多个LLM模型

### 使用xinference内置模型加载不成功/下载很慢

xinference默认使用huggiface源，如果是中国大陆机器使用内置模型就有访问问题，起动时增加`XINFERENCE_MODEL_SRC=modelscope` 将模型源改为中国大陆魔塔，如果是docker启动，docker run时增加`-e XINFERENCE_MODEL_SRC=modelscope`  ，更多环境变量配置，详见官方[环境变量](https://inference.readthedocs.io/zh-cn/latest/getting_started/environments.html)

### xinference如何升级

```bash
pip install —upgrade xinference
```

### xinference安装依赖很慢

推荐使用官方docker镜像安装，每天会有基于 main 分支的 nightly-main 版本，稳定版详见github

```bash
docker pull xprobe/xinference
```

### xinference 支持 配置lora 吗？

暂时不支持，需要手动合并到主模型

### xinference找不到自定义注册rerank模型入口

新升级infernece至最新版本，`0.7.3`及以下版本不支持

### xinference支持在华为昇腾310 或910的硬件上跑吗？

支持

### xinference是否支持openai兼容的api

支持，xinference除支持openai兼容api外，还有client api可以使用，详见官网[客户端API](https://inference.readthedocs.io/zh-cn/latest/user_guide/client_api.html)

### xinference加载模型时，多卡不生效，只加载到1张卡上了

如果是docker 使用 vllm 多卡推理时需要指定 --shm-size

如果是使用vLLM后端，禁用vLLM后推理

### xinfenrence 支持搭建chat模型做emedding吗？

以前支持，但是llm做embedding效果太差了，为了避免大家误用，现已移除。
