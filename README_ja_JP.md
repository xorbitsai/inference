<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits Inference: モデルサービングを簡単に 🤖

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=twitter&style=for-the-badge)](https://twitter.com/xorbitsio)

[English](README.md) | [中文介绍](README_zh_CN.md) | 日本語
</div>
<br />


Xorbits Inference(Xinference) は、言語、音声認識、マルチモーダルモデルのために
設計された強力で汎用性の高いライブラリです。 Xorbits Inference を使えば、たった 1 つのコマンドで、
あなたや最先端のビルトインモデルを簡単にデプロイし、提供することができます。 Xorbits Inference は、
研究者、開発者、データサイエンティストを問わず、最先端の AI モデルの可能性を最大限に引き出すことができます。

<div align="center">
<i><a href="https://join.slack.com/t/xorbitsio/shared_invite/zt-1z3zsm9ep-87yI9YZ_B79HLB2ccTq4WA">👉 Slack コミュニティにご参加ください！</a></i>
</div>


## 主な特徴
🌟 **モデルサービングを簡単に**: 大規模な言語、音声認識、マルチモーダルモデルの提供プロセスを簡素化します。
1つのコマンドで、実験用と本番用のモデルをセットアップしてデプロイできます。

⚡️ **最先端モデル**: コマンド1つで最先端のビルトインモデルを実験。
Inference は、最先端のオープンソースモデルへのアクセスを提供します！

🖥 **異機種ハードウェアの利用**: [ggml](https://github.com/ggerganov/ggml) でハードウェアリソースを最大限に活用しましょう。
Xorbits Inference は、GPU や CPU を含む異種ハードウェアをインテリジェントに利用し、モデル推論タスクを高速化します。

⚙️ **柔軟な API とインターフェース**: シームレスな管理とモニタリングのために、RPC、
RESTful API（OpenAI API と互換性あり）、CLI、WebUI をサポートしています。

🌐 **配布デプロイメント**: Excel の分散展開シナリオでは、複数のデバイスやマシンにモデルの推論をシームレスに分散させることができます。

🔌 **サードパーティライブラリとの組み込み統合**: Xorbits Inference は、[LangChain](https://python.langchain.com/docs/integrations/providers/xinference)
や [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window) のような人気のあるサードパーティライブラリと
シームレスに統合されています。

## はじめに
Xinference は PyPI から pip 経由でインストールできます。コンフリクトを避けるため、新しい仮想環境を作成することを強く推奨します。

### インストール
```bash
$ pip install "xinference"
```
`xinference` はモデルを提供するための基本的なパッケージをインストールします。

#### GGML でのインストール
ggml モデルを提供するためには、以下の追加依存関係をインストールする必要があります:
```bash
$ pip install "xinference[ggml]"
```
異なるハードウェアでアクセラレーションを実現したい場合は、
対応するパッケージのインストールマニュアルを参照してください。
- `baichuan`、`wizardlm-v1.0`、`vicuna-v1.3`、`orca` を実行するには、[llama-cpp-python](https://github.com/abetlen/llama-cpp-python#installation-from-pypi-recommended) が必要である。
- `chatglm` と `chatglm2` を実行するには、[chatglm-cpp-python](https://github.com/li-plus/chatglm.cpp#getting-started) が必要である。

#### PyTorch でのインストール
PyTorch のモデルを提供するには、以下の依存関係をインストールする必要があります:
```bash
$ pip install "xinference[pytorch]"
```

#### すべての依存関係を含むインストール
サポートされているすべてのモデルにサービスを提供したい場合は、すべての依存関係をインストールします:
```bash
$ pip install "xinference[all]"
```


### デプロイ
Xinference は、1 つのコマンドでローカルにデプロイすることも、分散クラスタにデプロイすることもできます。

#### ローカル
Xinference のローカルインスタンスを起動するには、以下のコマンドを実行します:
```bash
$ xinference
```

#### 配布

Xinference をクラスタに展開するには、1 台のサーバーで Xinference supervisor を起動し、他のサーバーで
Xinference workers を起動する必要があります。以下の手順に従ってください:

**supervisor のスタート**: Xinference supervisor を実行するサーバーで、以下のコマンドを実行します:
```bash
$ xinference-supervisor -H "${supervisor_host}"
```
`${supervisor_host}` を実際の supervisor サーバのホストに置き換えます。

**Workers のスタート**: Xinference ワーカーを実行したい他の各サーバーで、以下のコマンドを実行します:
```bash
$ xinference-worker -e "http://${supervisor_host}:9997"
```

Xinference が起動すると、CLI または Xinference クライアントからモデル管理のためのエンドポイントにアクセスできるようになります。

- ローカル配置の場合、エンドポイントは `http://localhost:9997` となります。
- クラスタ展開の場合、エンドポイントは `http://${supervisor_host}:9997` になります。
`${supervisor_host}` は supervisor が稼動しているサーバのホスト名または IP アドレスです。

また、Xinference エンドポイントを使用してウェブ UI を表示し、すべての内蔵モデルとチャットすることもできます。

![web UI](assets/index.jpg)

### Xinference CLI
Xinference には、モデル管理のためのコマンドラインインターフェース（CLI）が用意されています。便利なコマンドをいくつか紹介します:

- モデルを起動する（モデルの UID が返される）: `xinference launch`
- 実行中のモデルをリストアップする: `xinference list`
- 全てのビルトインモデルをリストアップする: `xinference registrations`
- モデルを終了する： モデルの終了: `xinference terminate --model-uid ${model_uid}`

### Xinference クライアント
Xinference は、プログラムでモデルを管理し、アクセスするためのクライアントも提供しています:

```python
from xinference.client import Client

client = Client("http://localhost:9997")
model_uid = client.launch_model(model_name="chatglm2")
model = client.get_model(model_uid)

chat_history = []
prompt = "What is the largest animal?"
model.chat(
    prompt,
    chat_history,
    generate_config={"max_tokens": 1024}
)
```

結果:
```json
{
  "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
  "model": "56f69622-1e73-11ee-a3bd-9af9f16816c6",
  "object": "chat.completion",
  "created": 1688919187,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The largest animal that has been scientifically measured is the blue whale, which has a maximum length of around 23 meters (75 feet) for adult animals and can weigh up to 150,000 pounds (68,000 kg). However, it is important to note that this is just an estimate and that the largest animal known to science may be larger still. Some scientists believe that the largest animals may not have a clear \"size\" in the same way that humans do, as their size can vary depending on the environment and the stage of their life."
      },
      "finish_reason": "None"
    }
  ],
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
```

その他の例は [examples](例) を参照。


## 内蔵モデル
内蔵モデルを表示するには、以下のコマンドを実行します:
```bash
$ xinference registrations
```

| Type | Name                | Language     | Ability                |
|------|---------------------|--------------|------------------------|
| LLM  | baichuan            | ['en', 'zh'] | ['embed', 'generate']  |
| LLM  | baichuan-2          | ['en', 'zh'] | ['embed', 'generate']  |
| LLM  | baichuan-chat       | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | baichuan-2-chat     | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | chatglm             | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | chatglm2            | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | chatglm2-32k        | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | code-llama          | ['en']       | ['generate']           |
| LLM  | code-llama-instruct | ['en']       | ['chat']               |
| LLM  | code-llama-python   | ['en']       | ['generate']           |
| LLM  | falcon              | ['en']       | ['embed', 'generate']  |
| LLM  | falcon-instruct     | ['en']       | ['embed', 'chat']      |
| LLM  | gpt-2               | ['en']       | ['generate']           |
| LLM  | internlm            | ['en', 'zh'] | ['embed', 'generate']  |
| LLM  | internlm-chat       | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | internlm-chat-8k    | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | llama-2             | ['en']       | ['embed', 'generate']  |
| LLM  | llama-2-chat        | ['en']       | ['embed', 'chat']      |
| LLM  | opt                 | ['en']       | ['embed', 'generate']  |
| LLM  | orca                | ['en']       | ['embed', 'chat']      |
| LLM  | qwen-chat           | ['en', 'zh'] | ['embed', 'chat']      |
| LLM  | starchat-beta       | ['en']       | ['embed', 'chat']      |
| LLM  | starcoder           | ['en']       | ['generate']           |
| LLM  | starcoderplus       | ['en']       | ['embed', 'generate']  |
| LLM  | vicuna-v1.3         | ['en']       | ['embed', 'chat']      |
| LLM  | vicuna-v1.5         | ['en']       | ['embed', 'chat']      |
| LLM  | vicuna-v1.5-16k     | ['en']       | ['embed', 'chat']      |
| LLM  | wizardlm-v1.0       | ['en']       | ['embed', 'chat']      |
| LLM  | wizardmath-v1.0     | ['en']       | ['embed', 'chat']      |

**注**:
- Xinference は自動的にモデルをダウンロードし、デフォルトでは `${USER}/.xinference/cache` の下に保存されます。
- Hugging Face からモデルをダウンロードする際に問題が発生した場合は、 `export XINFERENCE_MODEL_SRC=xorbits` を実行して、ミラーサイトからモデルをダウンロードしてください。

## カスタムモデル
[カスタムモデル](https://inference.readthedocs.io/en/latest/models/custom.html)を参照してください。




