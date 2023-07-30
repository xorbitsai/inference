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

🔌 **サードパーティライブラリとの組み込み統合**: Xorbits Inference は、LangChain や LlamaIndex のような人気のあるサードパーティライブラリと
シームレスに統合されています。(近日公開)

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
**2 つの最先端 AI モデルを並べてチャットし、パフォーマンスを比較することもできます**！

![web UI](assets/demo.gif)

### Xinference CLI
Xinference には、モデル管理のためのコマンドラインインターフェース（CLI）が用意されています。便利なコマンドをいくつか紹介します:

- モデルを起動する（モデルの UID が返される）: `xinference launch`
- 実行中のモデルをリストアップする: `xinference list`
- 全てのビルトインモデルをリストアップする: `xinference list --all`
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
$ xinference list --all
```

### ggmlv3 モデル

| Name          | Type             | Language | Format  | Size (in billions) | Quantization                            |
|---------------|------------------|----------|---------|--------------------|-----------------------------------------|
| llama-2       | Foundation Model | en       | ggmlv3  | 7, 13              | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| baichuan      | Foundation Model | en, zh   | ggmlv3  | 7                  | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| llama-2-chat  | RLHF Model       | en       | ggmlv3  | 7, 13, 70          | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| chatglm       | SFT Model        | en, zh   | ggmlv3  | 6                  | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
| chatglm2      | SFT Model        | en, zh   | ggmlv3  | 6                  | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
| wizardlm-v1.0 | SFT Model        | en       | ggmlv3  | 7, 13, 33          | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| wizardlm-v1.1 | SFT Model        | en       | ggmlv3  | 13                 | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| vicuna-v1.3   | SFT Model        | en       | ggmlv3  | 7, 13              | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
| orca          | SFT Model        | en       | ggmlv3  | 3, 7, 13           | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |

### pytorch モデル

| Name          | Type             | Language | Format  | Size (in billions) | Quantization             |
|---------------|------------------|----------|---------|--------------------|--------------------------|
| baichuan      | Foundation Model | en, zh   | pytorch | 7, 13              | '4-bit', '8-bit', 'none' |
| baichuan-chat | SFT Model        | en, zh   | pytorch | 13                 | '4-bit', '8-bit', 'none' |
| vicuna-v1.3   | SFT Model        | en       | pytorch | 7, 13, 33          | '4-bit', '8-bit', 'none' |


**注**:
- Xinference は自動的にモデルをダウンロードし、デフォルトでは `${USER}/.xinference/cache` の下に保存されます。
- Foundation モデルは `generate` インターフェースのみを提供する。
- RLHF と SFT のモデルは `generate` と `chat` の両方を提供する。
- Apple Metal GPU をアクセラレーションに使用する場合は、q4_0 と q4_1 の量子化方法を選択してください。
- `llama-2-chat` 70B ggmlv3 モデルは現在 q4_0 量子化しかサポートしていない。


## Pytorch モデルのベストプラクティス

最近 Pytorch が統合されました。使用シナリオを以下に説明します:

### サポートモデル
- 基礎モデル: baichuan（7B、13B）。
- SFT モデル: baichuan-chat（13B）、vicuna-v1.3（7B、13B、33B）。

### サポートデバイス
- CUDA： Linux と Windows システムでは、デフォルトで `cuda` デバイスが使用される。
- MPS： Mac M1/M2 デバイスでは、デフォルトで `mps` デバイスが使用される。
- CPU： `cpu` デバイスを使用することは推奨されない。多くのメモリを消費し、推論速度が非常に遅くなるからです。

### 量子化メソッド
- `none`: 量子化を行わないことを示す。
- `8-bit`: 8 ビット量子化を使用する。
- `4-bit`: 4 ビット量子化を使用する。注意：4ビット量子化は Linux システムと CUDA デバイスでのみサポートされています。

### その他の命令
- MacOSシステムでは、baichuan-chat モデルはサポートされておらず、baichuan モデルは 8 ビット量子化を使用できない

### ユースケース

以下の表は、一部のモデルのメモリ使用量と対応デバイスを示しています。

| Name          | Size (B) | OS    | No quantization (MB) | Quantization 8-bit (MB) | Quantization 4-bit (MB) |
|---------------|----------|-------|----------------------|-------------------------|-------------------------|
| baichuan-chat | 13       | linux | not currently tested | 13275                   | 7263                    |
| baichuan-chat | 13       | macos | not supported        | not supported           | not supported           |
| vicuna-v1.3   | 7        | linux | 12884                | 6708                    | 3620                    |
| vicuna-v1.3   | 7        | macos | 12916                | 565                     | not supported           |
| baichuan      | 7        | linux | 13480                | 7304                    | 4216                    |
| baichuan      | 7        | macos | 13480                | not supported           | not supported           |



## ロードマップ
Xinference は現在活発に開発中です。今後数週間の開発予定ロードマップは以下の通りです:

### Langchain と LlamaIndex 統合
Xinference があれば、ユーザーはこれらのライブラリを使用し、LLM でアプリケーションを構築することがより簡単になります。
