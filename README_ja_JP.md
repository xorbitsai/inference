<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits Inference: モデルサービングを簡単に 🤖

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=x&style=for-the-badge)](https://twitter.com/xorbitsio)

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

⚙️ **柔軟な API とインターフェース**: OpenAI互換のRESTful API（Function Callingを含む）、RPC、コマンドライン、Web UIなど、
多様なインターフェースを提供し、モデルの管理と相互作用を容易にします。

🌐 **配布デプロイメント**: Excel の分散展開シナリオでは、複数のデバイスやマシンにモデルの推論をシームレスに分散させることができます。

🔌 **サードパーティライブラリとの組み込み統合**: Xorbits Inference は、[LangChain](https://python.langchain.com/docs/integrations/providers/xinference)
や [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window) のような人気のあるサードパーティライブラリと
シームレスに統合されています。

## なぜ Xinference を選ぶのか
| 機能 | Xinference | FastChat | OpenLLM | RayLLM |
|------|------------|----------|---------|--------|
| OpenAI 互換の RESTful API | ✅ | ✅ | ✅ | ✅ |
| vLLM 統合 | ✅ | ✅ | ✅ | ✅ |
| その他の推論エンジン（GGML、TensorRT） | ✅ | ❌ | ✅ | ✅ |
| その他のプラットフォーム（CPU、Metal） | ✅ | ✅ | ❌ | ❌ |
| マルチノードクラスター展開 | ✅ | ❌ | ❌ | ✅ |
| 画像モデル（テキストから画像へ） | ✅ | ✅ | ❌ | ❌ |
| テキスト埋め込みモデル | ✅ | ❌ | ❌ | ❌ |
| マルチモーダルモデル | ✅ | ❌ | ❌ | ❌ |
| より多くのOpenAI機能（関数呼び出し） | ✅ | ❌ | ❌ | ❌ |

## 入門ガイド

**始める前に、GitHubで私たちにスターを付けてください。そうすると、新しいリリースの通知を即座に受け取ることができます！**

* [ドキュメント](https://inference.readthedocs.io/en/latest/index.html)
* [組み込みモデル](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [カスタムモデル](https://inference.readthedocs.io/en/latest/models/custom.html)
* [デプロイメントドキュメント](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)
* [例とチュートリアル](https://inference.readthedocs.io/en/latest/examples/index.html)

### Jupyter Notebook

Xinferenceを体験する最軽量な方法は、私たちの[Google Colab上のJupyterノートブック](https://colab.research.google.com/github/xorbitsai/inference/blob/main/examples/Xinference_Quick_Start.ipynb)を試すことです]。

### Docker

あなたは[Xinference Dockerイメージ](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html)を使用してXinferenceサーバーを開始することができます。インストールコマンドを実行する前に、システムに[Docker](https://docs.docker.com/get-docker/)と[CUDA](https://developer.nvidia.com/cuda-downloads)が設定されていることを確認してください。

### クイックスタート

以下のようにpipを使用してXinferenceをインストールします。（他のオプションについては、[インストールページ](https://inference.readthedocs.io/en/latest/getting_started/installation.html)を参照してください。）

```bash
pip install "xinference[all]"
```

ローカルインスタンスのXinferenceを開始するには、次のコマンドを実行します：

```bash
$ xinference-local
```

Xinferenceが実行されると、Web UI、cURL、コマンドライン、またはXinferenceのPythonクライアントを介して試すことができます。詳細は[ドキュメント](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html#run-xinference-locally)をご覧ください。

![Web UI](assets/screenshot.png)

## 関与する

| プラットフォーム                                                                                   | 目的                                              |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [Github イシュー](https://github.com/xorbitsai/inference/issues)                                  | バグ報告と機能リクエストの提出。                  |
| [Slack](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)      | 他のXorbitsユーザーとの協力。                      |
| [Twitter](https://twitter.com/xorbitsio)                                                          | 新機能に関する最新情報の入手。                    |
