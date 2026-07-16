<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference: モデルサービングを簡単に 🤖

<p align="center">
  <a href="https://xinference.co">Xinference エンタープライズ</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">セルフホスティング</a> ·
  <a href="https://inference.readthedocs.io/">ドキュメント</a>
</p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Docker Pulls](https://img.shields.io/docker/pulls/xprobe/xinference?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xprobe/xinference)
[![Discord](https://img.shields.io/badge/join_Discord-5462eb.svg?logo=discord&style=for-the-badge&logoColor=%23f5f5f5)](https://discord.gg/Xw9tszSkr5)
[![Telegram](https://img.shields.io/badge/join_Telegram-26A5E4.svg?logo=telegram&style=for-the-badge&logoColor=white)](https://t.me/+nCNpwmySwk9iYmI1)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=x&style=for-the-badge)](https://twitter.com/xorbitsio)

<p align="center">
  <a href="../README.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9?style=for-the-badge"></a>
  <a href="./README_ja_JP.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-454545?style=for-the-badge"></a>
  <a href="./README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9?style=for-the-badge"></a>
  <a href="./README_de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9?style=for-the-badge"></a>
  <a href="./README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9?style=for-the-badge"></a>
  <br>
  <a href="./README_es.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9?style=for-the-badge"></a>
  <a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
  <a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />

Xorbits Inference（Xinference）は、言語、音声認識、マルチモーダルモデル向けの高機能で汎用性の高いライブラリです。Xorbits Inference を使えば、たった一つのコマンドで自分のモデルや組み込みの最先端モデルを簡単にデプロイしてサービス化できます。研究者、開発者、データサイエンティストいずれにとっても、最先端 AI モデルの能力を存分に引き出すことができます。

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 Discord コミュニティに参加してください！</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Telegram グループに参加！</a></i>
</div>

## 🔥 注目のトピック
### フレームワークの強化
- Agent ネイティブ配信：Xinference は [Xagent](https://github.com/xorbitsai/xagent) と統合し、動的プランニング、ツール利用、自己完結型の複数ステップ推論を可能にし、静的なパイプラインの限界を超えます。
- 自動バッチ処理：複数の同時リクエストを自動的にバッチ化し、スループットを大幅に向上させます。: [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): Xinference チームが管理する新しい llama.cpp の Python バインディングは、継続的なバッチ処理をサポートし、より本番運用に適しています。: [#2997](https://github.com/xorbitsai/inference/pull/2997)
- 分散推論：ワーカー間でモデルを実行できます: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- VLLM の強化：複数レプリカ間で KV キャッシュを共有: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### 新規モデル
- 組み込み VibeThinker シリーズ（[1.5B](https://huggingface.co/WeiboAI/VibeThinker-1.5B)、[3B](https://huggingface.co/WeiboAI/VibeThinker-3B)）: [#5085](https://github.com/xorbitsai/inference/pull/5085)
- 組み込み Nex-N2 シリーズ（[mini](https://huggingface.co/nex-agi/Nex-N2-mini)、[Pro](https://huggingface.co/nex-agi/Nex-N2-Pro)、[Pro-fp8](https://huggingface.co/nex-agi/Nex-N2-Pro-fp8)）: [#5094](https://github.com/xorbitsai/inference/pull/5094)
- 組み込み [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR): [#5103](https://github.com/xorbitsai/inference/pull/5103)
- 組み込み [Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B): [#5119](https://github.com/xorbitsai/inference/pull/5119)
- 組み込み [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B): [#5010](https://github.com/xorbitsai/inference/pull/5010)
- 組み込み jina-embeddings-v5 シリーズ（[text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)、[text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)、[omni-nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano)、[omni-small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)）: [#5018](https://github.com/xorbitsai/inference/pull/5018)
- 組み込み MiniCPM-V-4.6 シリーズ（[MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)、[MiniCPM-V-4.6-Thinking](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)）: [#5025](https://github.com/xorbitsai/inference/pull/5025)
- 組み込み Tencent Hy-MT2 シリーズ（[1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)、[7B](https://huggingface.co/tencent/Hy-MT2-7B)、[30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B)）: [#5029](https://github.com/xorbitsai/inference/pull/5029)
### 統合
- [Xagent](https://github.com/xorbitsai/xagent): 計画、メモリ、ツール利用を備えたエンタープライズ向けエージェントプラットフォームです。
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): LLMOps プラットフォームで、視覚化・操作可能な形で迅速にアプリを構築できます。
- [FastGPT](https://github.com/labring/FastGPT): LLM ベースのナレッジプラットフォームで、データ処理やモデル呼び出し機能を提供します。
- [RAGFlow](https://github.com/infiniflow/ragflow): 深層ドキュメント理解に基づくオープンソース RAG エンジンです。
- [MaxKB](https://github.com/1Panel-dev/MaxKB): RAG を統合したオープンソースの知識ベースアシスタントです。

## 主な機能
🌟 **モデルサービングを簡単に**: 大規模言語モデル、音声認識、マルチモーダルモデルの提供プロセスを簡素化します。実験用・本番用のモデルをワンコマンドでセットアップしてデプロイできます。

⚡️ **最先端モデルを手軽に**: 組み込みの最先端モデルをコマンド一つで試せます。Xinference はオープンソースの最先端モデルへのアクセスを提供します。

🖥 **異種ハードウェアの活用**: [ggml](https://github.com/ggerganov/ggml) を用いて GPU と CPU を効率的に利用し、推論を高速化します。

⚙️ **柔軟な API とインターフェース**: OpenAI 互換の RESTful API（Function Calling を含む）、RPC、CLI、Web UI など、多様なインターフェースでモデルを操作できます。

🌐 **分散デプロイ**: 複数デバイスやマシンにまたがる分散デプロイを容易にし、推論をシームレスに分散します。

🔌 **サードパーティとの統合**: [LangChain](https://python.langchain.com/docs/integrations/providers/xinference)、[LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window)、[Dify](https://docs.dify.ai/advanced/model-configuration/xinference)、[Chatbox](https://chatboxai.app/) 等とシームレスに連携します。

## なぜ Xinference か
| 機能                                        | Xinference | FastChat | OpenLLM | RayLLM |
|---------------------------------------------|------------|----------|---------|--------|
| OpenAI 互換の RESTful API                    | ✅ | ✅ | ✅ | ✅ |
| vLLM 統合                                   | ✅ | ✅ | ✅ | ✅ |
| 多様な推論エンジン（GGML、TensorRT）        | ✅ | ❌ | ✅ | ✅ |
| 多様なプラットフォーム（CPU、Metal）        | ✅ | ✅ | ❌ | ❌ |
| マルチノードクラスタデプロイ                | ✅ | ❌ | ❌ | ✅ |
| 画像モデル（テキスト→画像）                 | ✅ | ✅ | ❌ | ❌ |
| テキスト埋め込みモデル                      | ✅ | ❌ | ❌ | ❌ |
| マルチモーダルモデル                        | ✅ | ❌ | ❌ | ❌ |
| 音声モデル                                  | ✅ | ❌ | ❌ | ❌ |
| OpenAI 機能（関数呼び出し）                 | ✅ | ❌ | ❌ | ❌ |

## Xinference の使い方

- **セルフホスティング Xinference Community Edition**
  この [スターターガイド](#getting-started) に従って、自分の環境で Xinference を素早く起動してください。詳細はドキュメント（https://inference.readthedocs.io/）を参照してください。

- **企業/組織向け Xinference**
  企業向けの追加機能を提供しています。企業ニーズについてはメール（mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry）でお問い合わせください。

## 常に先を行くために

GitHub で Xinference にスターを付けると、新しいリリースの通知を受け取れます。

![star-us](../assets/stay_ahead.gif)

## 入門

* [ドキュメント](https://inference.readthedocs.io/en/latest/index.html)
* [組み込みモデル](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [カスタムモデル](https://inference.readthedocs.io/en/latest/models/custom.html)
* [デプロイメントドキュメント](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Docker

Nvidia GPU ユーザーは [Xinference Docker イメージ](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) を使って Xinference サーバを起動できます。インストール実行前に、システムに [Docker](https://docs.docker.com/get-docker/) と [CUDA](https://developer.nvidia.com/cuda-downloads) が導入されていることを確認してください。

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (helm)

Kubernetes クラスタで GPU を有効にした後、次のようにインストールします。

```
# リポジトリを追加
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# インデックスを更新し、バージョンを確認
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Xinference をインストール
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

詳細な K8s 向けカスタムインストールについてはドキュメントを参照してください。

### クイックスタート

pip を使って Xinference をインストールします（詳細はインストールページ参照）。

```bash
pip install "xinference[all]"
```

ローカルインスタンスを起動するには次を実行します：

```bash
$ xinference-local
```

起動後は Web UI、cURL、CLI、または Xinference の Python クライアントを通じて試すことができます。詳細はドキュメントを参照してください。

![web UI](../assets/screenshot.png)

## 参加方法

| プラットフォーム                                                                                        | 目的                                     |
|---------------------------------------------------------------------------------------------------------|-----------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)                                           | バグ報告・機能要望の提出                  |
| [Discord](https://discord.gg/Xw9tszSkr5)                                                                 | 他の Xinference ユーザーとの協力         |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                                                               | 他の Xinference ユーザーとの対話         |
| [Twitter](https://twitter.com/xorbitsio)                                                                 | 新機能の最新情報                         |

## 引用

このプロジェクトが役に立った場合、以下のように引用してください：

```bibtex
@inproceedings{lu2024xinference,
    title = "Xinference: Making Large Model Serving Easy",
    author = "Lu, Weizheng and Xiong, Lingfeng and Zhang, Feng and Qin, Xuye and Chen, Yueguo",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-demo.30",
    pages = "291--300",
}
```

## コントリビューター

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## Star 履歴

[![Star History Chart](https://api.star-history.com/svg?repos=xorbitsai/inference&type=Date)](https://star-history.com/#xorbitsai/inference&Date)
