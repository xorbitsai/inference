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
- Xinference 3.0.0 が公開され、移行メモと破壊的変更を確認できます: [リリースノート](https://xinference.co/release_notes/v3.0.0.html)
- Agent ネイティブ配信：Xinference は [Xagent](https://github.com/xorbitsai/xagent) と統合し、動的プランニング、ツール利用、自己完結型の複数ステップ推論を可能にし、静的なパイプラインの限界を超えます。
- 自動バッチ処理：複数の同時リクエストを自動的にバッチ化し、スループットを大幅に向上させます。: [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): Xinference チームが管理する新しい llama.cpp の Python バインディングは、継続的なバッチ処理をサポートし、より本番運用に適しています。: [#2997](https://github.com/xorbitsai/inference/pull/2997)
- 分散推論：ワーカー間でモデルを実行できます: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- VLLM の強化：複数レプリカ間で KV キャッシュを共有: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### 新規モデル
- [Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) を組み込みでサポート: [#5437](https://github.com/xorbitsai/inference/pull/5437)
- WeMM-Embedding シリーズ（[2B](https://huggingface.co/tencent/WeMM-Embedding-2B)、[4B](https://huggingface.co/tencent/WeMM-Embedding-4B)、[9B](https://huggingface.co/tencent/WeMM-Embedding-9B)）を組み込みでサポート: [#5439](https://github.com/xorbitsai/inference/pull/5439)
- [NaviDC-OCR](https://huggingface.co/StarDoc-AI/NaviDC-OCR) を組み込みでサポート: [#5431](https://github.com/xorbitsai/inference/pull/5431)
- [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) を組み込みでサポート: [#5417](https://github.com/xorbitsai/inference/pull/5417)
- 世界モデル（[Matrix-Game-3.0-5B](https://huggingface.co/Skywork/Matrix-Game-3.0)、[HY-WorldPlay-5B](https://huggingface.co/tencent/HY-WorldPlay)、[Astra](https://huggingface.co/EvanEternal/Astra)）を組み込みでサポート: [#5414](https://github.com/xorbitsai/inference/pull/5414)
- Krea 2 シリーズ（[Raw](https://huggingface.co/krea/Krea-2-Raw)、[Turbo](https://huggingface.co/krea/Krea-2-Turbo)）を組み込みでサポート: [#5419](https://github.com/xorbitsai/inference/pull/5419)
- [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) を組み込みでサポート: [#5413](https://github.com/xorbitsai/inference/pull/5413)
- Ornith 1.5 シリーズ（[35B-A3B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-35B-A3B)、[397B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-397B)）を組み込みでサポート: [#5406](https://github.com/xorbitsai/inference/pull/5406)、[#5405](https://github.com/xorbitsai/inference/pull/5405)
- [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) を組み込みでサポート: [#5404](https://github.com/xorbitsai/inference/pull/5404)
- [GLM-Image](https://huggingface.co/zai-org/GLM-Image) を組み込みでサポート: [#5394](https://github.com/xorbitsai/inference/pull/5394)
- HiDream-O1 シリーズ（[Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image)、[Image-Dev](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev)、[Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604)）を組み込みでサポート: [#5370](https://github.com/xorbitsai/inference/pull/5370)
- [SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) を組み込みでサポート: [#5369](https://github.com/xorbitsai/inference/pull/5369)
- [Ideogram4](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers) を組み込みでサポート: [#5367](https://github.com/xorbitsai/inference/pull/5367)
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) を組み込みでサポート: [#5371](https://github.com/xorbitsai/inference/pull/5371)
- [FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3) を組み込みでサポート: [#5352](https://github.com/xorbitsai/inference/pull/5352)
- [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) を組み込みでサポート: [#5345](https://github.com/xorbitsai/inference/pull/5345)
- Qwen3.8 シリーズ（[27B](https://huggingface.co/Qwen/Qwen3.8-27B)、[2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)）を組み込みでサポート: [#5337](https://github.com/xorbitsai/inference/pull/5337)、[#5339](https://github.com/xorbitsai/inference/pull/5339)
- [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) を組み込みでサポート: [#5327](https://github.com/xorbitsai/inference/pull/5327)
- [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) を組み込みでサポート: [#5322](https://github.com/xorbitsai/inference/pull/5322)
- [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) を組み込みでサポート: [#5319](https://github.com/xorbitsai/inference/pull/5319)
- Ling-3.0 シリーズ（[tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)、[flash](https://huggingface.co/inclusionAI/Ling-3.0-flash)）を組み込みでサポート: [#5311](https://github.com/xorbitsai/inference/pull/5311)
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) と [Lightning LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) を組み込みでサポート: [#5321](https://github.com/xorbitsai/inference/pull/5321)、[#5338](https://github.com/xorbitsai/inference/pull/5338)
- Wan2.2 Animate 2 シリーズ（[14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers)、[14B Distilled](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers)）を組み込みでサポート: [#5309](https://github.com/xorbitsai/inference/pull/5309)
- [FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) を組み込みでサポート: [#5306](https://github.com/xorbitsai/inference/pull/5306)
- CAMPPlus 話者埋め込みシリーズ（[中国語](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)、[中国語・英語高度版](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)）を組み込みでサポート: [#5298](https://github.com/xorbitsai/inference/pull/5298)
- [DeepDoc](https://huggingface.co/Xorbits/deepdoc) を組み込みでサポート: [#5230](https://github.com/xorbitsai/inference/pull/5230)
- [jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) を組み込みでサポート: [#5269](https://github.com/xorbitsai/inference/pull/5269)
- R3 シリーズ（[embedding](https://huggingface.co/tencent/R3-embedding-0.6b)、[reranking](https://huggingface.co/tencent/R3-rerank-0.6b)）を組み込みでサポート: [#5272](https://github.com/xorbitsai/inference/pull/5272)
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

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
