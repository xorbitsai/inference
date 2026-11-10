# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference（Xorbits Inference）は、オープンソースの分散推論プラットフォームです。**

ローカルマシンから分散クラスターまで、異なる種類のハードウェアにモデルをデプロイして実行できます。言語、視覚、埋め込み、リランキング、画像、音声、動画、ワールド生成（実験的機能）向けに 300 以上の組み込みモデルを提供します。独自モデルの登録も可能で、Flexible タイプでは従来型の機械学習モデルも扱えます。Web UI でデプロイを管理し、OpenAI 互換 API、Python SDK、CLI からモデルを利用できます。

[クイックスタート](#クイックスタート) · [主な機能](#主な機能) · [プラットフォーム比較](#プラットフォーム比較) · [セルフホスティング](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [ドキュメント](https://inference.readthedocs.io/en/latest/) · [モデルカタログ](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [リリース](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![GitHub リリース](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![PyPI バージョン](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Docker タグ](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Python バージョン](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![Python CI](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Apache 2.0 ライセンス](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![GitHub スター](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![GitHub フォーク](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![コントリビューター](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Docker プル数](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Discord コミュニティ](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Telegram コミュニティ](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![X で Xinference をフォロー](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**言語** · [English](../README.md) · **日本語** · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · [Français](./README_fr.md) · [Español](./README_es.md) · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## クイックスタート

### 1. サーバーを起動する

初回起動時に、Xinference がモデルをダウンロードし、エンジンに必要な依存関係をインストールします。

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

[http://127.0.0.1:9997](http://127.0.0.1:9997) を開きます。初回はサーバーログに表示されるセットアップトークンで管理者アカウントを作成し、ログインしてください。 [初回セットアップ](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. チャットモデルをデプロイする

**Launch Model** で `qwen3.5` を検索し、デプロイ画面を開きます。**Recommend configuration** をクリックすると、ハードウェアに応じた設定が入力されます。内容を確認して **Deploy** をクリックします。

![推奨設定を適用した Qwen3.5 のデプロイ画面全体](../assets/guide-launch-recommend-auto.webp)

### 3. 最初のリクエストを送る

**Running Models** でデプロイしたモデルを開き、Web UI でプロンプトを試します。最初のインスタンスのモデル UID は `qwen3.5` です。別の UID が表示される場合は、下の `model` にその値を指定します。API を呼び出すには **API Key Management** でキーを作成します。

```bash
export XINFERENCE_API_KEY="<your-api-key>"
curl --fail-with-body http://127.0.0.1:9997/v1/chat/completions \
  -H "Authorization: Bearer $XINFERENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Translate this sentence into Spanish: The model is ready."}],
    "enable_thinking": false,
    "max_tokens": 96
  }'
```

![Qwen3.5 の応答が表示されたチャット画面](../assets/guide-chat-en.webp)

## 主な機能

| 分野 | Xinference でできること |
| --- | --- |
| モデルカタログ | 言語、視覚、埋め込み、リランキング、画像、音声、動画、ワールド生成（実験的機能）向けに 300 以上の組み込みモデルを提供します。Flexible タイプの従来型機械学習モデルを含め、独自モデルも登録できます。 [モデルカタログ](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| エンジンとハードウェア | 対応する LLM では vLLM、SGLang、TRT-LLM、Transformers、llama.cpp、MLX を選択できます。CPU、GPU、Apple Silicon、一部の MLU 構成への対応はエンジンによって異なります。 [推論バックエンド](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [MLU 対応](https://github.com/xorbitsai/inference/pull/3693) |
| 分散推論 | 単一マシンで実行し、ワーカーやレプリカを追加できます。対応エンジンでは大型モデルを複数のワーカーで実行でき、対応構成では異なるハードウェアに Prefill と Decode を分離できます。 [分散推論](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D 分離](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| API とツール | OpenAI および Anthropic 互換のエンドポイントを提供し、対応モデルではストリーミングやツール呼び出しを利用できます。Web UI、CLI、Python SDK でデプロイと管理が可能です。 [クライアントガイド](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [ツール呼び出し](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| 性能と運用 | Web UI で稼働中のチャットモデルの性能を測定できます。対応構成では Xavier によるプレフィックスキャッシュの共有、一部の実行時設定の変更、レプリカの順次再読み込みが可能です。 [性能テスト](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [実行時設定](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| 多言語対応 | Web UI は英語、簡体字中国語、繁体字中国語、日本語、韓国語に対応しています。README は 10 言語で提供しています。 |

## プラットフォーム比較

| 機能 | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **単体起動（Kubernetes 不要）** | ![対応][compare-supported] | ![対応][compare-supported] | ![公式文書に対応の記載なし][compare-undocumented] | ![対応][compare-supported] |
| **マルチノード推論** | ![対応][compare-supported] | ![対応][compare-supported] | ![対応][compare-supported] | ![対応][compare-supported] |
| **CPU / GPU / Apple Metal** | ![対応][compare-supported] | ![対応][compare-supported] | ![一部対応または追加連携が必要][compare-partial] | ![一部対応または追加連携が必要][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![対応][compare-supported] | ![対応][compare-supported] | ![一部対応または追加連携が必要][compare-partial] | ![一部対応または追加連携が必要][compare-partial] |
| **CUDA と Metal をまたぐ Prefill/Decode** | ![対応][compare-supported] | ![公式文書に対応の記載なし][compare-undocumented] | ![公式文書に対応の記載なし][compare-undocumented] | ![公式文書に対応の記載なし][compare-undocumented] |
| **同一モデル ID に異なるエンジンのレプリカを混在** | ![対応][compare-supported] | ![公式文書に対応の記載なし][compare-undocumented] | ![一部対応または追加連携が必要][compare-partial] | ![一部対応または追加連携が必要][compare-partial] |
| **GPU 上の重みを再利用したレプリカの順次再設定** | ![対応][compare-supported] | ![公式文書に対応の記載なし][compare-undocumented] | ![公式文書に対応の記載なし][compare-undocumented] | ![公式文書に対応の記載なし][compare-undocumented] |
| モデルカタログ | ![対応][compare-supported] 300+ | ![対応][compare-supported] | ![一部対応または追加連携が必要][compare-partial] | ![一部対応または追加連携が必要][compare-partial] |
| Web UI からモデルをデプロイ | ![対応][compare-supported] | ![対応][compare-supported] | ![一部対応または追加連携が必要][compare-partial] | ![公式文書に対応の記載なし][compare-undocumented] |
| 埋め込みとリランキング | ![対応][compare-supported] | ![対応][compare-supported] | ![対応][compare-supported] | ![対応][compare-supported] |
| 画像・動画生成 | ![対応][compare-supported] | ![対応][compare-supported] | ![公式文書に対応の記載なし][compare-undocumented] | ![対応][compare-supported] |
| 音声認識と音声合成 | ![対応][compare-supported] | ![対応][compare-supported] | ![公式文書に対応の記載なし][compare-undocumented] | ![一部対応または追加連携が必要][compare-partial] |

![][compare-supported] 対応 · ![][compare-partial] 一部対応または追加連携が必要 · ![][compare-undocumented] 公式文書に対応の記載なし

<details>
<summary>比較の範囲と出典</summary>

オープンソース版の公式文書に記載された機能を比較しています。エンジンとハードウェアの対応はモデルによって異なります。CUDA と Metal 間の P/D 分離には、対応するモデルとエンジンの組み合わせが必要です。

Xinference では、同一モデル ID の新しいレプリカにエンジンを指定できます。KServe と Dynamo は別々のデプロイ間でリクエストをルーティングできます。GPU 上の重みの再利用は対応する vLLM と SGLang の構成が対象です。

- **Xinference** · [モデルカタログ](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [バックエンド](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [分散推論](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [CUDA と Metal 間のクロスエンジン P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [レプリカのエンジン指定](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [順次再設定](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [モデルとデプロイ](https://localai.io/docs/features/index.html) · [エンジンとハードウェア](https://localai.io/docs/reference/index.print.html) · [分散モード](https://localai.io/docs/features/distributed-mode/) · [エイリアスとフェイルオーバー](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [デプロイとハードウェア](https://kserve.github.io/website/docs/intro) · [組み込みの生成ランタイム](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Kubeflow モデルカタログ](https://www.kubeflow.org/docs/components/hub/installation/) · [Kubeflow Web UI](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [カナリアデプロイ](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [デプロイ、ハードウェア、エンジン](https://docs.dynamo.nvidia.com/dynamo) · [デプロイレシピ](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [画像・動画・音声生成](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [埋め込みとリランキング](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [デプロイ間のルーティング](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## 最近の更新

[v4.2.0 リリースノート](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>直近 2 リリース（v4.1、v4.2）で追加された組み込みモデル</summary>

- [Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b) · [#6003](https://github.com/xorbitsai/inference/pull/6003)
- [MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) · [#5916](https://github.com/xorbitsai/inference/pull/5916)
- [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3) · [#5914](https://github.com/xorbitsai/inference/pull/5914)
- [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) · [#5913](https://github.com/xorbitsai/inference/pull/5913)
- [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) · [#5792](https://github.com/xorbitsai/inference/pull/5792)
- FIBO 2.0 ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)) · [#5790](https://github.com/xorbitsai/inference/pull/5790)
- Ming-3.0 TTS ([Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)) · [#5788](https://github.com/xorbitsai/inference/pull/5788)
- [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) · [#5764](https://github.com/xorbitsai/inference/pull/5764)
- [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [#5755](https://github.com/xorbitsai/inference/pull/5755)
- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) · [#5753](https://github.com/xorbitsai/inference/pull/5753)
- Iris-3B ([Base](https://huggingface.co/speridlabs/iris-3b), [Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)) · [#5686](https://github.com/xorbitsai/inference/pull/5686)
- MiniCPM-V-4.7 ([1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)) · [#5676](https://github.com/xorbitsai/inference/pull/5676)
- Qwen3.8-Flash-Next ([Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)) · [#5673](https://github.com/xorbitsai/inference/pull/5673)
- GLM-5.3 ([BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), [FP8](https://huggingface.co/zai-org/GLM-5.3)) · [#5666](https://github.com/xorbitsai/inference/pull/5666)
- [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) · [#5665](https://github.com/xorbitsai/inference/pull/5665)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [#5661](https://github.com/xorbitsai/inference/pull/5661)

</details>

## モデルの性能テスト

**Performance Tests** を開き、稼働中のチャットモデルとテストする同時実行数を選びます。スループット、最初のトークンまでの時間、出力トークンあたりの時間を比較して、配信構成を選択できます。 [性能テストガイド](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![同時実行数ごとのスループットとレイテンシーのグラフ](../assets/guide-benchmark-en.webp)

*Qwen3.5-0.8B をローカルで実行した例です。性能はハードウェア、エンジン、ワークロードによって異なります。*

## Xinference の使い方

| やりたいこと | 参照先 |
| --- | --- |
| Linux、macOS、Windows にインストール | [1 コマンドでインストール](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [インストールガイド](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| コンテナーまたは Kubernetes で実行 | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| 独自モデルを登録 | [カスタムモデル](https://inference.readthedocs.io/en/latest/models/custom.html) |
| アプリケーションを接続 | [OpenAI 互換 API](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**アプリとフレームワーク** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Xinference Enterprise は自社のクラウドやデータセンターにデプロイできます。Xinference Cloud はマネージド専用デプロイを提供し、Model API はホスト型のモデルエンドポイントを提供します。 [Enterprise の概要](https://xinference.co/enterprise.html)

**法人のお問い合わせ** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## プロジェクトへの参加

| プラットフォーム | 用途 |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | 不具合を報告し、機能を提案する。 |
| [Discord](https://discord.gg/Xw9tszSkr5) | 質問や他のユーザーとの情報交換を行う。 |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Xinference コミュニティで交流する。 |
| [X](https://x.com/Xinference_ai) | プロジェクトのニュースやリリースを確認する。 |

## 引用

研究で Xinference を使用する際は、次の論文を引用してください。 [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

Xinference に貢献してくださる皆さんに感謝します。モデルの追加、ドキュメントの改善、不具合の修正など、最初のプルリクエストをお待ちしています。 [コントリビューションガイド](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Xinference コントリビューターのアバター一覧](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
