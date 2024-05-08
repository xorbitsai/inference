# Benchmarking Xinference

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Benchmarking latency

This tool will sample prompts from dataset, and run benchmark with serialized requests.

```bash
python benchmark_latency.py --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
                            --tokenizer /path/to/tokenizer \
                            --num-prompt 100 \
                            --model-uid ${model_uid}
```

## Benchmarking serving

This tool will sample prompts from dataset, and run benchmark with parallel requests.

```bash
python benchmark_serving.py --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
                            --tokenizer /path/to/tokenizer \
                            --model-uid ${model_uid} \
                            --num-prompt 100 --concurrency 50
```

## Benchmarking long context serving

This tool will generate long prompts to sort random numbers, according to specified context length.

```
python benchmark/benchmark_long.py --context-length ${context_length} --tokenizer /path/to/tokenizer \
							--model-uid ${model_uid} \
							--num-prompts 32 -c 16
```
