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
                            --num-prompts 100 \
                            --model-uid ${model_uid}
```

## Benchmarking serving

This tool will sample prompts from dataset, and run benchmark with parallel requests.

```bash
python benchmark_serving.py --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
                            --tokenizer /path/to/tokenizer \
                            --model-uid ${model_uid} \
                            --num-prompts 100 --concurrency 50
```

It can also generate synthetic prompts without a dataset, similar to vLLM's
random benchmark mode:

```bash
python benchmark_serving.py --dataset-name random \
                            --tokenizer /path/to/tokenizer \
                            --model-uid ${model_uid} \
                            --input-len 1024 --output-len 128 \
                            --random-range-ratio 0.2 \
                            --num-prompts 1000 --concurrency 50 \
                            --stream --ignore-eos
```

## Benchmarking embeddings

Launch an embedding model first, then use its model UID:

```bash
python benchmark/benchmark_embedding.py --host localhost --port 9997 \
    --model-uid bge-m3 --num-query 1000 --concurrency 32
```

The default dataset is `clue/tnews`. Each request contains one sentence, so
concurrent requests can exercise server-side batching. Repeat with concurrency
1, 8, 32, and 64 while inspecting the server's batch logs.

The benchmark reuses HTTP connections and sends each selected input once.
`--num-query` caps the number of dataset rows used; it does not repeat a smaller
dataset to reach that count. Up to five warm-up requests are excluded from the
results. Timing ends after all measured requests finish, and throughput counts
only successful requests. Successful and failed request counts are reported
separately; use `--print-error` for failure details.

## Benchmarking long context serving

This tool will generate long prompts to sort random numbers, according to specified context length.

```
python benchmark/benchmark_long.py --context-length ${context_length} --tokenizer /path/to/tokenizer \
							--model-uid ${model_uid} \
							--num-prompts 32 -c 16
```

## Common Options for Benchmarking Tools
- `--stream`. You can enable streaming responses by using the option, which is useful for real-time data processing and receiving incremental data without waiting for the entire dataset to be processed. 

- `--print-error`. For troubleshooting and more detailed output, the option can be used to print detailed error messages if any errors are encountered during the execution. 

These options are available for use in all benchmarking tools provided in this suite, enhancing flexibility and providing essential debugging information.
