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

## Comparing SGLang PD paths

For a comparison through the same Xinference API and P/D router, run the
controlled benchmark on an idle two-GPU host:

```bash
python benchmark/benchmark_sglang_pd.py --model-path /path/to/Qwen2.5-0.5B-Instruct \
    --source-commit "$(git rev-parse HEAD)" --output-dir /tmp/sglang-pd-controls \
    --trials 2 --requests 300 --overlap-rounds 2
```

Install SGLang >=0.5.21, `nixl` and `xoscar[nixl]>=0.11.1` in worker and model
environments. The script starts a fresh server for each backend and trial,
reverses backend order on alternate trials, primes twelve mixed short/long
chat requests, measures C16/C32 throughput, and injects 32 cold long prompts
after eight 1024-token background decodes start. It records actual output
lengths, transfer counters, source hashes, runtime versions and GPU-process
snapshots; any unrelated GPU workload during sampling rejects the trial.
Use `--idle-pid` only for a verified idle CUDA context. The native launch selects
`transfer_backend_type="nixl"`; Xavier remains the default.

Both backends use FP16, eager execution, page size 64, identical model weights
and the same memory fraction. Requests ignore EOS and override chat-family stop
strings with an unused marker; the script requires exactly 64 output tokens for
throughput/cold arrivals and 1024 for background decode.
The historical vLLM BF16 numbers are a workload reference, not a direct
cross-engine performance baseline. SGLang Xavier GPU P/D currently retains no
history, so this test does not measure the vLLM tiered-history benefit.

`benchmark_pd.py` accepts a Xinference launch JSON with explicit P/D placements
and a JSONL workload of OpenAI chat or completion request bodies. Use completion
prompts when comparing the transfer paths without differences in chat templates.
Run ordinary replicas and
Xavier PD sequentially against an otherwise idle server. SGLang Xavier P/D uses
GPU-to-GPU NIXL transfer and the native SGLang P/D lifecycle; omit CPU HiCache
settings and `xavier_cache_bytes` from the launch JSON. Install
`xoscar[nixl]>=0.11.1` in worker and model environments:

```bash
python benchmark/benchmark_pd.py --endpoint http://localhost:9997 \
    --launch sglang-launch.json --workload workload.jsonl --output xavier.json \
    --modes hybrid xavier --concurrency 1 4 --repeats 1
```

For a standalone native baseline, launch SGLang's prefill/decode workers and router separately, using
the same weights, tokenizer, dtype, GPU placement and token/memory limits.
With no Xinference model replicas running, measure that existing router:

```bash
python benchmark/benchmark_pd.py --endpoint http://localhost:9997 \
    --launch sglang-launch.json --workload workload.jsonl --output native.json \
    --modes sglang-native --native-sglang-endpoint http://localhost:8000 \
    --native-sglang-model /path/to/model --concurrency 1 4 --repeats 1
```

The native mode never launches or terminates external models. Record the native
server arguments and transport version alongside the results. The runner
reports TTFT, TPOT, latency, output throughput and errors using the same
measurement code for both endpoints. Use disjoint prompts or flush caches
between independent cold runs; repeated prompts and later concurrency runs may
reuse cache. Kernel warmup uses a separate prompt.

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
