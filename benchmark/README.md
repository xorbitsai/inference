# Benchmarking Xinference

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Benchmarking latency
```bash
python benchmark_latency.py --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
                            --tokenizer /path/to/tokenizer \
                            --num-prompt 100 \
                            --model-uid ${model_uid}
```

## Benchmarking serving
```bash
python benchmark_serving.py --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
                            --tokenizer /path/to/tokenizer \
                            --num-prompt 100 \
                            --model-uid ${model_uid}
```
