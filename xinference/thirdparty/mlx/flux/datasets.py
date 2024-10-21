import json
from pathlib import Path

from PIL import Image


class Dataset:
    def __getitem__(self, index: int):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class LocalDataset(Dataset):
    prompt_key = "prompt"

    def __init__(self, dataset: str, data_file):
        self.dataset_base = Path(dataset)
        with open(data_file, "r") as fid:
            self._data = [json.loads(l) for l in fid]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        item = self._data[index]
        image = Image.open(self.dataset_base / item["image"])
        return image, item[self.prompt_key]


class LegacyDataset(LocalDataset):
    prompt_key = "text"

    def __init__(self, dataset: str):
        self.dataset_base = Path(dataset)
        with open(self.dataset_base / "index.json") as f:
            self._data = json.load(f)["data"]


class HuggingFaceDataset(Dataset):

    def __init__(self, dataset: str):
        from datasets import load_dataset as hf_load_dataset

        self._df = hf_load_dataset(dataset)["train"]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index: int):
        item = self._df[index]
        return item["image"], item["prompt"]


def load_dataset(dataset: str):
    dataset_base = Path(dataset)
    data_file = dataset_base / "train.jsonl"
    legacy_file = dataset_base / "index.json"

    if data_file.exists():
        print(f"Load the local dataset {data_file} .", flush=True)
        dataset = LocalDataset(dataset, data_file)
    elif legacy_file.exists():
        print(f"Load the local dataset {legacy_file} .")
        print()
        print("     WARNING: 'index.json' is deprecated in favor of 'train.jsonl'.")
        print("              See the README for details.")
        print(flush=True)
        dataset = LegacyDataset(dataset)
    else:
        print(f"Load the Hugging Face dataset {dataset} .", flush=True)
        dataset = HuggingFaceDataset(dataset)

    return dataset
