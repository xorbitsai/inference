r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import json
import os
import sys
from pathlib import Path

import lightning
import numpy as np
import rootutils
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from torch import nn
from tqdm.auto import tqdm

from matcha.cli import get_device
from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.logging_utils import pylogger
from matcha.utils.utils import get_phoneme_durations

log = pylogger.get_pylogger(__name__)


def save_durations_to_folder(
    attn: torch.Tensor, x_length: int, y_length: int, filepath: str, output_folder: Path, text: str
):
    durations = attn.squeeze().sum(1)[:x_length].numpy()
    durations_json = get_phoneme_durations(durations, text)
    output = output_folder / Path(filepath).name.replace(".wav", ".npy")
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(durations_json, f, indent=4, ensure_ascii=False)

    np.save(output, durations)


@torch.inference_mode()
def compute_durations(data_loader: torch.utils.data.DataLoader, model: nn.Module, device: torch.device, output_folder):
    """Generate durations from the model for each datapoint and save it in a folder

    Args:
        data_loader (torch.utils.data.DataLoader): Dataloader
        model (nn.Module): MatchaTTS model
        device (torch.device): GPU or CPU
    """

    for batch in tqdm(data_loader, desc="🍵 Computing durations 🍵:"):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        x = x.to(device)
        y = y.to(device)
        x_lengths = x_lengths.to(device)
        y_lengths = y_lengths.to(device)
        spks = spks.to(device) if spks is not None else None

        _, _, _, attn = model(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
        )
        attn = attn.cpu()
        for i in range(attn.shape[0]):
            save_durations_to_folder(
                attn[i],
                x_lengths[i].item(),
                y_lengths[i].item(),
                batch["filepaths"][i],
                output_folder,
                batch["x_texts"][i],
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        default="ljspeech.yaml",
        help="The name of the yaml config file under configs/data",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default="32",
        help="Can have increased batch size for faster computation",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        required=False,
        help="force overwrite the file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file to load the model from",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default=None,
        help="Output folder to save the data statistics",
    )

    parser.add_argument(
        "--cpu", action="store_true", help="Use CPU for inference, not recommended (default: use GPU if available)"
    )

    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))
        cfg["load_durations"] = False

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
    else:
        output_folder = Path(cfg["train_filelist_path"]).parent / "durations"

    print(f"Output folder set to: {output_folder}")

    if os.path.exists(output_folder) and not args.force:
        print("Folder already exists. Use -f to force overwrite")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing: {cfg['name']} from training filelist: {cfg['train_filelist_path']}")
    print("Loading model...")
    device = get_device(args)
    model = MatchaTTS.load_from_checkpoint(args.checkpoint_path, map_location=device)

    text_mel_datamodule = TextMelDataModule(**cfg)
    text_mel_datamodule.setup()
    try:
        print("Computing stats for training set if exists...")
        train_dataloader = text_mel_datamodule.train_dataloader()
        compute_durations(train_dataloader, model, device, output_folder)
    except lightning.fabric.utilities.exceptions.MisconfigurationException:
        print("No training set found")

    try:
        print("Computing stats for validation set if exists...")
        val_dataloader = text_mel_datamodule.val_dataloader()
        compute_durations(val_dataloader, model, device, output_folder)
    except lightning.fabric.utilities.exceptions.MisconfigurationException:
        print("No validation set found")

    try:
        print("Computing stats for test set if exists...")
        test_dataloader = text_mel_datamodule.test_dataloader()
        compute_durations(test_dataloader, model, device, output_folder)
    except lightning.fabric.utilities.exceptions.MisconfigurationException:
        print("No test set found")

    print(f"[+] Done! Data statistics saved to: {output_folder}")


if __name__ == "__main__":
    # Helps with generating durations for the dataset to train other architectures
    # that cannot learn to align due to limited size of dataset
    # Example usage:
    # python python matcha/utils/get_durations_from_trained_model.py -i ljspeech.yaml -c pretrained_model
    # This will create a folder in data/processed_data/durations/ljspeech with the durations
    main()
