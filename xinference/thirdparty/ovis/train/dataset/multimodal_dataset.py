import json
import logging
import os
import traceback
from typing import Dict, Sequence, Union, List

import numpy as np
import torch
import moviepy.editor as mp
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ovis.model.modeling_ovis import Ovis
from ovis.train.arguments import TrainingArguments
from ovis.util.constants import IGNORE_ID


class MultimodalDataset(Dataset):
    def __init__(self, name: str, info: Dict, model: Ovis, training_args: TrainingArguments):
        self.name = name
        self.meta_file = info['meta_file']
        self.image_dir = info['image_dir']
        self.caption_template = info.get('caption_template', None)
        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()
        self.image_height, self.image_width = self.visual_tokenizer.get_image_size()
        self.model = model
        self.text_max_length = training_args.text_max_length
        self.min_frames = training_args.min_frames
        self.max_frames = training_args.max_frames
        self.max_partitions = dict(
            zip(["single_image", "multiple_image", "video"],
                [int(m.strip()) for m in training_args.max_partitions.split('|')])
        )
        self.samples = self.load()

    def load(self):
        raise NotImplementedError

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def read_image(self, path):
        try:
            full_path = os.path.join(self.image_dir, path)
            image = Image.open(full_path).convert('RGB')
            return image, None
        except Exception as e:
            return None, e

    def read_video(self, sample, min_frames, max_frames):
        def _sampling_idx(_len, _min, _max):
            if _len < _min or _len > _max:
                tgt_len = _min if _len < _min else _max
                stride = _len / tgt_len
                sampled_ids = []
                for i in range(tgt_len):
                    start = int(np.round(stride * i))
                    end = int(np.round(stride * (i + 1)))
                    sampled_ids.append(min(_len - 1, (start + end) // 2))
                return sampled_ids
            else:
                return list(range(_len))

        if "video_frames" in sample:
            frames = []
            frames_paths = sample['video_frames']
            sampled_ids = _sampling_idx(len(frames_paths), min_frames, max_frames)
            for idx in sampled_ids:
                frame, last_e = self.read_image(os.path.join(self.image_dir, frames_paths[idx]))
                if frame is None:
                    return None, last_e
                frames.append(frame)
            return frames, None
        elif "video" in sample:
            video_path = os.path.join(self.image_dir, sample['video'])

            max_tries = 2
            last_e = None
            for _ in range(max_tries):
                try:
                    with mp.VideoFileClip(video_path) as clip:
                        total_frames = int(clip.fps * clip.duration)
                        sampled_ids = _sampling_idx(total_frames, min_frames, max_frames)
                        frames = [clip.get_frame(idx / clip.fps) for idx in sampled_ids]
                        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]

                    if len(frames) == 0 or any(frame.size[0] < 5 or frame.size[1] < 5 for frame in frames):
                        raise ValueError("frames are empty or there exists very small frame")
                    return frames, None
                except Exception as e:
                    last_e = f"read video error: {e}\n detailed info: {traceback.format_exc()}"
            return None, last_e
        else:
            return None, RuntimeError(f"missing `video_frames` and `video` in sample: {json.dumps(sample)}")


class DataCollatorForMultimodalDataset:
    def __init__(self, text_tokenizer: PreTrainedTokenizer):
        self.text_tokenizer = text_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        pixel_values, input_ids, labels = tuple([instance[key] for instance in instances]
                                                for key in ("pixel_values", "input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_ID)
        num_valid_label = torch.not_equal(labels, IGNORE_ID).sum().item()
        if num_valid_label == 0:
            logging.warning(
                f'[DataCollatorForMultimodalDataset] All labels in a batch are ignored, which may lead to training instability\n{input_ids=}\n{attention_mask=}\n{labels=}')
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values
        )
