import logging
from datetime import datetime
from typing import Dict

import pandas
import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import IMAGE_TOKEN, IGNORE_ID
from ovis.util.utils import rank0_print


class CaptionDataset(MultimodalDataset):

    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        samples = pandas.read_parquet(self.meta_file, engine='pyarrow')
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples.iloc[i]
        text = sample['caption']
        image_path = sample['image_path']
        multimodal_type = "single_image"

        # read and preprocess image
        max_partition = sample.get('max_partition') or self.max_partitions[multimodal_type]
        pixel_values, image_placeholders = self.visual_tokenizer.mock_input()
        valid_image = False
        image, e = self.read_image(image_path)
        if image is None:
            logging.warning(
                f'reading image failed with index: {i}, image path: {image_path}, and exception: {e}')
        else:
            try:
                pixel_values, image_placeholders = self.visual_tokenizer.preprocess_image(
                    image, max_partition=max_partition)
                valid_image = True
            except Exception as e:
                logging.warning(
                    f'preprocessing image failed with index: {i}, image path: {image_path}, and exception: {e}')

        # preprocess text
        if text is None:
            logging.warning(f'text is `None`, index: {i}')
            text = ""
        if not valid_image:
            logging.warning(f'image is not valid, so set text as empty, index: {i}, image path: {image_path}')
            text = ""
        text = text.replace(IMAGE_TOKEN, '').strip()
        head, tail = self.caption_template.split(IMAGE_TOKEN)
        head_ids = self.text_tokenizer(head, add_special_tokens=False).input_ids
        tail_ids = self.text_tokenizer(tail, add_special_tokens=False).input_ids
        text_ids = self.text_tokenizer(text, add_special_tokens=False).input_ids
        input_ids = head_ids + image_placeholders + tail_ids + text_ids
        labels = [IGNORE_ID] * (len(input_ids) - len(text_ids)) + text_ids

        input_ids = input_ids[:self.text_max_length]
        labels = labels[:self.text_max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
