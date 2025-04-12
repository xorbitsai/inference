import copy
import json
import logging
from datetime import datetime
from typing import Dict

import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import VIDEO_TOKEN, IMAGE_TOKEN
from ovis.util.utils import rank0_print


class ConversationDataset(MultimodalDataset):
    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        rank0_print(f'#samples: {len(samples)}')
        rank0_print(f'sample: {samples[0]}')
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        conversations = copy.deepcopy(sample["conversations"])

        images = None
        max_partition = sample.get('max_partition', None)
        multimodal_type = "text"
        if 'image' in sample:
            multimodal_type = "single_image"
            image_paths = sample['image']
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            images = []
            for image_path in image_paths:
                image, e = self.read_image(image_path)
                if image is None:
                    logging.warning(
                        f'reading image failed with index: {i}, image path: {image_path}, and exception: {e}')
                    images = None
                    break
                images.append(image)
            if images and len(images) > 1:
                multimodal_type = "multiple_image"
        elif "video" in sample or "video_frames" in sample:
            multimodal_type = "video"
            images, e = self.read_video(sample, min_frames=self.min_frames, max_frames=self.max_frames)
            if images:
                num_video_token = 0
                for conv in conversations:
                    if conv['from'] == 'human':
                        num_video_token += conv['value'].count(VIDEO_TOKEN)
                        conv['value'] = conv['value'].replace(VIDEO_TOKEN, '\n'.join([IMAGE_TOKEN] * len(images)))
                if num_video_token != 1:
                    images = None
                    logging.warning(f'invalid sample (currently, only supports single <video>): {json.dumps(sample)}')
            else:
                logging.warning(
                    f'reading video failed with index: {i}, and exception: {e} in sample: {json.dumps(sample)}')

        conv_text = '\n'.join(conv['value'] for conv in conversations)
        if multimodal_type == "text":
            assert conv_text.count(IMAGE_TOKEN) == 0, f'invalid `IMAGE_TOKEN` in sample: {sample}'
        else:
            assert images is None or conv_text.count(IMAGE_TOKEN) == len(images), \
                f'mismatch between #IMAGE_TOKEN and #images in sample: {json.dumps(sample)}'
            max_partition = max_partition or self.max_partitions[multimodal_type]

        prompt, input_ids, pixel_values, labels = self.model.preprocess_inputs(
            conversations,
            images,
            max_partition=max_partition,
            generation_preface=None,
            return_labels=True,
            propagate_exception=False
        )

        if pixel_values is None:
            pixel_values, _ = self.visual_tokenizer.mock_input()

        input_ids = input_ids[:self.text_max_length]
        labels = labels[:self.text_max_length]

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
