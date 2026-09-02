import os
import torch
from ..utils.load_image import load_image
from loguru import logger
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List, Union

class MonkeyChat_transformers:
    def __init__(self, model_path: str, max_batch_size: int = 10, max_new_tokens=4096, device: str = None):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers is not installed. Please install it following: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md "
                              "to use MonkeyChat_transformers.")
        self.model_name = os.path.basename(model_path)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        bf16_supported = False
        if self.device.startswith("cuda"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif self.device.startswith("mps"):
            bf16_supported = True
            
        logger.info(f"Loading Qwen2.5VL model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Max batch size: {self.max_batch_size}")
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16 if bf16_supported else torch.float16,
                        attn_implementation="flash_attention_2" if self.device.startswith("cuda") else 'sdpa',
                        device_map=self.device,
                    )
                
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"
            
            self.model.eval()
            logger.info("Qwen2.5VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def prepare_messages(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[List[dict]]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")
        
        all_messages = []
        for image, question in zip(images, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": load_image(image, max_size=1600),
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            all_messages.append(messages)
        
        return all_messages
    
    def batch_inference(self, images: List[Union[str, Image.Image]], questions: List[str]) -> List[str]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")
        
        results = []
        total_items = len(images)
        
        for i in range(0, total_items, self.max_batch_size):
            batch_end = min(i + self.max_batch_size, total_items)
            batch_images = images[i:batch_end]
            batch_questions = questions[i:batch_end]
            
            logger.info(f"Processing batch {i//self.max_batch_size + 1}/{(total_items-1)//self.max_batch_size + 1} "
                       f"(items {i+1}-{batch_end})")
            
            try:
                batch_results = self._process_batch(batch_images, batch_questions)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed for items {i+1}-{batch_end}: {e}")
                logger.info("Falling back to single processing...")
                for img, q in zip(batch_images, batch_questions):
                    try:
                        single_result = self._process_single(img, q)
                        results.append(single_result)
                    except Exception as single_e:
                        logger.error(f"Single processing also failed: {single_e}")
                        results.append(f"Error: {str(single_e)}")
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def _process_batch(self, batch_images: List[Union[str, Image.Image]], batch_questions: List[str]) -> List[str]:
        all_messages = self.prepare_messages(batch_images, batch_questions)
        
        texts = []
        image_inputs = []
        
        for messages in all_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_inputs.append(process_vision_info(messages)[0])
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.05,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return [text.strip() for text in output_texts]
    
    def _process_single(self, image: Union[str, Image.Image], question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.05,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
    
    def single_inference(self, image: Union[str, Image.Image], question: str) -> str:
        return self._process_single(image, question)
