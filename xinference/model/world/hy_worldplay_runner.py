# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Structured HY-WorldPlay runner used by the Xinference adapter.

The upstream CLI overloads ``--input`` as either a file name or a
``prompt@image`` string.  This wrapper calls the pinned runner directly so a
public prompt is always treated as one literal prompt.
"""

import argparse
import os


def _report_progress(progress: float, info: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"XINFERENCE_PROGRESS:{progress:.2f}:{info}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--negative_prompt",
        default=(
            "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
            "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
            "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
            "杂乱的背景,三条腿,背景人很多,倒着走"
        ),
    )
    parser.add_argument("--image_path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--ar_model_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--num_chunk", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--num_inference_steps", type=int, required=True)
    args = parser.parse_args()

    import torch
    from diffusers.utils import export_to_video
    from wan.generate import WanRunner

    try:
        _report_progress(0.05, "Loading HY-WorldPlay weights")
        runner = WanRunner(
            model_id=args.model_id,
            ckpt_path=args.ckpt_path,
            ar_model_path=args.ar_model_path,
        )
        _report_progress(0.15, "HY-WorldPlay weights loaded")
        _report_progress(0.18, f"Generating {args.num_chunk} chunk(s)")
        result = runner.predict(
            {
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": 1,
                "height": 704,
                "width": 1280,
                "image_path": args.image_path,
                "use_memory": True,
                "context_window_length": 16,
                "seed": 0,
                "pose": args.pose,
                "num_chunk": args.num_chunk,
            }
        )
        _report_progress(0.94, "Encoding HY-WorldPlay video")
        if int(os.environ.get("RANK", "0")) == 0:
            os.makedirs(args.out, exist_ok=True)
            export_to_video(
                result["video"][0], os.path.join(args.out, "world.mp4"), fps=16
            )
            _report_progress(0.98, "Saving HY-WorldPlay video")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
