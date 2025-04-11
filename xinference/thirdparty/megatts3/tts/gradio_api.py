# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import torch
import os
from functools import partial
import gradio as gr
import traceback
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        inp_audio_path, inp_npy_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content, latent_file=inp_npy_path)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            output_queue.put(None)


def main(inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w, processes, input_queue, output_queue):
    print("Push task to the inp queue |", inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        return res
    else:
        print("")
        return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    else:
        devices = None
    
    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    print("Start open workers")
    for i in range(num_workers):
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, i % len(devices) if devices is not None else None))
        p.start()
        processes.append(p)

    api_interface = gr.Interface(fn=
                                partial(main, processes=processes, input_queue=input_queue, 
                                        output_queue=output_queue), 
                                inputs=[gr.Audio(type="filepath", label="Upload .wav"), gr.File(type="filepath", label="Upload .npy"), "text", 
                                        gr.Number(label="infer timestep", value=32),
                                        gr.Number(label="Intelligibility Weight", value=1.4),
                                        gr.Number(label="Similarity Weight", value=3.0)], outputs=[gr.Audio(label="Synthesized Audio")],
                                title="MegaTTS3",  
                                description="Upload a speech clip as a reference for timbre, " +
                                "upload the pre-extracted latent file, "+
                                "input the target text, and receive the cloned voice.", concurrency_limit=1)
    api_interface.launch(server_name='0.0.0.0', server_port=7929, debug=True)
    for p in processes:
        p.join()
