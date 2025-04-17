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

import contextlib
import glob
import os
import re
import subprocess
import traceback

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


@contextlib.contextmanager
def dist_load(path):
    if not dist.is_initialized() or dist.get_world_size() == 1 or os.path.realpath(path).startswith('/dev/shm'):
        yield path
    else:
        from tts.utils.commons.hparams import hparams
        from tts.utils.commons.trainer import LOCAL_RANK
        tmpdir = '/dev/shm'
        assert len(os.path.basename(path)) > 0
        shm_ckpt_path = f'{tmpdir}/{hparams["exp_name"]}/{os.path.basename(path)}'
        if LOCAL_RANK == 0:
            subprocess.check_call(
                f'mkdir -p {os.path.dirname(shm_ckpt_path)}; '
                f'cp -Lr {path} {shm_ckpt_path}', shell=True)
        dist.barrier()
        yield shm_ckpt_path
        dist.barrier()
        if LOCAL_RANK == 0:
            subprocess.check_call(f'rm -rf {shm_ckpt_path}', shell=True)


def torch_load_dist(path, map_location='cpu'):
    with dist_load(path) as tmp_path:
        checkpoint = torch.load(tmp_path, map_location=map_location)
    return checkpoint


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch_load_dist(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None or steps == 0:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True,
              silent=False, load_opt=False, opts=None, steps=None, checkpoint=None, ckpt_path='', delete_unmatch=True):
    if checkpoint is None:
        if os.path.isfile(ckpt_base_dir):
            base_dir = os.path.dirname(ckpt_base_dir)
            ckpt_path = ckpt_base_dir
            checkpoint = torch_load_dist(ckpt_base_dir, map_location='cpu')
        else:
            base_dir = ckpt_base_dir
            if load_opt:
                checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
            else:
                ckpt_path = f'{ckpt_base_dir}/model_only_last.ckpt'
                if os.path.exists(ckpt_path):
                    checkpoint = torch_load_dist(ckpt_path, map_location='cpu')
                else:
                    checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
    if checkpoint is not None:
        state_dict_all = {
            k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in checkpoint["state_dict"].items()}
        if not isinstance(cur_model, list):
            cur_models = [cur_model]
            model_names = [model_name]
        else:
            cur_models = cur_model
            model_names = model_name
        for model_name, cur_model in zip(model_names, cur_models):
            if isinstance(cur_model, DistributedDataParallel):
                cur_model = cur_model.module
            device = next(cur_model.parameters()).device
            if '.' not in model_name:
                state_dict = state_dict_all[model_name]
            else:
                base_model_name = model_name.split('.')[0]
                rest_model_name = model_name[len(base_model_name) + 1:]
                state_dict = {
                    k[len(rest_model_name) + 1:]: v for k, v in state_dict_all[base_model_name].items()
                    if k.startswith(f'{rest_model_name}.')}
            state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            if not strict and delete_unmatch:
                try:
                    cur_model.load_state_dict(state_dict, strict=True)
                    if not silent:
                        print(f"| loaded '{model_name}' from '{ckpt_path}' with strict=True.")
                except:
                    cur_model_state_dict = cur_model.state_dict()
                    cur_model_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in
                                            cur_model_state_dict.items()}
                    unmatched_keys = []
                    for key, param in state_dict.items():
                        if key in cur_model_state_dict:
                            new_param = cur_model_state_dict[key]
                            if new_param.shape != param.shape:
                                unmatched_keys.append(key)
                                print("| Unmatched keys: ", key, "cur model: ", new_param.shape,
                                        "ckpt model: ", param.shape)
                    for key in unmatched_keys:
                        del state_dict[key]
            load_results = cur_model.load_state_dict(state_dict, strict=strict)
            cur_model.to(device)
            if not silent:
                print(f"| loaded '{model_name}' from '{ckpt_path}'.")
                missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
                print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if load_opt:
            optimizer_states = checkpoint['optimizer_states']
            assert len(opts) == len(optimizer_states)
            for optimizer, opt_state in zip(opts, optimizer_states):
                opt_state = {k.replace('_orig_mod.', ''): v for k, v in opt_state.items()}
                if optimizer is None:
                    return
                try:
                    optimizer.load_state_dict(opt_state)
                    for i, state in enumerate(optimizer.state.values()):
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except ValueError:
                    print(f"| WARMING: optimizer {optimizer} parameters not match !!!")
        return checkpoint.get('global_step', 0)
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)


def load_with_size_mismatch(model, state_dict, prefix=""):
    current_model_dict = model.state_dict()
    cm_keys = current_model_dict.keys()
    mismatch_keys = {k.replace(prefix, "") for k, v in state_dict.items() if k.replace(prefix, "") in cm_keys and v.size() != current_model_dict[k.replace(prefix, "")].size()}
    new_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.replace(prefix, "") in cm_keys and v.size() == current_model_dict[k.replace(prefix, "")].size()}
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"| mismatch keys: ", mismatch_keys)
    if len(missing_keys) > 0:
        print(f"| missing_keys in dit: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"| unexpected_keys in dit: {unexpected_keys}")
