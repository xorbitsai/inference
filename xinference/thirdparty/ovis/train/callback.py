import deepspeed
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from ovis.util.constants import END_LINE, BEGIN_LINE
from ovis.util.utils import rank0_print


class TuneTauCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        visual_tokenizer = kwargs['model'].get_visual_tokenizer()
        current_step = state.global_step
        max_step = state.max_steps
        ratio = current_step / max_step
        visual_tokenizer.config.tau = args.visual_max_tau - (args.visual_max_tau - args.visual_min_tau) * ratio


class MonitorCallback(TrainerCallback):
    def _monitoring(self, model, step):
        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(model.get_monitor_tensors().values()):
                for k, v in model.get_monitor_tensors().items():
                    rank0_print(BEGIN_LINE)
                    rank0_print(f'{k} @ step {step} with sum: {v.sum().item()} and content: ')
                    rank0_print(v)
                    rank0_print(END_LINE)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        step = state.global_step
        if step % args.monitor_step == 0 or step == 10:  # monitor at step 10 for fast check
            self._monitoring(model, step)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        step = state.global_step
        self._monitoring(model, step)
