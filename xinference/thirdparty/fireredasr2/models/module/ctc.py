# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import numpy as np
import torch
import torch.nn.functional as F


class CTC(torch.nn.Module):
    def __init__(self, odim, encoder_output_size):
        super().__init__()
        self.ctc_lo = torch.nn.Linear(encoder_output_size, odim)

    def forward(self, encoder_output_pad):
        """encoder_output_pad: (N, T, H)"""
        return F.log_softmax(self.ctc_lo(encoder_output_pad), dim=2)

    @classmethod
    def ctc_align(cls, ctc_probs, y, blank_id=0):
        """ctc forced alignment.

        Args:
            torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
            torch.Tensor y: id sequence tensor 1d tensor (L)
            int blank_id: blank symbol index
        Returns:
            torch.Tensor: alignment result
        """
        y_insert_blank = insert_blank(y, blank_id)

        log_alpha = torch.zeros((ctc_probs.size(0), len(y_insert_blank)))
        log_alpha = log_alpha - float('inf')  # log of zero
        state_path = (torch.zeros(
            (ctc_probs.size(0), len(y_insert_blank)), dtype=torch.int16) - 1
        )  # state path

        # init start state
        log_alpha[0, 0] = ctc_probs[0][y_insert_blank[0]]
        log_alpha[0, 1] = ctc_probs[0][y_insert_blank[1]]

        for t in range(1, ctc_probs.size(0)):
            for s in range(len(y_insert_blank)):
                if y_insert_blank[s] == blank_id or s < 2 or y_insert_blank[
                        s] == y_insert_blank[s - 2]:
                    candidates = torch.tensor(
                        [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = torch.tensor([
                        log_alpha[t - 1, s],
                        log_alpha[t - 1, s - 1],
                        log_alpha[t - 1, s - 2],
                    ])
                    prev_state = [s, s - 1, s - 2]
                log_alpha[t, s] = torch.max(candidates) + ctc_probs[t][y_insert_blank[s]]
                state_path[t, s] = prev_state[torch.argmax(candidates)]

        state_seq = -1 * torch.ones((ctc_probs.size(0), 1), dtype=torch.int16)

        candidates = torch.tensor([
            log_alpha[-1, len(y_insert_blank) - 1],
            log_alpha[-1, len(y_insert_blank) - 2]
        ])
        prev_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
        state_seq[-1] = prev_state[torch.argmax(candidates)]
        for t in range(ctc_probs.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_alignment = []
        for t in range(0, ctc_probs.size(0)):
            output_alignment.append(y_insert_blank[state_seq[t, 0]])

        return output_alignment

    @classmethod
    def ctc_alignment_to_timestamp(cls, ys_with_blank, subsampling, blank_id=0):
        start_times: List[float] = []
        end_times: List[float] = []
        frame_shift = 10 # ms, hard code
        T = len(ys_with_blank)
        t = 0
        ctc_durs = []
        while t < T:
            token = ys_with_blank[t]
            t += 1
            if token != blank_id:
                start_t = t
                timestamp = frame_shift * subsampling * t / 1000.0 # s
                start_times.append(timestamp)
                if len(start_times) == len(end_times) + 2:
                    end_times.append(start_times[-1])
                # skip repeat token
                while t < T and token == ys_with_blank[t]:
                    t += 1
                assert t-start_t >= 0
                ctc_durs.append((t-start_t+1) * frame_shift * subsampling / 1000.0)
        end_times.append((frame_shift * subsampling * T + 25)/ 1000.0)
        if len(start_times) == 0:
            start_times.append(0.0)

        # Refine end_times
        assert len(ctc_durs) == len(end_times) and len(start_times) == len(end_times)
        avg_dur = sum(e-s for s, e in zip(start_times, end_times)) / len(end_times)
        new_end_times = []
        for s, e, ctc_dur in zip(start_times, end_times, ctc_durs):
            if e - s > 2 * avg_dur:
                e = s + max(1.5*avg_dur, ctc_dur)
            new_end_times.append(round(e, 3))
        end_times = new_end_times
        return start_times, end_times


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label
