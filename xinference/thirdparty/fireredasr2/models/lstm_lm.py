# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmLm(nn.Module):
    @classmethod
    def from_args(cls, args):
        args.padding_idx = 2
        args.sos_id = 3
        args.eos_id = 4
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.idim, args.embedding_dim,
                                      padding_idx=args.padding_idx)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_size, args.num_layers,
                            batch_first=True, dropout=args.dropout)
        self.fc_in_dim = args.embedding_dim
        self.fc = nn.Linear(args.embedding_dim, args.odim)

        self._tie_weights(args)
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id
        self.ignore_index = args.padding_idx

    @torch.jit.ignore
    def _tie_weights(self, args):
        if args.tie_weights:
            if self.fc_in_dim != args.embedding_dim or args.idim != args.odim:
                raise ValueError('When using the tied flag, fc_in_dim must be equal to embedding_dim')
            self.fc.weight = self.embedding.weight

    @torch.jit.export
    def init_hidden(self, tensor, batch_size):
        # type: (Tensor, int) -> Tuple[Tensor, Tensor]
        return (tensor.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).float(),
                tensor.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).float())

    @torch.jit.export
    def forward_model(self, padded_inputs, lengths=None, hidden=None):
        # type: (Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # Embedding Layer
        padded_inputs = self.embedding(padded_inputs)  # N, T, D
        # LSTM Layers
        if lengths is None:
            output, new_hidden = self.lstm(padded_inputs, hidden)
        else:
            lengths = lengths.cpu().int()
            total_length = padded_inputs.size(1)  # get the max sequence length
            packed_input = pack_padded_sequence(padded_inputs, lengths,
                                                batch_first=True,
                                                enforce_sorted=False)
            #self.lstm.flatten_parameters()
            packed_output, new_hidden = self.lstm(packed_input, hidden)
            output, _ = pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=total_length)
        # Output Layer
        score = self.fc(output)  # (N, T, V)
        return score, new_hidden
