# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import traceback

import torch
import torchaudio

from .module.conformer_encoder import ConformerEncoder
from .module.ctc import CTC
from .module.transformer_decoder import TransformerDecoder


class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id

        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        self.decoder = TransformerDecoder(
            args.sos_id, args.eos_id, args.pad_id, args.odim,
            args.n_layers_dec, args.n_head, args.d_model,
            args.residual_dropout, args.pe_maxlen)

        self.ctc = CTC(args.odim, args.d_model)

    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
                   return_timestamp=False, elm=None, elm_weight=0.0):
        enc_outputs, enc_lengths, enc_mask = self.encoder(padded_input, input_lengths)
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty,
            elm, elm_weight)
        if return_timestamp:
            nbest_hyps = self.get_token_timestamp_torchaudio(enc_outputs, enc_lengths, nbest_hyps)
        return nbest_hyps

    def get_token_timestamp_torchaudio(self, enc_outputs, enc_lengths, nbest_hyps):
        ctc_logits = self.ctc(enc_outputs)
        enc_lengths = enc_lengths
        for n in range(enc_outputs.size(0)):
            try:
                n_ctc_logits = ctc_logits[n, :enc_lengths[n]]
                y = nbest_hyps[n][0]["yseq"]
                y = y[y!=0]  # 0 is blank
                if y.numel() == 0 or n_ctc_logits.size()[0] == 0:
                    logger.debug("skip null output")
                    nbest_hyps[n][0]["timestamp"] = None
                    continue
                elif y.numel() > n_ctc_logits.size()[0]:
                    nbest_hyps[n][0]["timestamp"] = None
                    continue

                alignment, _ = torchaudio.functional.forced_align(
                    n_ctc_logits.unsqueeze(0), y.unsqueeze(0), blank=0)
                alignment = alignment[0].cpu().tolist()
                start_times, end_times = self.ctc.ctc_alignment_to_timestamp(alignment,
                    self.encoder.input_preprocessor.subsampling, blank_id=0)
                nbest_hyps[n][0]["timestamp"] = (start_times, end_times)
            except:
                traceback.print_exc()
                nbest_hyps[n][0]["timestamp"] = None
        return nbest_hyps
