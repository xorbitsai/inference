
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi
from typing import Dict, Any, Optional

class SpkembExtractor:

    def __init__(self,
        campplus_model: str,
        target_sr: int = 16000,
    ):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 2
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.target_sr = target_sr


    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding

    def __call__(self, waveform, **kwargs) -> Optional[Dict[str, Any]]:

        spk_emb = self._extract_spk_embedding(waveform)

        return spk_emb
