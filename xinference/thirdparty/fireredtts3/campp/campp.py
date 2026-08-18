import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from fireredtts3.campp.DTDNN import CAMPPlus


def extract_kaldi_mel(audio: torch.Tensor, audio_sr: int):
    audio = audio[:1]
    if audio_sr != 16000:
        audio = torchaudio.functional.resample(audio, audio_sr, 16000)
        audio_sr = 16000
    feat = kaldi.fbank(
        audio,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)    # (t, c=80)
    return feat


class CamppEmbedding(torch.nn.Module):
    def __init__(self, model_path:str):
        super().__init__()
        self.model_path = model_path
        model = CAMPPlus(feat_dim=80, embedding_size=512)
        sd = torch.load(model_path, weights_only=True, map_location='cpu')
        model.load_state_dict(sd)
        self.model = model
        self.model.eval()
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio:torch.Tensor, audio_sr:int):
        """
        Returns:
            embedding(torch.Tensor): shape (1, 512)
        """
        feat = extract_kaldi_mel(audio, audio_sr)
        feat = feat.unsqueeze(0)
        with torch.no_grad():
            embedding = self.model.forward(feat.to(self.device))    # (b=1, c=512)
            embedding = embedding.cpu()
        return embedding



if __name__ == '__main__':
    campp = CamppEmbedding(
        model_path='pretrained_models/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin',
    )

    audio, audio_sr = torchaudio.load('tests/input/siteng_clean.wav')

    campp.forward(audio, audio_sr)

