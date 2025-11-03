import os

import numpy as np
import torch

from .. import AudioSignal


def stoi(
    estimates: AudioSignal,
    references: AudioSignal,
    extended: int = False,
):
    """Short term objective intelligibility
    Computes the STOI (See [1][2]) of a denoised signal compared to a clean
    signal, The output is expected to have a monotonic relation with the
    subjective speech-intelligibility, where a higher score denotes better
    speech intelligibility. Uses pystoi under the hood.

    Parameters
    ----------
    estimates : AudioSignal
        Denoised speech
    references : AudioSignal
        Clean original speech
    extended : int, optional
        Boolean, whether to use the extended STOI described in [3], by default False

    Returns
    -------
    Tensor[float]
        Short time objective intelligibility measure between clean and
        denoised speech

    References
    ----------
    1.  C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
        Objective Intelligibility Measure for Time-Frequency Weighted Noisy
        Speech', ICASSP 2010, Texas, Dallas.
    2.  C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
        Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
        IEEE Transactions on Audio, Speech, and Language Processing, 2011.
    3.  Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
        Intelligibility of Speech Masked by Modulated Noise Maskers',
        IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """
    import pystoi

    estimates = estimates.clone().to_mono()
    references = references.clone().to_mono()

    stois = []
    for i in range(estimates.batch_size):
        _stoi = pystoi.stoi(
            references.audio_data[i, 0].detach().cpu().numpy(),
            estimates.audio_data[i, 0].detach().cpu().numpy(),
            references.sample_rate,
            extended=extended,
        )
        stois.append(_stoi)
    return torch.from_numpy(np.array(stois))


def pesq(
    estimates: AudioSignal,
    references: AudioSignal,
    mode: str = "wb",
    target_sr: float = 16000,
):
    """_summary_

    Parameters
    ----------
    estimates : AudioSignal
        Degraded AudioSignal
    references : AudioSignal
        Reference AudioSignal
    mode : str, optional
        'wb' (wide-band) or 'nb' (narrow-band), by default "wb"
    target_sr : float, optional
        Target sample rate, by default 16000

    Returns
    -------
    Tensor[float]
        PESQ score: P.862.2 Prediction (MOS-LQO)
    """
    from pesq import pesq as pesq_fn

    estimates = estimates.clone().to_mono().resample(target_sr)
    references = references.clone().to_mono().resample(target_sr)

    pesqs = []
    for i in range(estimates.batch_size):
        _pesq = pesq_fn(
            estimates.sample_rate,
            references.audio_data[i, 0].detach().cpu().numpy(),
            estimates.audio_data[i, 0].detach().cpu().numpy(),
            mode,
        )
        pesqs.append(_pesq)
    return torch.from_numpy(np.array(pesqs))


def visqol(
    estimates: AudioSignal,
    references: AudioSignal,
    mode: str = "audio",
):  # pragma: no cover
    """ViSQOL score.

    Parameters
    ----------
    estimates : AudioSignal
        Degraded AudioSignal
    references : AudioSignal
        Reference AudioSignal
    mode : str, optional
        'audio' or 'speech', by default 'audio'

    Returns
    -------
    Tensor[float]
        ViSQOL score (MOS-LQO)
    """
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    estimates = estimates.clone().to_mono().resample(target_sr)
    references = references.clone().to_mono().resample(target_sr)

    visqols = []
    for i in range(estimates.batch_size):
        _visqol = api.Measure(
            references.audio_data[i, 0].detach().cpu().numpy().astype(float),
            estimates.audio_data[i, 0].detach().cpu().numpy().astype(float),
        )
        visqols.append(_visqol.moslqo)
    return torch.from_numpy(np.array(visqols))
