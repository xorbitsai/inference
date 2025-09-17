import copy
from contextlib import contextmanager
from inspect import signature
from typing import List

import numpy as np
import torch
from flatten_dict import flatten
from flatten_dict import unflatten
from numpy.random import RandomState

from .. import ml
from ..core import AudioSignal
from ..core import util
from .datasets import AudioLoader

tt = torch.tensor
"""Shorthand for converting things to torch.tensor."""


class BaseTransform:
    """This is the base class for all transforms that are implemented
    in this library. Transforms have two main operations: ``transform``
    and ``instantiate``.

    ``instantiate`` sets the parameters randomly
    from distribution tuples for each parameter. For example, for the
    ``BackgroundNoise`` transform, the signal-to-noise ratio (``snr``)
    is chosen randomly by instantiate. By default, it chosen uniformly
    between 10.0 and 30.0 (the tuple is set to ``("uniform", 10.0, 30.0)``).

    ``transform`` applies the transform using the instantiated parameters.
    A simple example is as follows:

    >>> seed = 0
    >>> signal = ...
    >>> transform = transforms.NoiseFloor(db = ("uniform", -50.0, -30.0))
    >>> kwargs = transform.instantiate()
    >>> output = transform(signal.clone(), **kwargs)

    By breaking apart the instantiation of parameters from the actual audio
    processing of the transform, we can make things more reproducible, while
    also applying the transform on batches of data efficiently on GPU,
    rather than on individual audio samples.

    ..  note::
        We call ``signal.clone()`` for the input to the ``transform`` function
        because signals are modified in-place! If you don't clone the signal,
        you will lose the original data.

    Parameters
    ----------
    keys : list, optional
        Keys that the transform looks for when
        calling ``self.transform``, by default []. In general this is
        set automatically, and you won't need to manipulate this argument.
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0

    Examples
    --------

    >>> seed = 0
    >>>
    >>> audio_path = "tests/audio/spk/f10_script4_produced.wav"
    >>> signal = AudioSignal(audio_path, offset=10, duration=2)
    >>> transform = tfm.Compose(
    >>>     [
    >>>         tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
    >>>         tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
    >>>     ],
    >>> )
    >>>
    >>> kwargs = transform.instantiate(seed, signal)
    >>> output = transform(signal, **kwargs)

    """

    def __init__(self, keys: list = [], name: str = None, prob: float = 1.0):
        # Get keys from the _transform signature.
        tfm_keys = list(signature(self._transform).parameters.keys())

        # Filter out signal and kwargs keys.
        ignore_keys = ["signal", "kwargs"]
        tfm_keys = [k for k in tfm_keys if k not in ignore_keys]

        # Combine keys specified by the child class, the keys found in
        # _transform signature, and the mask key.
        self.keys = keys + tfm_keys + ["mask"]

        self.prob = prob

        if name is None:
            name = self.__class__.__name__
        self.name = name

    def _prepare(self, batch: dict):
        sub_batch = batch[self.name]

        for k in self.keys:
            assert k in sub_batch.keys(), f"{k} not in batch"

        return sub_batch

    def _transform(self, signal):
        return signal

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {}

    @staticmethod
    def apply_mask(batch: dict, mask: torch.Tensor):
        """Applies a mask to the batch.

        Parameters
        ----------
        batch : dict
            Batch whose values will be masked in the ``transform`` pass.
        mask : torch.Tensor
            Mask to apply to batch.

        Returns
        -------
        dict
            A dictionary that contains values only where ``mask = True``.
        """
        masked_batch = {k: v[mask] for k, v in flatten(batch).items()}
        return unflatten(masked_batch)

    def transform(self, signal: AudioSignal, **kwargs):
        """Apply the transform to the audio signal,
        with given keyword arguments.

        Parameters
        ----------
        signal : AudioSignal
            Signal that will be modified by the transforms in-place.
        kwargs: dict
            Keyword arguments to the specific transforms ``self._transform``
            function.

        Returns
        -------
        AudioSignal
            Transformed AudioSignal.

        Examples
        --------

        >>> for seed in range(10):
        >>>     kwargs = transform.instantiate(seed, signal)
        >>>     output = transform(signal.clone(), **kwargs)

        """
        tfm_kwargs = self._prepare(kwargs)
        mask = tfm_kwargs["mask"]

        if torch.any(mask):
            tfm_kwargs = self.apply_mask(tfm_kwargs, mask)
            tfm_kwargs = {k: v for k, v in tfm_kwargs.items() if k != "mask"}
            signal[mask] = self._transform(signal[mask], **tfm_kwargs)

        return signal

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def instantiate(
        self,
        state: RandomState = None,
        signal: AudioSignal = None,
    ):
        """Instantiates parameters for the transform.

        Parameters
        ----------
        state : RandomState, optional
            _description_, by default None
        signal : AudioSignal, optional
            _description_, by default None

        Returns
        -------
        dict
            Dictionary containing instantiated arguments for every keyword
            argument to ``self._transform``.

        Examples
        --------

        >>> for seed in range(10):
        >>>     kwargs = transform.instantiate(seed, signal)
        >>>     output = transform(signal.clone(), **kwargs)

        """
        state = util.random_state(state)

        # Not all instantiates need the signal. Check if signal
        # is needed before passing it in, so that the end-user
        # doesn't need to have variables they're not using flowing
        # into their function.
        needs_signal = "signal" in set(signature(self._instantiate).parameters.keys())
        kwargs = {}
        if needs_signal:
            kwargs = {"signal": signal}

        # Instantiate the parameters for the transform.
        params = self._instantiate(state, **kwargs)
        for k in list(params.keys()):
            v = params[k]
            if isinstance(v, (AudioSignal, torch.Tensor, dict)):
                params[k] = v
            else:
                params[k] = tt(v)
        mask = state.rand() <= self.prob
        params[f"mask"] = tt(mask)

        # Put the params into a nested dictionary that will be
        # used later when calling the transform. This is to avoid
        # collisions in the dictionary.
        params = {self.name: params}

        return params

    def batch_instantiate(
        self,
        states: list = None,
        signal: AudioSignal = None,
    ):
        """Instantiates arguments for every item in a batch,
        given a list of states. Each state in the list
        corresponds to one item in the batch.

        Parameters
        ----------
        states : list, optional
            List of states, by default None
        signal : AudioSignal, optional
            AudioSignal to pass to the ``self.instantiate`` section
            if it is needed for this transform, by default None

        Returns
        -------
        dict
            Collated dictionary of arguments.

        Examples
        --------

        >>> batch_size = 4
        >>> signal = AudioSignal(audio_path, offset=10, duration=2)
        >>> signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])
        >>>
        >>> states = [seed + idx for idx in list(range(batch_size))]
        >>> kwargs = transform.batch_instantiate(states, signal_batch)
        >>> batch_output = transform(signal_batch, **kwargs)
        """
        kwargs = []
        for state in states:
            kwargs.append(self.instantiate(state, signal))
        kwargs = util.collate(kwargs)
        return kwargs


class Identity(BaseTransform):
    """This transform just returns the original signal."""

    pass


class SpectralTransform(BaseTransform):
    """Spectral transforms require STFT data to exist, since manipulations
    of the STFT require the spectrogram. This just calls ``stft`` before
    the transform is called, and calls ``istft`` after the transform is
    called so that the audio data is written to after the spectral
    manipulation.
    """

    def transform(self, signal, **kwargs):
        signal.stft()
        super().transform(signal, **kwargs)
        signal.istft()
        return signal


class Compose(BaseTransform):
    """Compose applies transforms in sequence, one after the other. The
    transforms are passed in as positional arguments or as a list like so:

    >>> transform = tfm.Compose(
    >>>     [
    >>>         tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
    >>>         tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
    >>>     ],
    >>> )

    This will convolve the signal with a room impulse response, and then
    add background noise to the signal. Instantiate instantiates
    all the parameters for every transform in the transform list so the
    interface for using the Compose transform is the same as everything
    else:

    >>> kwargs = transform.instantiate()
    >>> output = transform(signal.clone(), **kwargs)

    Under the hood, the transform maps each transform to a unique name
    under the hood of the form ``{position}.{name}``, where ``position``
    is the index of the transform in the list. ``Compose`` can nest
    within other ``Compose`` transforms, like so:

    >>> preprocess = transforms.Compose(
    >>>     tfm.GlobalVolumeNorm(),
    >>>     tfm.CrossTalk(),
    >>>     name="preprocess",
    >>> )
    >>> augment = transforms.Compose(
    >>>     tfm.RoomImpulseResponse(),
    >>>     tfm.BackgroundNoise(),
    >>>     name="augment",
    >>> )
    >>> postprocess = transforms.Compose(
    >>>     tfm.VolumeChange(),
    >>>     tfm.RescaleAudio(),
    >>>     tfm.ShiftPhase(),
    >>>     name="postprocess",
    >>> )
    >>> transform = transforms.Compose(preprocess, augment, postprocess),

    This defines 3 composed transforms, and then composes them in sequence
    with one another.

    Parameters
    ----------
    *transforms : list
        List of transforms to apply
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(self, *transforms: list, name: str = None, prob: float = 1.0):
        if isinstance(transforms[0], list):
            transforms = transforms[0]

        for i, tfm in enumerate(transforms):
            tfm.name = f"{i}.{tfm.name}"

        keys = [tfm.name for tfm in transforms]
        super().__init__(keys=keys, name=name, prob=prob)

        self.transforms = transforms
        self.transforms_to_apply = keys

    @contextmanager
    def filter(self, *names: list):
        """This can be used to skip transforms entirely when applying
        the sequence of transforms to a signal. For example, take
        the following transforms with the names ``preprocess, augment, postprocess``.

        >>> preprocess = transforms.Compose(
        >>>     tfm.GlobalVolumeNorm(),
        >>>     tfm.CrossTalk(),
        >>>     name="preprocess",
        >>> )
        >>> augment = transforms.Compose(
        >>>     tfm.RoomImpulseResponse(),
        >>>     tfm.BackgroundNoise(),
        >>>     name="augment",
        >>> )
        >>> postprocess = transforms.Compose(
        >>>     tfm.VolumeChange(),
        >>>     tfm.RescaleAudio(),
        >>>     tfm.ShiftPhase(),
        >>>     name="postprocess",
        >>> )
        >>> transform = transforms.Compose(preprocess, augment, postprocess)

        If we wanted to apply all 3 to a signal, we do:

        >>> kwargs = transform.instantiate()
        >>> output = transform(signal.clone(), **kwargs)

        But if we only wanted to apply the ``preprocess`` and ``postprocess``
        transforms to the signal, we do:

        >>> with transform_fn.filter("preprocess", "postprocess"):
        >>>     output = transform(signal.clone(), **kwargs)

        Parameters
        ----------
        *names : list
            List of transforms, identified by name, to apply to signal.
        """
        old_transforms = self.transforms_to_apply
        self.transforms_to_apply = names
        yield
        self.transforms_to_apply = old_transforms

    def _transform(self, signal, **kwargs):
        for transform in self.transforms:
            if any([x in transform.name for x in self.transforms_to_apply]):
                signal = transform(signal, **kwargs)
        return signal

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        parameters = {}
        for transform in self.transforms:
            parameters.update(transform.instantiate(state, signal=signal))
        return parameters

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __len__(self):
        return len(self.transforms)

    def __iter__(self):
        for transform in self.transforms:
            yield transform


class Choose(Compose):
    """Choose logic is the same as :py:func:`audiotools.data.transforms.Compose`,
    but instead of applying all the transforms in sequence, it applies just a single transform,
    which is chosen for each item in the batch.

    Parameters
    ----------
    *transforms : list
        List of transforms to apply
    weights : list
        Probability of choosing any specific transform.
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0

    Examples
    --------

    >>> transforms.Choose(tfm.LowPass(), tfm.HighPass())
    """

    def __init__(
        self,
        *transforms: list,
        weights: list = None,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(*transforms, name=name, prob=prob)

        if weights is None:
            _len = len(self.transforms)
            weights = [1 / _len for _ in range(_len)]
        self.weights = np.array(weights)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        kwargs = super()._instantiate(state, signal)
        tfm_idx = list(range(len(self.transforms)))
        tfm_idx = state.choice(tfm_idx, p=self.weights)
        one_hot = []
        for i, t in enumerate(self.transforms):
            mask = kwargs[t.name]["mask"]
            if mask.item():
                kwargs[t.name]["mask"] = tt(i == tfm_idx)
            one_hot.append(kwargs[t.name]["mask"])
        kwargs["one_hot"] = one_hot
        return kwargs


class Repeat(Compose):
    """Repeatedly applies a given transform ``n_repeat`` times."

    Parameters
    ----------
    transform : BaseTransform
        Transform to repeat.
    n_repeat : int, optional
        Number of times to repeat transform, by default 1
    """

    def __init__(
        self,
        transform,
        n_repeat: int = 1,
        name: str = None,
        prob: float = 1.0,
    ):
        transforms = [copy.copy(transform) for _ in range(n_repeat)]
        super().__init__(transforms, name=name, prob=prob)

        self.n_repeat = n_repeat


class RepeatUpTo(Choose):
    """Repeatedly applies a given transform up to ``max_repeat`` times."

    Parameters
    ----------
    transform : BaseTransform
        Transform to repeat.
    max_repeat : int, optional
        Max number of times to repeat transform, by default 1
    weights : list
        Probability of choosing any specific number up to ``max_repeat``.
    """

    def __init__(
        self,
        transform,
        max_repeat: int = 5,
        weights: list = None,
        name: str = None,
        prob: float = 1.0,
    ):
        transforms = []
        for n in range(1, max_repeat):
            transforms.append(Repeat(transform, n_repeat=n))
        super().__init__(transforms, name=name, prob=prob, weights=weights)

        self.max_repeat = max_repeat


class ClippingDistortion(BaseTransform):
    """Adds clipping distortion to signal. Corresponds
    to :py:func:`audiotools.core.effects.EffectMixin.clip_distortion`.

    Parameters
    ----------
    perc : tuple, optional
        Clipping percentile. Values are between 0.0 to 1.0.
        Typical values are 0.1 or below, by default ("uniform", 0.0, 0.1)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        perc: tuple = ("uniform", 0.0, 0.1),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.perc = perc

    def _instantiate(self, state: RandomState):
        return {"perc": util.sample_from_dist(self.perc, state)}

    def _transform(self, signal, perc):
        return signal.clip_distortion(perc)


class Equalizer(BaseTransform):
    """Applies an equalization curve to the audio signal. Corresponds
    to :py:func:`audiotools.core.effects.EffectMixin.equalizer`.

    Parameters
    ----------
    eq_amount : tuple, optional
        The maximum dB cut to apply to the audio in any band,
        by default ("const", 1.0 dB)
    n_bands : int, optional
        Number of bands in EQ, by default 6
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        return {"eq": eq}

    def _transform(self, signal, eq):
        return signal.equalizer(eq)


class Quantization(BaseTransform):
    """Applies quantization to the input waveform. Corresponds
    to :py:func:`audiotools.core.effects.EffectMixin.quantization`.

    Parameters
    ----------
    channels : tuple, optional
        Number of evenly spaced quantization channels to quantize
        to, by default ("choice", [8, 32, 128, 256, 1024])
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        channels: tuple = ("choice", [8, 32, 128, 256, 1024]),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, signal, channels):
        return signal.quantization(channels)


class MuLawQuantization(BaseTransform):
    """Applies mu-law quantization to the input waveform. Corresponds
    to :py:func:`audiotools.core.effects.EffectMixin.mulaw_quantization`.

    Parameters
    ----------
    channels : tuple, optional
        Number of mu-law spaced quantization channels to quantize
        to, by default ("choice", [8, 32, 128, 256, 1024])
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        channels: tuple = ("choice", [8, 32, 128, 256, 1024]),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.channels = channels

    def _instantiate(self, state: RandomState):
        return {"channels": util.sample_from_dist(self.channels, state)}

    def _transform(self, signal, channels):
        return signal.mulaw_quantization(channels)


class NoiseFloor(BaseTransform):
    """Adds a noise floor of Gaussian noise to the signal at a specified
    dB.

    Parameters
    ----------
    db : tuple, optional
        Level of noise to add to signal, by default ("const", -50.0)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        db: tuple = ("const", -50.0),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        db = util.sample_from_dist(self.db, state)
        audio_data = state.randn(signal.num_channels, signal.signal_length)
        nz_signal = AudioSignal(audio_data, signal.sample_rate)
        nz_signal.normalize(db)
        return {"nz_signal": nz_signal}

    def _transform(self, signal, nz_signal):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal + nz_signal


class BackgroundNoise(BaseTransform):
    """Adds background noise from audio specified by a set of CSV files.
    A valid CSV file looks like, and is typically generated by
    :py:func:`audiotools.data.preprocess.create_csv`:

    ..  csv-table::
        :header: path

        room_tone/m6_script2_clean.wav
        room_tone/m6_script2_cleanraw.wav
        room_tone/m6_script2_ipad_balcony1.wav
        room_tone/m6_script2_ipad_bedroom1.wav
        room_tone/m6_script2_ipad_confroom1.wav
        room_tone/m6_script2_ipad_confroom2.wav
        room_tone/m6_script2_ipad_livingroom1.wav
        room_tone/m6_script2_ipad_office1.wav

    ..  note::
        All paths are relative to an environment variable called ``PATH_TO_DATA``,
        so that CSV files are portable across machines where data may be
        located in different places.

    This transform calls :py:func:`audiotools.core.effects.EffectMixin.mix`
    and :py:func:`audiotools.core.effects.EffectMixin.equalizer` under the
    hood.

    Parameters
    ----------
    snr : tuple, optional
        Signal-to-noise ratio, by default ("uniform", 10.0, 30.0)
    sources : List[str], optional
        Sources containing folders, or CSVs with paths to audio files,
        by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    eq_amount : tuple, optional
        Amount of equalization to apply, by default ("const", 1.0)
    n_bands : int, optional
        Number of bands in equalizer, by default 3
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    loudness_cutoff : float, optional
        Loudness cutoff when loading from audio files, by default None
    """

    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        sources: List[str] = None,
        weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 3,
        name: str = None,
        prob: float = 1.0,
        loudness_cutoff: float = None,
    ):
        super().__init__(name=name, prob=prob)

        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.loader = AudioLoader(sources, weights)
        self.loudness_cutoff = loudness_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = util.sample_from_dist(self.snr, state)

        bg_signal = self.loader(
            state,
            signal.sample_rate,
            duration=signal.signal_duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=signal.num_channels,
        )["signal"]

        return {"eq": eq, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, signal, bg_signal, snr, eq):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal.mix(bg_signal.clone(), snr, eq)


class CrossTalk(BaseTransform):
    """Adds crosstalk between speakers, whose audio is drawn from a CSV file
    that was produced via :py:func:`audiotools.data.preprocess.create_csv`.

    This transform calls :py:func:`audiotools.core.effects.EffectMixin.mix`
    under the hood.

    Parameters
    ----------
    snr : tuple, optional
        How loud cross-talk speaker is relative to original signal in dB,
        by default ("uniform", 0.0, 10.0)
    sources : List[str], optional
        Sources containing folders, or CSVs with paths to audio files,
        by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    loudness_cutoff : float, optional
        Loudness cutoff when loading from audio files, by default -40
    """

    def __init__(
        self,
        snr: tuple = ("uniform", 0.0, 10.0),
        sources: List[str] = None,
        weights: List[float] = None,
        name: str = None,
        prob: float = 1.0,
        loudness_cutoff: float = -40,
    ):
        super().__init__(name=name, prob=prob)

        self.snr = snr
        self.loader = AudioLoader(sources, weights)
        self.loudness_cutoff = loudness_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        snr = util.sample_from_dist(self.snr, state)
        crosstalk_signal = self.loader(
            state,
            signal.sample_rate,
            duration=signal.signal_duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=signal.num_channels,
        )["signal"]

        return {"crosstalk_signal": crosstalk_signal, "snr": snr}

    def _transform(self, signal, crosstalk_signal, snr):
        # Clone bg_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        loudness = signal.loudness()
        mix = signal.mix(crosstalk_signal.clone(), snr)
        mix.normalize(loudness)
        return mix


class RoomImpulseResponse(BaseTransform):
    """Convolves signal with a room impulse response, at a specified
    direct-to-reverberant ratio, with equalization applied. Room impulse
    response data is drawn from a CSV file that was produced via
    :py:func:`audiotools.data.preprocess.create_csv`.

    This transform calls :py:func:`audiotools.core.effects.EffectMixin.apply_ir`
    under the hood.

    Parameters
    ----------
    drr : tuple, optional
        _description_, by default ("uniform", 0.0, 30.0)
    sources : List[str], optional
        Sources containing folders, or CSVs with paths to audio files,
        by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    eq_amount : tuple, optional
        Amount of equalization to apply, by default ("const", 1.0)
    n_bands : int, optional
        Number of bands in equalizer, by default 6
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    use_original_phase : bool, optional
        Whether or not to use the original phase, by default False
    offset : float, optional
        Offset from each impulse response file to use, by default 0.0
    duration : float, optional
        Duration of each impulse response, by default 1.0
    """

    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        sources: List[str] = None,
        weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        offset: float = 0.0,
        duration: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.drr = drr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.use_original_phase = use_original_phase

        self.loader = AudioLoader(sources, weights)
        self.offset = offset
        self.duration = duration

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = util.sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        drr = util.sample_from_dist(self.drr, state)

        ir_signal = self.loader(
            state,
            signal.sample_rate,
            offset=self.offset,
            duration=self.duration,
            loudness_cutoff=None,
            num_channels=signal.num_channels,
        )["signal"]
        ir_signal.zero_pad_to(signal.sample_rate)

        return {"eq": eq, "ir_signal": ir_signal, "drr": drr}

    def _transform(self, signal, ir_signal, drr, eq):
        # Clone ir_signal so that transform can be repeatedly applied
        # to different signals with the same effect.
        return signal.apply_ir(
            ir_signal.clone(), drr, eq, use_original_phase=self.use_original_phase
        )


class VolumeChange(BaseTransform):
    """Changes the volume of the input signal.

    Uses :py:func:`audiotools.core.effects.EffectMixin.volume_change`.

    Parameters
    ----------
    db : tuple, optional
        Change in volume in decibels, by default ("uniform", -12.0, 0.0)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        db: tuple = ("uniform", -12.0, 0.0),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        self.db = db

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, signal, db):
        return signal.volume_change(db)


class VolumeNorm(BaseTransform):
    """Normalizes the volume of the excerpt to a specified decibel.

    Uses :py:func:`audiotools.core.effects.EffectMixin.normalize`.

    Parameters
    ----------
    db : tuple, optional
        dB to normalize signal to, by default ("const", -24)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        db: tuple = ("const", -24),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, signal, db):
        return signal.normalize(db)


class GlobalVolumeNorm(BaseTransform):
    """Similar to :py:func:`audiotools.data.transforms.VolumeNorm`, this
    transform also normalizes the volume of a signal, but it uses
    the volume of the entire audio file the loaded excerpt comes from,
    rather than the volume of just the excerpt. The volume of the
    entire audio file is expected in ``signal.metadata["loudness"]``.
    If loading audio from a CSV generated by :py:func:`audiotools.data.preprocess.create_csv`
    with ``loudness = True``, like the following:

    ..  csv-table::
        :header: path,loudness

        daps/produced/f1_script1_produced.wav,-16.299999237060547
        daps/produced/f1_script2_produced.wav,-16.600000381469727
        daps/produced/f1_script3_produced.wav,-17.299999237060547
        daps/produced/f1_script4_produced.wav,-16.100000381469727
        daps/produced/f1_script5_produced.wav,-16.700000762939453
        daps/produced/f3_script1_produced.wav,-16.5

    The ``AudioLoader`` will automatically load the loudness column into
    the metadata of the signal.

    Uses :py:func:`audiotools.core.effects.EffectMixin.volume_change`.

    Parameters
    ----------
    db : tuple, optional
        dB to normalize signal to, by default ("const", -24)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        db: tuple = ("const", -24),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        if "loudness" not in signal.metadata:
            db_change = 0.0
        elif float(signal.metadata["loudness"]) == float("-inf"):
            db_change = 0.0
        else:
            db = util.sample_from_dist(self.db, state)
            db_change = db - float(signal.metadata["loudness"])

        return {"db": db_change}

    def _transform(self, signal, db):
        return signal.volume_change(db)


class Silence(BaseTransform):
    """Zeros out the signal with some probability.

    Parameters
    ----------
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 0.1
    """

    def __init__(self, name: str = None, prob: float = 0.1):
        super().__init__(name=name, prob=prob)

    def _transform(self, signal):
        _loudness = signal._loudness
        signal = AudioSignal(
            torch.zeros_like(signal.audio_data),
            sample_rate=signal.sample_rate,
            stft_params=signal.stft_params,
        )
        # So that the amound of noise added is as if it wasn't silenced.
        # TODO: improve this hack
        signal._loudness = _loudness

        return signal


class LowPass(BaseTransform):
    """Applies a LowPass filter.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.low_pass`.

    Parameters
    ----------
    cutoff : tuple, optional
        Cutoff frequency distribution,
        by default ``("choice", [4000, 8000, 16000])``
    zeros : int, optional
        Number of zero-crossings in filter, argument to
        ``julius.LowPassFilters``, by default 51
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        cutoff: tuple = ("choice", [4000, 8000, 16000]),
        zeros: int = 51,
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)

        self.cutoff = cutoff
        self.zeros = zeros

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.low_pass(cutoff, zeros=self.zeros)


class HighPass(BaseTransform):
    """Applies a HighPass filter.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.high_pass`.

    Parameters
    ----------
    cutoff : tuple, optional
        Cutoff frequency distribution,
        by default ``("choice", [50, 100, 250, 500, 1000])``
    zeros : int, optional
        Number of zero-crossings in filter, argument to
        ``julius.LowPassFilters``, by default 51
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        cutoff: tuple = ("choice", [50, 100, 250, 500, 1000]),
        zeros: int = 51,
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)

        self.cutoff = cutoff
        self.zeros = zeros

    def _instantiate(self, state: RandomState):
        return {"cutoff": util.sample_from_dist(self.cutoff, state)}

    def _transform(self, signal, cutoff):
        return signal.high_pass(cutoff, zeros=self.zeros)


class RescaleAudio(BaseTransform):
    """Rescales the audio so it is in between ``-val`` and ``val``
    only if the original audio exceeds those bounds. Useful if
    transforms have caused the audio to clip.

    Uses :py:func:`audiotools.core.effects.EffectMixin.ensure_max_of_audio`.

    Parameters
    ----------
    val : float, optional
        Max absolute value of signal, by default 1.0
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(self, val: float = 1.0, name: str = None, prob: float = 1):
        super().__init__(name=name, prob=prob)

        self.val = val

    def _transform(self, signal):
        return signal.ensure_max_of_audio(self.val)


class ShiftPhase(SpectralTransform):
    """Shifts the phase of the audio.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.shift)phase`.

    Parameters
    ----------
    shift : tuple, optional
        How much to shift phase by, by default ("uniform", -np.pi, np.pi)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        shift: tuple = ("uniform", -np.pi, np.pi),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.shift = shift

    def _instantiate(self, state: RandomState):
        return {"shift": util.sample_from_dist(self.shift, state)}

    def _transform(self, signal, shift):
        return signal.shift_phase(shift)


class InvertPhase(ShiftPhase):
    """Inverts the phase of the audio.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.shift_phase`.

    Parameters
    ----------
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(self, name: str = None, prob: float = 1):
        super().__init__(shift=("const", np.pi), name=name, prob=prob)


class CorruptPhase(SpectralTransform):
    """Corrupts the phase of the audio.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.corrupt_phase`.

    Parameters
    ----------
    scale : tuple, optional
        How much to corrupt phase by, by default ("uniform", 0, np.pi)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self, scale: tuple = ("uniform", 0, np.pi), name: str = None, prob: float = 1
    ):
        super().__init__(name=name, prob=prob)
        self.scale = scale

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        scale = util.sample_from_dist(self.scale, state)
        corruption = state.normal(scale=scale, size=signal.phase.shape[1:])
        return {"corruption": corruption.astype("float32")}

    def _transform(self, signal, corruption):
        return signal.shift_phase(shift=corruption)


class FrequencyMask(SpectralTransform):
    """Masks a band of frequencies at a center frequency
    from the audio.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.mask_frequencies`.

    Parameters
    ----------
    f_center : tuple, optional
        Center frequency between 0.0 and 1.0 (Nyquist), by default ("uniform", 0.0, 1.0)
    f_width : tuple, optional
        Width of zero'd out band, by default ("const", 0.1)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        f_center: tuple = ("uniform", 0.0, 1.0),
        f_width: tuple = ("const", 0.1),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.f_center = f_center
        self.f_width = f_width

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        f_center = util.sample_from_dist(self.f_center, state)
        f_width = util.sample_from_dist(self.f_width, state)

        fmin = max(f_center - (f_width / 2), 0.0)
        fmax = min(f_center + (f_width / 2), 1.0)

        fmin_hz = (signal.sample_rate / 2) * fmin
        fmax_hz = (signal.sample_rate / 2) * fmax

        return {"fmin_hz": fmin_hz, "fmax_hz": fmax_hz}

    def _transform(self, signal, fmin_hz: float, fmax_hz: float):
        return signal.mask_frequencies(fmin_hz=fmin_hz, fmax_hz=fmax_hz)


class TimeMask(SpectralTransform):
    """Masks out contiguous time-steps from signal.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.mask_timesteps`.

    Parameters
    ----------
    t_center : tuple, optional
        Center time in terms of 0.0 and 1.0 (duration of signal),
        by default ("uniform", 0.0, 1.0)
    t_width : tuple, optional
        Width of dropped out portion, by default ("const", 0.025)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        t_center: tuple = ("uniform", 0.0, 1.0),
        t_width: tuple = ("const", 0.025),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.t_center = t_center
        self.t_width = t_width

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        t_center = util.sample_from_dist(self.t_center, state)
        t_width = util.sample_from_dist(self.t_width, state)

        tmin = max(t_center - (t_width / 2), 0.0)
        tmax = min(t_center + (t_width / 2), 1.0)

        tmin_s = signal.signal_duration * tmin
        tmax_s = signal.signal_duration * tmax
        return {"tmin_s": tmin_s, "tmax_s": tmax_s}

    def _transform(self, signal, tmin_s: float, tmax_s: float):
        return signal.mask_timesteps(tmin_s=tmin_s, tmax_s=tmax_s)


class MaskLowMagnitudes(SpectralTransform):
    """Masks low magnitude regions out of signal.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.mask_low_magnitudes`.

    Parameters
    ----------
    db_cutoff : tuple, optional
        Decibel value for which things below it will be masked away,
        by default ("uniform", -10, 10)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        db_cutoff: tuple = ("uniform", -10, 10),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.db_cutoff = db_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {"db_cutoff": util.sample_from_dist(self.db_cutoff, state)}

    def _transform(self, signal, db_cutoff: float):
        return signal.mask_low_magnitudes(db_cutoff)


class Smoothing(BaseTransform):
    """Convolves the signal with a smoothing window.

    Uses :py:func:`audiotools.core.effects.EffectMixin.convolve`.

    Parameters
    ----------
    window_type : tuple, optional
        Type of window to use, by default ("const", "average")
    window_length : tuple, optional
        Length of smoothing window, by
        default ("choice", [8, 16, 32, 64, 128, 256, 512])
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        window_type: tuple = ("const", "average"),
        window_length: tuple = ("choice", [8, 16, 32, 64, 128, 256, 512]),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.window_type = window_type
        self.window_length = window_length

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        window_type = util.sample_from_dist(self.window_type, state)
        window_length = util.sample_from_dist(self.window_length, state)
        window = signal.get_window(
            window_type=window_type, window_length=window_length, device="cpu"
        )
        return {"window": AudioSignal(window, signal.sample_rate)}

    def _transform(self, signal, window):
        sscale = signal.audio_data.abs().max(dim=-1, keepdim=True).values
        sscale[sscale == 0.0] = 1.0

        out = signal.convolve(window)

        oscale = out.audio_data.abs().max(dim=-1, keepdim=True).values
        oscale[oscale == 0.0] = 1.0

        out = out * (sscale / oscale)
        return out


class TimeNoise(TimeMask):
    """Similar to :py:func:`audiotools.data.transforms.TimeMask`, but
    replaces with noise instead of zeros.

    Parameters
    ----------
    t_center : tuple, optional
        Center time in terms of 0.0 and 1.0 (duration of signal),
        by default ("uniform", 0.0, 1.0)
    t_width : tuple, optional
        Width of dropped out portion, by default ("const", 0.025)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        t_center: tuple = ("uniform", 0.0, 1.0),
        t_width: tuple = ("const", 0.025),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(t_center=t_center, t_width=t_width, name=name, prob=prob)

    def _transform(self, signal, tmin_s: float, tmax_s: float):
        signal = signal.mask_timesteps(tmin_s=tmin_s, tmax_s=tmax_s, val=0.0)
        mag, phase = signal.magnitude, signal.phase

        mag_r, phase_r = torch.randn_like(mag), torch.randn_like(phase)
        mask = (mag == 0.0) * (phase == 0.0)

        mag[mask] = mag_r[mask]
        phase[mask] = phase_r[mask]

        signal.magnitude = mag
        signal.phase = phase
        return signal


class FrequencyNoise(FrequencyMask):
    """Similar to :py:func:`audiotools.data.transforms.FrequencyMask`, but
    replaces with noise instead of zeros.

    Parameters
    ----------
    f_center : tuple, optional
        Center frequency between 0.0 and 1.0 (Nyquist), by default ("uniform", 0.0, 1.0)
    f_width : tuple, optional
        Width of zero'd out band, by default ("const", 0.1)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        f_center: tuple = ("uniform", 0.0, 1.0),
        f_width: tuple = ("const", 0.1),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(f_center=f_center, f_width=f_width, name=name, prob=prob)

    def _transform(self, signal, fmin_hz: float, fmax_hz: float):
        signal = signal.mask_frequencies(fmin_hz=fmin_hz, fmax_hz=fmax_hz)
        mag, phase = signal.magnitude, signal.phase

        mag_r, phase_r = torch.randn_like(mag), torch.randn_like(phase)
        mask = (mag == 0.0) * (phase == 0.0)

        mag[mask] = mag_r[mask]
        phase[mask] = phase_r[mask]

        signal.magnitude = mag
        signal.phase = phase
        return signal


class SpectralDenoising(Equalizer):
    """Applies denoising algorithm detailed in
    :py:func:`audiotools.ml.layers.spectral_gate.SpectralGate`,
    using a randomly generated noise signal for denoising.

    Parameters
    ----------
    eq_amount : tuple, optional
        Amount of eq to apply to noise signal, by default ("const", 1.0)
    denoise_amount : tuple, optional
        Amount to denoise by, by default ("uniform", 0.8, 1.0)
    nz_volume : float, optional
        Volume of noise to denoise with, by default -40
    n_bands : int, optional
        Number of bands in equalizer, by default 6
    n_freq : int, optional
        Number of frequency bins to smooth by, by default 3
    n_time : int, optional
        Number of time bins to smooth by, by default 5
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        eq_amount: tuple = ("const", 1.0),
        denoise_amount: tuple = ("uniform", 0.8, 1.0),
        nz_volume: float = -40,
        n_bands: int = 6,
        n_freq: int = 3,
        n_time: int = 5,
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(eq_amount=eq_amount, n_bands=n_bands, name=name, prob=prob)

        self.nz_volume = nz_volume
        self.denoise_amount = denoise_amount
        self.spectral_gate = ml.layers.SpectralGate(n_freq, n_time)

    def _transform(self, signal, nz, eq, denoise_amount):
        nz = nz.normalize(self.nz_volume).equalizer(eq)
        self.spectral_gate = self.spectral_gate.to(signal.device)
        signal = self.spectral_gate(signal, nz, denoise_amount)
        return signal

    def _instantiate(self, state: RandomState):
        kwargs = super()._instantiate(state)
        kwargs["denoise_amount"] = util.sample_from_dist(self.denoise_amount, state)
        kwargs["nz"] = AudioSignal(state.randn(22050), 44100)
        return kwargs
