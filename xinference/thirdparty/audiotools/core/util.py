import csv
import glob
import math
import numbers
import os
import random
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import torch
import torchaudio
from flatten_dict import flatten
from flatten_dict import unflatten


@dataclass
class Info:
    """Shim for torchaudio.info API changes."""

    sample_rate: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate


def info(audio_path: str):
    """Shim for torchaudio.info to make 0.7.2 API match 0.8.0.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    """
    # try default backend first, then fallback to soundfile
    try:
        info = torchaudio.info(str(audio_path))
    except:  # pragma: no cover
        info = torchaudio.backend.soundfile_backend.info(str(audio_path))

    if isinstance(info, tuple):  # pragma: no cover
        signal_info = info[0]
        info = Info(sample_rate=signal_info.rate, num_frames=signal_info.length)
    else:
        info = Info(sample_rate=info.sample_rate, num_frames=info.num_frames)

    return info


def ensure_tensor(
    x: typing.Union[np.ndarray, torch.Tensor, float, int],
    ndim: int = None,
    batch_size: int = None,
):
    """Ensures that the input ``x`` is a tensor of specified
    dimensions and batch size.

    Parameters
    ----------
    x : typing.Union[np.ndarray, torch.Tensor, float, int]
        Data that will become a tensor on its way out.
    ndim : int, optional
        How many dimensions should be in the output, by default None
    batch_size : int, optional
        The batch size of the output, by default None

    Returns
    -------
    torch.Tensor
        Modified version of ``x`` as a tensor.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if ndim is not None:
        assert x.ndim <= ndim
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
    if batch_size is not None:
        if x.shape[0] != batch_size:
            shape = list(x.shape)
            shape[0] = batch_size
            x = x.expand(*shape)
    return x


def _get_value(other):
    from . import AudioSignal

    if isinstance(other, AudioSignal):
        return other.audio_data
    return other


def hz_to_bin(hz: torch.Tensor, n_fft: int, sample_rate: int):
    """Closest frequency bin given a frequency, number
    of bins, and a sampling rate.

    Parameters
    ----------
    hz : torch.Tensor
       Tensor of frequencies in Hz.
    n_fft : int
        Number of FFT bins.
    sample_rate : int
        Sample rate of audio.

    Returns
    -------
    torch.Tensor
        Closest bins to the data.
    """
    shape = hz.shape
    hz = hz.flatten()
    freqs = torch.linspace(0, sample_rate / 2, 2 + n_fft // 2)
    hz[hz > sample_rate / 2] = sample_rate / 2

    closest = (hz[None, :] - freqs[:, None]).abs()
    closest_bins = closest.min(dim=0).indices

    return closest_bins.reshape(*shape)


def random_state(seed: typing.Union[int, np.random.RandomState]):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : typing.Union[int, np.random.RandomState] or None
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState
        Random state object.

    Raises
    ------
    ValueError
        If seed is not valid, an error is thrown.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
        )


def seed(random_seed, set_cudnn=False):
    """
    Seeds all random states with the same random seed
    for reproducibility. Seeds ``numpy``, ``random`` and ``torch``
    random generators.
    For full reproducibility, two further options must be set
    according to the torch documentation:
    https://pytorch.org/docs/stable/notes/randomness.html
    To do this, ``set_cudnn`` must be True. It defaults to
    False, since setting it to True results in a performance
    hit.

    Args:
        random_seed (int): integer corresponding to random seed to
        use.
        set_cudnn (bool): Whether or not to set cudnn into determinstic
        mode and off of benchmark mode. Defaults to False.
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def _close_temp_files(tmpfiles: list):
    """Utility function for creating a context and closing all temporary files
    once the context is exited. For correct functionality, all temporary file
    handles created inside the context must be appended to the ```tmpfiles```
    list.

    This function is taken wholesale from Scaper.

    Parameters
    ----------
    tmpfiles : list
        List of temporary file handles
    """

    def _close():
        for t in tmpfiles:
            try:
                t.close()
                os.unlink(t.name)
            except:
                pass

    try:
        yield
    except:  # pragma: no cover
        _close()
        raise
    _close()


AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]


def find_audio(folder: str, ext: List[str] = AUDIO_EXTENSIONS):
    """Finds all audio files in a directory recursively.
    Returns a list.

    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    """
    folder = Path(folder)
    # Take care of case where user has passed in an audio file directly
    # into one of the calling functions.
    if str(folder).endswith(tuple(ext)):
        # if, however, there's a glob in the path, we need to
        # return the glob, not the file.
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        files += folder.glob(f"**/*{x}")
    return files


def read_sources(
    sources: List[str],
    remove_empty: bool = True,
    relative_path: str = "",
    ext: List[str] = AUDIO_EXTENSIONS,
):
    """Reads audio sources that can either be folders
    full of audio files, or CSV files that contain paths
    to audio files. CSV files that adhere to the expected
    format can be generated by
    :py:func:`audiotools.data.preprocess.create_csv`.

    Parameters
    ----------
    sources : List[str]
        List of audio sources to be converted into a
        list of lists of audio files.
    remove_empty : bool, optional
        Whether or not to remove rows with an empty "path"
        from each CSV file, by default True.

    Returns
    -------
    list
        List of lists of rows of CSV files.
    """
    files = []
    relative_path = Path(relative_path)
    for source in sources:
        source = str(source)
        _files = []
        if source.endswith(".csv"):
            with open(source, "r") as f:
                reader = csv.DictReader(f)
                for x in reader:
                    if remove_empty and x["path"] == "":
                        continue
                    if x["path"] != "":
                        x["path"] = str(relative_path / x["path"])
                    _files.append(x)
        else:
            for x in find_audio(source, ext=ext):
                x = str(relative_path / x)
                _files.append({"path": x})
        files.append(sorted(_files, key=lambda x: x["path"]))
    return files


def choose_from_list_of_lists(
    state: np.random.RandomState, list_of_lists: list, p: float = None
):
    """Choose a single item from a list of lists.

    Parameters
    ----------
    state : np.random.RandomState
        Random state to use when choosing an item.
    list_of_lists : list
        A list of lists from which items will be drawn.
    p : float, optional
        Probabilities of each list, by default None

    Returns
    -------
    typing.Any
        An item from the list of lists.
    """
    source_idx = state.choice(list(range(len(list_of_lists))), p=p)
    item_idx = state.randint(len(list_of_lists[source_idx]))
    return list_of_lists[source_idx][item_idx], source_idx, item_idx


@contextmanager
def chdir(newdir: typing.Union[Path, str]):
    """
    Context manager for switching directories to run a
    function. Useful for when you want to use relative
    paths to different runs.

    Parameters
    ----------
    newdir : typing.Union[Path, str]
        Directory to switch to.
    """
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)


def prepare_batch(batch: typing.Union[dict, list, torch.Tensor], device: str = "cpu"):
    """Moves items in a batch (typically generated by a DataLoader as a list
    or a dict) to the specified device. This works even if dictionaries
    are nested.

    Parameters
    ----------
    batch : typing.Union[dict, list, torch.Tensor]
        Batch, typically generated by a dataloader, that will be moved to
        the device.
    device : str, optional
        Device to move batch to, by default "cpu"

    Returns
    -------
    typing.Union[dict, list, torch.Tensor]
        Batch with all values moved to the specified device.
    """
    if isinstance(batch, dict):
        batch = flatten(batch)
        for key, val in batch.items():
            try:
                batch[key] = val.to(device)
            except:
                pass
        batch = unflatten(batch)
    elif torch.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch


def sample_from_dist(dist_tuple: tuple, state: np.random.RandomState = None):
    """Samples from a distribution defined by a tuple. The first
    item in the tuple is the distribution type, and the rest of the
    items are arguments to that distribution. The distribution function
    is gotten from the ``np.random.RandomState`` object.

    Parameters
    ----------
    dist_tuple : tuple
        Distribution tuple
    state : np.random.RandomState, optional
        Random state, or seed to use, by default None

    Returns
    -------
    typing.Union[float, int, str]
        Draw from the distribution.

    Examples
    --------
    Sample from a uniform distribution:

    >>> dist_tuple = ("uniform", 0, 1)
    >>> sample_from_dist(dist_tuple)

    Sample from a constant distribution:

    >>> dist_tuple = ("const", 0)
    >>> sample_from_dist(dist_tuple)

    Sample from a normal distribution:

    >>> dist_tuple = ("normal", 0, 0.5)
    >>> sample_from_dist(dist_tuple)

    """
    if dist_tuple[0] == "const":
        return dist_tuple[1]
    state = random_state(state)
    dist_fn = getattr(state, dist_tuple[0])
    return dist_fn(*dist_tuple[1:])


def collate(list_of_dicts: list, n_splits: int = None):
    """Collates a list of dictionaries (e.g. as returned by a
    dataloader) into a dictionary with batched values. This routine
    uses the default torch collate function for everything
    except AudioSignal objects, which are handled by the
    :py:func:`audiotools.core.audio_signal.AudioSignal.batch`
    function.

    This function takes n_splits to enable splitting a batch
    into multiple sub-batches for the purposes of gradient accumulation,
    etc.

    Parameters
    ----------
    list_of_dicts : list
        List of dictionaries to be collated.
    n_splits : int
        Number of splits to make when creating the batches (split into
        sub-batches). Useful for things like gradient accumulation.

    Returns
    -------
    dict
        Dictionary containing batched data.
    """

    from . import AudioSignal

    batches = []
    list_len = len(list_of_dicts)

    return_list = False if n_splits is None else True
    n_splits = 1 if n_splits is None else n_splits
    n_items = int(math.ceil(list_len / n_splits))

    for i in range(0, list_len, n_items):
        # Flatten the dictionaries to avoid recursion.
        list_of_dicts_ = [flatten(d) for d in list_of_dicts[i : i + n_items]]
        dict_of_lists = {
            k: [dic[k] for dic in list_of_dicts_] for k in list_of_dicts_[0]
        }

        batch = {}
        for k, v in dict_of_lists.items():
            if isinstance(v, list):
                if all(isinstance(s, AudioSignal) for s in v):
                    batch[k] = AudioSignal.batch(v, pad_signals=True)
                else:
                    # Borrow the default collate fn from torch.
                    batch[k] = torch.utils.data._utils.collate.default_collate(v)
        batches.append(unflatten(batch))

    batches = batches[0] if not return_list else batches
    return batches


BASE_SIZE = 864
DEFAULT_FIG_SIZE = (9, 3)


def format_figure(
    fig_size: tuple = None,
    title: str = None,
    fig=None,
    format_axes: bool = True,
    format: bool = True,
    font_color: str = "white",
):
    """Prettifies the spectrogram and waveform plots. A title
    can be inset into the top right corner, and the axes can be
    inset into the figure, allowing the data to take up the entire
    image. Used in

    - :py:func:`audiotools.core.display.DisplayMixin.specshow`
    - :py:func:`audiotools.core.display.DisplayMixin.waveplot`
    - :py:func:`audiotools.core.display.DisplayMixin.wavespec`

    Parameters
    ----------
    fig_size : tuple, optional
        Size of figure, by default (9, 3)
    title : str, optional
        Title to inset in top right, by default None
    fig : matplotlib.figure.Figure, optional
        Figure object, if None ``plt.gcf()`` will be used, by default None
    format_axes : bool, optional
        Format the axes to be inside the figure, by default True
    format : bool, optional
        This formatting can be skipped entirely by passing ``format=False``
        to any of the plotting functions that use this formater, by default True
    font_color : str, optional
        Color of font of axes, by default "white"
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if fig_size is None:
        fig_size = DEFAULT_FIG_SIZE
    if not format:
        return
    if fig is None:
        fig = plt.gcf()
    fig.set_size_inches(*fig_size)
    axs = fig.axes

    pixels = (fig.get_size_inches() * fig.dpi)[0]
    font_scale = pixels / BASE_SIZE

    if format_axes:
        axs = fig.axes

        for ax in axs:
            ymin, _ = ax.get_ylim()
            xmin, _ = ax.get_xlim()

            ticks = ax.get_yticks()
            for t in ticks[2:-1]:
                t = axs[0].annotate(
                    f"{(t / 1000):2.1f}k",
                    xy=(xmin, t),
                    xycoords="data",
                    xytext=(5, -5),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color=font_color,
                    fontsize=12 * font_scale,
                    alpha=0.75,
                )

            ticks = ax.get_xticks()[2:]
            for t in ticks[:-1]:
                t = axs[0].annotate(
                    f"{t:2.1f}s",
                    xy=(t, ymin),
                    xycoords="data",
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=font_color,
                    fontsize=12 * font_scale,
                    alpha=0.75,
                )

            ax.margins(0, 0)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if title is not None:
        t = axs[0].annotate(
            title,
            xy=(1, 1),
            xycoords="axes fraction",
            fontsize=20 * font_scale,
            xytext=(-5, -5),
            textcoords="offset points",
            ha="right",
            va="top",
            color="white",
        )
        t.set_bbox(dict(facecolor="black", alpha=0.5, edgecolor="black"))


def generate_chord_dataset(
    max_voices: int = 8,
    sample_rate: int = 44100,
    num_items: int = 5,
    duration: float = 1.0,
    min_note: str = "C2",
    max_note: str = "C6",
    output_dir: Path = "chords",
):
    """
    Generates a toy multitrack dataset of chords, synthesized from sine waves.


    Parameters
    ----------
    max_voices : int, optional
        Maximum number of voices in a chord, by default 8
    sample_rate : int, optional
        Sample rate of audio, by default 44100
    num_items : int, optional
        Number of items to generate, by default 5
    duration : float, optional
        Duration of each item, by default 1.0
    min_note : str, optional
        Minimum note in the dataset, by default "C2"
    max_note : str, optional
        Maximum note in the dataset, by default "C6"
    output_dir : Path, optional
        Directory to save the dataset, by default "chords"

    """
    import librosa
    from . import AudioSignal
    from ..data.preprocess import create_csv

    min_midi = librosa.note_to_midi(min_note)
    max_midi = librosa.note_to_midi(max_note)

    tracks = []
    for idx in range(num_items):
        track = {}
        # figure out how many voices to put in this track
        num_voices = random.randint(1, max_voices)
        for voice_idx in range(num_voices):
            # choose some random params
            midinote = random.randint(min_midi, max_midi)
            dur = random.uniform(0.85 * duration, duration)

            sig = AudioSignal.wave(
                frequency=librosa.midi_to_hz(midinote),
                duration=dur,
                sample_rate=sample_rate,
                shape="sine",
            )
            track[f"voice_{voice_idx}"] = sig
        tracks.append(track)

    # save the tracks to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for idx, track in enumerate(tracks):
        track_dir = output_dir / f"track_{idx}"
        track_dir.mkdir(exist_ok=True)
        for voice_name, sig in track.items():
            sig.write(track_dir / f"{voice_name}.wav")

    all_voices = list(set([k for track in tracks for k in track.keys()]))
    voice_lists = {voice: [] for voice in all_voices}
    for track in tracks:
        for voice_name in all_voices:
            if voice_name in track:
                voice_lists[voice_name].append(track[voice_name].path_to_file)
            else:
                voice_lists[voice_name].append("")

    for voice_name, paths in voice_lists.items():
        create_csv(paths, output_dir / f"{voice_name}.csv", loudness=True)

    return output_dir
