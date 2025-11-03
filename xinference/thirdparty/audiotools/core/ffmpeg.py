import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import ffmpy
import numpy as np
import torch


def r128stats(filepath: str, quiet: bool):
    """Takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter.

    Parameters
    ----------
    filepath : str
        Path to compute loudness stats on.
    quiet : bool
        Whether to show FFMPEG output during computation.

    Returns
    -------
    dict
        Dictionary containing loudness stats.
    """
    ffargs = [
        "ffmpeg",
        "-nostats",
        "-i",
        filepath,
        "-filter_complex",
        "ebur128",
        "-f",
        "null",
        "-",
    ]
    if quiet:
        ffargs += ["-hide_banner"]
    proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE, universal_newlines=True)
    stats = proc.communicate()[1]
    summary_index = stats.rfind("Summary:")

    summary_list = stats[summary_index:].split()
    i_lufs = float(summary_list[summary_list.index("I:") + 1])
    i_thresh = float(summary_list[summary_list.index("I:") + 4])
    lra = float(summary_list[summary_list.index("LRA:") + 1])
    lra_thresh = float(summary_list[summary_list.index("LRA:") + 4])
    lra_low = float(summary_list[summary_list.index("low:") + 1])
    lra_high = float(summary_list[summary_list.index("high:") + 1])
    stats_dict = {
        "I": i_lufs,
        "I Threshold": i_thresh,
        "LRA": lra,
        "LRA Threshold": lra_thresh,
        "LRA Low": lra_low,
        "LRA High": lra_high,
    }

    return stats_dict


def ffprobe_offset_and_codec(path: str) -> Tuple[float, str]:
    """Given a path to a file, returns the start time offset and codec of
    the first audio stream.
    """
    ff = ffmpy.FFprobe(
        inputs={path: None},
        global_options="-show_entries format=start_time:stream=duration,start_time,codec_type,codec_name,start_pts,time_base -of json -v quiet",
    )
    streams = json.loads(ff.run(stdout=subprocess.PIPE)[0])["streams"]
    seconds_offset = 0.0
    codec = None

    # Get the offset and codec of the first audio stream we find
    # and return its start time, if it has one.
    for stream in streams:
        if stream["codec_type"] == "audio":
            seconds_offset = stream.get("start_time", 0.0)
            codec = stream.get("codec_name")
            break
    return float(seconds_offset), codec


class FFMPEGMixin:
    _loudness = None

    def ffmpeg_loudness(self, quiet: bool = True):
        """Computes loudness of audio file using FFMPEG.

        Parameters
        ----------
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        torch.Tensor
            Loudness of every item in the batch, computed via
            FFMPEG.
        """
        loudness = []

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            for i in range(self.batch_size):
                self[i].write(f.name)
                loudness_stats = r128stats(f.name, quiet=quiet)
                loudness.append(loudness_stats["I"])

        self._loudness = torch.from_numpy(np.array(loudness)).float()
        return self.loudness()

    def ffmpeg_resample(self, sample_rate: int, quiet: bool = True):
        """Resamples AudioSignal using FFMPEG. More memory-efficient
        than using julius.resample for long audio files.

        Parameters
        ----------
        sample_rate : int
            Sample rate to resample to.
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        AudioSignal
            Resampled AudioSignal.
        """
        from audiotools import AudioSignal

        if sample_rate == self.sample_rate:
            return self

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.write(f.name)
            f_out = f.name.replace("wav", "rs.wav")
            command = f"ffmpeg -i {f.name} -ar {sample_rate} {f_out}"
            if quiet:
                command += " -hide_banner -loglevel error"
            subprocess.check_call(shlex.split(command))
            resampled = AudioSignal(f_out)
            Path.unlink(Path(f_out))
        return resampled

    @classmethod
    def load_from_file_with_ffmpeg(cls, audio_path: str, quiet: bool = True, **kwargs):
        """Loads AudioSignal object after decoding it to a wav file using FFMPEG.
        Useful for loading audio that isn't covered by librosa's loading mechanism. Also
        useful for loading mp3 files, without any offset.

        Parameters
        ----------
        audio_path : str
            Path to load AudioSignal from.
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        AudioSignal
            AudioSignal loaded from file with FFMPEG.
        """
        audio_path = str(audio_path)
        with tempfile.TemporaryDirectory() as d:
            wav_file = str(Path(d) / "extracted.wav")
            padded_wav = str(Path(d) / "padded.wav")

            global_options = "-y"
            if quiet:
                global_options += " -loglevel error"

            ff = ffmpy.FFmpeg(
                inputs={audio_path: None},
                # For inputs that are m4a (and others?), the input audio can
                # have samples that don't match the sample rate. This aresample
                # option forces ffmpeg to read timing information in the source
                # file instead of assuming constant sample rate.
                #
                # This fixes an issue where an input m4a file might be a
                # different length than the output wav file
                outputs={wav_file: "-af aresample=async=1000"},
                global_options=global_options,
            )
            ff.run()

            # We pad the file using the start time offset in case it's an audio
            # stream starting at some offset in a video container.
            pad, codec = ffprobe_offset_and_codec(audio_path)

            # For mp3s, don't pad files with discrepancies less than 0.027s -
            # it's likely due to codec latency. The amount of latency introduced
            # by mp3 is 1152, which is 0.0261 44khz. So we set the threshold
            # here slightly above that.
            # Source: https://lame.sourceforge.io/tech-FAQ.txt.
            if codec == "mp3" and pad < 0.027:
                pad = 0.0
            ff = ffmpy.FFmpeg(
                inputs={wav_file: None},
                outputs={padded_wav: f"-af 'adelay={pad*1000}:all=true'"},
                global_options=global_options,
            )
            ff.run()

            signal = cls(padded_wav, **kwargs)

        return signal
