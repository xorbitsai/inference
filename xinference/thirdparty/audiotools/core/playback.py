"""
These are utilities that allow one to embed an AudioSignal
as a playable object in a Jupyter notebook, or to play audio from
the terminal, etc.
"""  # fmt: skip
import base64
import io
import random
import string
import subprocess
from tempfile import NamedTemporaryFile

import importlib_resources as pkg_resources

from . import templates
from .util import _close_temp_files
from .util import format_figure

headers = pkg_resources.files(templates).joinpath("headers.html").read_text()
widget = pkg_resources.files(templates).joinpath("widget.html").read_text()

DEFAULT_EXTENSION = ".wav"


def _check_imports():  # pragma: no cover
    try:
        import ffmpy
    except:
        ffmpy = False

    try:
        import IPython
    except:
        raise ImportError("IPython must be installed in order to use this function!")
    return ffmpy, IPython


class PlayMixin:
    def embed(self, ext: str = None, display: bool = True, return_html: bool = False):
        """Embeds audio as a playable audio embed in a notebook, or HTML
        document, etc.

        Parameters
        ----------
        ext : str, optional
            Extension to use when saving the audio, by default ".wav"
        display : bool, optional
            This controls whether or not to display the audio when called. This
            is used when the embed is the last line in a Jupyter cell, to prevent
            the audio from being embedded twice, by default True
        return_html : bool, optional
            Whether to return the data wrapped in an HTML audio element, by default False

        Returns
        -------
        str
            Either the element for display, or the HTML string of it.
        """
        if ext is None:
            ext = DEFAULT_EXTENSION
        ext = f".{ext}" if not ext.startswith(".") else ext
        ffmpy, IPython = _check_imports()
        sr = self.sample_rate
        tmpfiles = []

        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(mode="w+", suffix=".wav", delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name)
            if ext != ".wav" and ffmpy:
                tmp_converted = NamedTemporaryFile(mode="w+", suffix=ext, delete=False)
                tmpfiles.append(tmp_wav)
                ff = ffmpy.FFmpeg(
                    inputs={tmp_wav.name: None},
                    outputs={
                        tmp_converted.name: "-write_xing 0 -codec:a libmp3lame -b:a 128k -y -hide_banner -loglevel error"
                    },
                )
                ff.run()
            else:
                tmp_converted = tmp_wav

            audio_element = IPython.display.Audio(data=tmp_converted.name, rate=sr)
            if display:
                IPython.display.display(audio_element)

        if return_html:
            audio_element = (
                f"<audio "
                f"  controls "
                f"  src='{audio_element.src_attr()}'> "
                f"</audio> "
            )
        return audio_element

    def widget(
        self,
        title: str = None,
        ext: str = ".wav",
        add_headers: bool = True,
        player_width: str = "100%",
        margin: str = "10px",
        plot_fn: str = "specshow",
        return_html: bool = False,
        **kwargs,
    ):
        """Creates a playable widget with spectrogram. Inspired (heavily) by
        https://sjvasquez.github.io/blog/melnet/.

        Parameters
        ----------
        title : str, optional
            Title of plot, placed in upper right of top-most axis.
        ext : str, optional
            Extension for embedding, by default ".mp3"
        add_headers : bool, optional
            Whether or not to add headers (use for first embed, False for later embeds), by default True
        player_width : str, optional
            Width of the player, as a string in a CSS rule, by default "100%"
        margin : str, optional
            Margin on all sides of player, by default "10px"
        plot_fn : function, optional
            Plotting function to use (by default self.specshow).
        return_html : bool, optional
            Whether to return the data wrapped in an HTML audio element, by default False
        kwargs : dict, optional
            Keyword arguments to plot_fn (by default self.specshow).

        Returns
        -------
        HTML
            HTML object.
        """
        import matplotlib.pyplot as plt

        def _save_fig_to_tag():
            buffer = io.BytesIO()

            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close()

            buffer.seek(0)
            data_uri = base64.b64encode(buffer.read()).decode("ascii")
            tag = "data:image/png;base64,{0}".format(data_uri)

            return tag

        _, IPython = _check_imports()

        header_html = ""

        if add_headers:
            header_html = headers.replace("PLAYER_WIDTH", str(player_width))
            header_html = header_html.replace("MARGIN", str(margin))
            IPython.display.display(IPython.display.HTML(header_html))

        widget_html = widget
        if isinstance(plot_fn, str):
            plot_fn = getattr(self, plot_fn)
            kwargs["title"] = title
        plot_fn(**kwargs)

        fig = plt.gcf()
        pixels = fig.get_size_inches() * fig.dpi

        tag = _save_fig_to_tag()

        # Make the source image for the levels
        self.specshow()
        format_figure((12, 1.5))
        levels_tag = _save_fig_to_tag()

        player_id = "".join(random.choice(string.ascii_uppercase) for _ in range(10))

        audio_elem = self.embed(ext=ext, display=False)
        widget_html = widget_html.replace("AUDIO_SRC", audio_elem.src_attr())
        widget_html = widget_html.replace("IMAGE_SRC", tag)
        widget_html = widget_html.replace("LEVELS_SRC", levels_tag)
        widget_html = widget_html.replace("PLAYER_ID", player_id)

        # Calculate width/height of figure based on figure size.
        widget_html = widget_html.replace("PADDING_AMOUNT", f"{int(pixels[1])}px")
        widget_html = widget_html.replace("MAX_WIDTH", f"{int(pixels[0])}px")

        IPython.display.display(IPython.display.HTML(widget_html))

        if return_html:
            html = header_html if add_headers else ""
            html += widget_html
            return html

    def play(self):
        """
        Plays an audio signal if ffplay from the ffmpeg suite of tools is installed.
        Otherwise, will fail. The audio signal is written to a temporary file
        and then played with ffplay.
        """
        tmpfiles = []
        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(suffix=".wav", delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name)
            print(self)
            subprocess.call(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    tmp_wav.name,
                ]
            )
        return self


if __name__ == "__main__":  # pragma: no cover
    from audiotools import AudioSignal

    signal = AudioSignal(
        "tests/audio/spk/f10_script4_produced.mp3", offset=5, duration=5
    )

    wave_html = signal.widget(
        "Waveform",
        plot_fn="waveplot",
        return_html=True,
    )

    spec_html = signal.widget("Spectrogram", return_html=True, add_headers=False)

    combined_html = signal.widget(
        "Waveform + spectrogram",
        plot_fn="wavespec",
        return_html=True,
        add_headers=False,
    )

    signal.low_pass(8000)
    lowpass_html = signal.widget(
        "Lowpassed audio",
        plot_fn="wavespec",
        return_html=True,
        add_headers=False,
    )

    with open("/tmp/index.html", "w") as f:
        f.write(wave_html)
        f.write(spec_html)
        f.write(combined_html)
        f.write(lowpass_html)
