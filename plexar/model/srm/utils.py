# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import platform
import queue
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version

import numpy

assert numpy


def install_audio_packages(pkg_name):
    try:
        version(pkg_name)
    except PackageNotFoundError:
        subprocess.check_call(["pip", "install", pkg_name])


def install_ffmpeg():
    system = platform.system()

    if system == "Linux":
        distro = platform.version()

        if "Ubuntu" or "Debian" in distro:
            subprocess.run(["sudo", "apt", "update"])
            subprocess.run(["sudo", "apt", "install", "ffmpeg"])
        elif "Arch" in distro:
            subprocess.run(["sudo", "pacman", "-S", "ffmpeg"])
        else:
            print("Unsupported Linux distribution.")

    elif system == "Darwin":  # macOS
        subprocess.run(["pip", "install", "ffmpeg-python"])

    elif system == "Windows":
        package_managers = ["choco", "scoop"]

        for manager in package_managers:
            if subprocess.run(["where", manager], capture_output=True).returncode == 0:
                if manager == "choco":
                    subprocess.run(["choco", "install", "ffmpeg"])
                elif manager == "scoop":
                    subprocess.run(["scoop", "install", "ffmpeg"])
                break
        else:
            print("Unsupported package manager (chocolatey or scoop) not found.")

    else:
        print("Unsupported operating system.")


def get_audio_devices() -> str:
    install_audio_packages("sounddevice")
    install_audio_packages("soundfile")
    import sounddevice as sd

    devices = sd.query_devices()
    print("Audio devices:")
    print(devices)
    return input("Please select the audio device you want to record: ")


q: queue.Queue = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def record_unlimited() -> numpy.ndarray:
    import os

    import sounddevice as sd
    import soundfile as sf

    user_device = int(get_audio_devices())
    filename = tempfile.mktemp(prefix="delme_rec_unlimited_", suffix=".wav", dir="")
    try:
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(filename, mode="x", samplerate=48000, channels=1) as file:
            with sd.InputStream(
                samplerate=48000, device=user_device, channels=1, callback=callback
            ):
                print("#" * 80)
                print("press Ctrl+C to stop the recording")
                print("#" * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print("\nRecording finished: " + repr(filename))
    except Exception as e:
        print(type(e).__name__ + ": " + str(e))

    try:
        import ffmpeg
    except ImportError:
        install_ffmpeg()

    try:
        y, _ = (
            ffmpeg.input(os.path.abspath(filename), threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    os.remove(filename)
    return numpy.frombuffer(y, numpy.int16).flatten().astype(numpy.float32) / 32768.0
