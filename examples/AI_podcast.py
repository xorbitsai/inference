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

import argparse
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
import warnings
from typing import List

warnings.filterwarnings("ignore")


try:
    import ffmpeg
except ImportError:
    raise ImportError(
        "Failed to import ffmpeg, please install ffmpeg with `brew install ffmpeg & pip install"
        " ffmpeg`"
    )

try:
    import sounddevice as sd
except ImportError:
    raise ImportError(
        "Failed to import sounddevice, please install sounddevice with `pip install sounddevice`"
    )

try:
    import soundfile as sf
except:
    raise ImportError(
        "Failed to import soundfile, please install soundfile with `pip install soundfile`"
    )

try:
    import emoji
except ImportError:
    raise ImportError(
        "Failed to import emoji, please check the "
        "correct package at https://pypi.org/project/emoji/"
    )

try:
    import numpy
except ImportError:
    raise ImportError(
        "Failed to import numpy, please check the "
        "correct package at https://pypi.org/project/numpy/1.24.1/"
    )

try:
    import whisper
except ImportError:
    raise ImportError(
        "Failed to import whisper, please check the "
        "correct package at https://pypi.org/project/openai-whisper/"
    )

try:
    from xinference.client import Client
    from xinference.types import ChatCompletionMessage
except ImportError:
    raise ImportError(
        "Failed to import xinference, please check the "
        "correct package at https://pypi.org/project/xinference/"
    )

# ------------------------------------- global variable initialization ---------------------------------------------- #
logger = logging.getLogger(__name__)
# global variable to store the audio device choices.
audio_devices = "-1"

# ----------------------------------------- decorator libraries ----------------------------------------------------- #
emoji_man = "\U0001F9D4"
emoji_women = emoji.emojize(":woman:")
emoji_system = emoji.emojize(":robot:")
emoji_user = emoji.emojize(":supervillain:")
emoji_speaking = emoji.emojize(":speaking_head:")
emoji_sparkiles = emoji.emojize(":sparkles:")
emoji_jack_o_lantern = emoji.emojize(":jack-o-lantern:")
emoji_microphone = emoji.emojize(":studio_microphone:")
emoji_rocket = emoji.emojize(":rocket:")


# --------------------------------- supplemented util to get the record --------------------------------------------- #
def get_audio_devices() -> str:
    global audio_devices

    if audio_devices != "-1":
        return str(audio_devices)

    devices = sd.query_devices()
    print("\n")
    print(emoji_microphone, end="")
    print("  Audio devices:")
    print(devices)
    print(emoji_microphone, end="")
    audio_devices = input("  Please select the audio device you want to record: ")
    return audio_devices


q: queue.Queue = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# function to take audio input and transcript it into text-file.
def record_unlimited() -> numpy.ndarray:
    user_device = int(get_audio_devices())
    print("")
    terminal_size = os.get_terminal_size()
    print("-" * terminal_size.columns)
    print(emoji_speaking, end="")
    input("  Press Enter to start talking and press Ctrl+C to stop the recording:")
    filename = tempfile.mktemp(prefix="delme_rec_unlimited_", suffix=".wav", dir="")
    try:
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(filename, mode="x", samplerate=48000, channels=1) as file:
            with sd.InputStream(
                samplerate=48000, device=user_device, channels=1, callback=callback
            ):
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(type(e).__name__ + ": " + str(e))

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


def format_prompt(model, audio_input) -> str:
    # the second parameters of transcribe enable us to define the language we are speaking.
    return model.transcribe(audio_input)["text"]


# transcript the generated chatbot word to audio output so the user will hear the result.
def text_to_audio(response, voice_id):
    # for audio output, we apply the mac initiated "say" command to provide. For Windows users, if
    # you want audio output, you can try on pyttsx3 or gtts package to see their functionality!

    text = response
    if voice_id == "Bob":
        voice = "Daniel"
    elif voice_id == "Alice":
        voice = "Karen"
    # anything not belongs to alice or bob are said by system voice.
    else:
        voice = "Moira"
    # Execute the "say" command and wait the command to be completed.
    process = subprocess.Popen(["say", "-v", voice, text])
    process.wait()


def chat_with_bot(
    format_input, chat_history, alice_or_bob_state, system_prompt, model_ref
):
    completion = model_ref.chat(
        prompt=format_input,
        system_prompt=system_prompt,
        chat_history=chat_history,
        generate_config={"max_tokens": 1024},
    )

    if alice_or_bob_state == "Alice":
        print(emoji_women, end="")
        print(" Alice:", end="")
    else:
        print(emoji_man, end="")
        print(" Bob:", end="")

    chat_history: List["ChatCompletionMessage"] = []

    content = completion["choices"][0]["message"]["content"]
    print(content)

    chat_history.append(ChatCompletionMessage(role="assistant", content=content))

    return content


# ---------------------------------------- The program will run from below: ------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="Xinference endpoint, required",
        required=True,
    )
    args = parser.parse_args()

    endpoint = args.endpoint

    client = Client(endpoint)

    model_a = "vicuna-v1.3"
    print(
        f"{emoji_rocket} Launching model {model_a}. The initial download of the model may require a certain"
        f" amount of time."
    )
    model_a_uid = client.launch_model(
        model_name=model_a,
        model_format="ggmlv3",
        model_size_in_billions=7,
        quantization="q4_0",
        n_ctx=2048,
    )
    model_a_ref = client.get_model(model_a_uid)

    model_b = "wizardlm-v1.0"
    print(
        f"{emoji_rocket} Launching model {model_b}. The initial download of the model may require a certain"
        f" amount of time."
    )
    model_b_uid = client.launch_model(
        model_name=model_b,
        model_format="ggmlv3",
        model_size_in_billions=7,
        quantization="q4_0",
        n_ctx=2048,
    )
    model_b_ref = client.get_model(model_b_uid)

    # ---------- program finally start! ------------ #
    chat_history = []
    alice_or_bob_state = "0"
    print("")
    print(emoji_jack_o_lantern, end="")
    print(" Welcome to the Xorbits inference chatroom ", end="")
    print(emoji_jack_o_lantern)
    print(emoji_sparkiles, end="")
    print(
        " Say something with 'exit', 'quit', 'bye', or 'see you' to leave the chatroom ",
        end="",
    )
    print(emoji_sparkiles)

    # receive the username.
    print("")
    print(emoji_system, end="")
    welcome_prompt = ": Please tell us who is attending the conversation today: "
    text_to_audio(welcome_prompt, "0")
    username = input(welcome_prompt)

    # define names for the chatbots and create welcome message for chat-room.
    system_prompt_alice = (
        "This is a conversation between a Human user and two AI assistants. "
        "The first AI assistant is called Alice, and the second AI assistant is called Bob."
        f"The Human User is called {username}"
    )
    system_prompt_bob = system_prompt_alice

    # we can change the scale of the model here, the bigger the model, the higher the accuracy.
    model = whisper.load_model("medium")

    welcome_prompt2 = (
        f": Nice to meet you, {username}. Hope you will enjoy the conversation with our AI agents"
        f" at Xorbits inference chatroom. Later, our system will guide you on when to speak "
        f"with your microphone, please follow the steps correctly. "
        f"We hope you have a pleasant journey with our AI agents. "
    )

    print("")
    print(emoji_system, end="")
    print(welcome_prompt2)
    text_to_audio(welcome_prompt2, "0")

    while True:
        audio_input = record_unlimited()

        start = time.time()
        format_input = format_prompt(model, audio_input)
        logger.info(f"Time spent on transcribing: {time.time() - start}")

        # set up the separation between each chat block.
        print("")
        print(emoji_user, end="")
        print(f" {username}:", end="")
        print(format_input)

        # for un-natural exit audio inputs.
        if "exit" in format_input.lower() or "quit" in format_input.lower():
            break

        # for natural exit, both bot is expected to send greeting message.
        if "bye" in format_input.lower() or "see you" in format_input.lower():
            alice_or_bob_state = "Alice"
            content_alice = f": Nice Chat with you, {username}. Have a Nice Day!"
            print(emoji_women, end="")
            print(content_alice)
            text_to_audio(content_alice, alice_or_bob_state)
            alice_or_bob_state = "Bob"
            content_bob = (
                f": It's my honor to chat with you, {username}. Enjoy your time!"
            )
            print(emoji_man, end="")
            print(content_bob)
            text_to_audio(content_bob, alice_or_bob_state)
            break

        chat_history.append(ChatCompletionMessage(role="user", content=format_input))

        system_prompt = system_prompt_alice
        # we choose to set Alice to default.
        model_ref = model_a_ref

        # check whether alice and bob are both in the prompt and their position.
        def check_word_order(string, first_word, second_word) -> int:
            # split the string into words, and exclude punctuation.
            words = re.findall(r"\b\w+\b", string)

            if first_word in words and second_word in words:
                first_position = words.index(first_word)
                second_position = words.index(second_word)

                if first_position < second_position:
                    return 1
                if first_position > second_position:
                    return 2

            # neither of the words is present in the string
            return -1

        # if the user says alice first, we assume that the user want alice.
        if check_word_order(format_input.lower(), "alice", "bob") == 1:
            alice_or_bob_state = "Alice"
            system_prompt = system_prompt_alice
            model_ref = model_a_ref
        # if bob is first, then we assume the user want bob.
        elif check_word_order(format_input.lower(), "alice", "bob") == 2:
            alice_or_bob_state = "Bob"
            system_prompt = system_prompt_bob
            model_ref = model_b_ref
        # if not both of them presents, user says he wants to talk with Alice, we assign Alice,
        # otherwise we assign Bob.
        else:
            if "alice" in format_input.lower():
                alice_or_bob_state = "Alice"
                system_prompt = system_prompt_alice
                model_ref = model_a_ref
            elif "bob" in format_input.lower():
                alice_or_bob_state = "Bob"
                system_prompt = system_prompt_bob
                model_ref = model_b_ref
            # if both of them are not presents, if we don't have any assignment to agents,
            # we shall tell the user to do so.
            else:
                if alice_or_bob_state == "0":
                    tips = "Please feel free to call our agents' names and they are ready to chat with you!"
                    print(emoji_system, end="")
                    print(tips)
                    text_to_audio(tips, "0")
                    continue
        content = chat_with_bot(
            format_input, chat_history, alice_or_bob_state, system_prompt, model_ref
        )

        text_to_audio(content, alice_or_bob_state)

    del chat_history
    bye_msg1 = (
        ": Thank you for chatting with our extraordinary AI agents from Xprobe.inc. "
    )
    bye_msg2 = (
        ": We will keep helping more people in need and make the world a better place!"
    )
    print("\n")
    print(emoji_system, end="")
    print(bye_msg1)
    print(emoji_system, end="")
    print(bye_msg2)
    text_to_audio(bye_msg1, "0")
    text_to_audio(bye_msg2, "0")
