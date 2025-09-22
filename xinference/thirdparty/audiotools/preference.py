##############################################################
### Tools for creating preference tests (MUSHRA, ABX, etc) ###
##############################################################
import copy
import csv
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List

import gradio as gr

from audiotools.core.util import find_audio

################################################################
### Logic for audio player, and adding audio / play buttons. ###
################################################################

WAVESURFER = """<div id="waveform"></div><div id="wave-timeline"></div>"""

CUSTOM_CSS = """
.gradio-container {
    max-width: 840px !important;
}
region.wavesurfer-region:before {
    content: attr(data-region-label);
}

block {
    min-width: 0 !important;
}

#wave-timeline {
    background-color: rgba(0, 0, 0, 0.8);
}

.head.svelte-1cl284s {
    display: none;
}
"""

load_wavesurfer_js = """
function load_wavesurfer() {
    function load_script(url) {
        const script = document.createElement('script');
        script.src = url;
        document.body.appendChild(script);

        return new Promise((res, rej) => {
            script.onload = function() {
                res();
            }
            script.onerror = function () {
                rej();
            }
        });
    }

    function create_wavesurfer() {
        var options = {
            container: '#waveform',
            waveColor: '#F2F2F2', // Set a darker wave color
            progressColor: 'white', // Set a slightly lighter progress color
            loaderColor: 'white', // Set a slightly lighter loader color
            cursorColor: 'black', // Set a slightly lighter cursor color
            backgroundColor: '#00AAFF', // Set a black background color
            barWidth: 4,
            barRadius: 3,
            barHeight: 1, // the height of the wave
            plugins: [
                WaveSurfer.regions.create({
                    regionsMinLength: 0.0,
                    dragSelection: {
                        slop: 5
                    },
                    color: 'hsla(200, 50%, 70%, 0.4)',
                }),
                 WaveSurfer.timeline.create({
                    container: "#wave-timeline",
                    primaryLabelInterval: 5.0,
                    secondaryLabelInterval: 1.0,
                    primaryFontColor: '#F2F2F2',
                    secondaryFontColor: '#F2F2F2',
                }),
            ]
        };
        wavesurfer = WaveSurfer.create(options);
        wavesurfer.on('region-created', region => {
            wavesurfer.regions.clear();
        });
        wavesurfer.on('finish', function () {
            var loop =  document.getElementById("loop-button").textContent.includes("ON");
            if (loop) {
                wavesurfer.play();
            }
            else {
                var button_elements = document.getElementsByClassName('playpause')
                var buttons = Array.from(button_elements);

                for (let j = 0; j < buttons.length; j++) {
                    buttons[j].classList.remove("primary");
                    buttons[j].classList.add("secondary");
                    buttons[j].textContent = buttons[j].textContent.replace("Stop", "Play")
                }
            }
        });

        wavesurfer.on('region-out', function () {
            var loop =  document.getElementById("loop-button").textContent.includes("ON");
            if (!loop) {
                var button_elements = document.getElementsByClassName('playpause')
                var buttons = Array.from(button_elements);

                for (let j = 0; j < buttons.length; j++) {
                    buttons[j].classList.remove("primary");
                    buttons[j].classList.add("secondary");
                    buttons[j].textContent = buttons[j].textContent.replace("Stop", "Play")
                }
                wavesurfer.pause();
            }
        });

        console.log("Created WaveSurfer object.")
    }

    load_script('https://unpkg.com/wavesurfer.js@6.6.4')
        .then(() => {
            load_script("https://unpkg.com/wavesurfer.js@6.6.4/dist/plugin/wavesurfer.timeline.min.js")
                .then(() => {
                    load_script('https://unpkg.com/wavesurfer.js@6.6.4/dist/plugin/wavesurfer.regions.min.js')
                        .then(() => {
                            console.log("Loaded regions");
                            create_wavesurfer();
                            document.getElementById("start-survey").click();
                        })
                })
        });
}
"""

play = lambda i: """
function play() {
    var audio_elements = document.getElementsByTagName('audio');
    var button_elements = document.getElementsByClassName('playpause')

    var audio_array = Array.from(audio_elements);
    var buttons = Array.from(button_elements);

    var src_link = audio_array[{i}].getAttribute("src");
    console.log(src_link);

    var loop = document.getElementById("loop-button").textContent.includes("ON");
    var playing = buttons[{i}].textContent.includes("Stop");

    for (let j = 0; j < buttons.length; j++) {
        if (j != {i} || playing) {
            buttons[j].classList.remove("primary");
            buttons[j].classList.add("secondary");
            buttons[j].textContent = buttons[j].textContent.replace("Stop", "Play")
        }
        else {
            buttons[j].classList.remove("secondary");
            buttons[j].classList.add("primary");
            buttons[j].textContent = buttons[j].textContent.replace("Play", "Stop")
        }
    }

    if (playing) {
        wavesurfer.pause();
        wavesurfer.seekTo(0.0);
    }
    else {
        wavesurfer.load(src_link);
        wavesurfer.on('ready', function () {
            var region = Object.values(wavesurfer.regions.list)[0];

            if (region != null) {
                region.loop = loop;
                region.play();
            } else {
                wavesurfer.play();
            }
        });
    }
}
""".replace(
    "{i}", str(i)
)

clear_regions = """
function clear_regions() {
    wavesurfer.clearRegions();
}
"""

reset_player = """
function reset_player() {
    wavesurfer.clearRegions();
    wavesurfer.pause();
    wavesurfer.seekTo(0.0);

    var button_elements = document.getElementsByClassName('playpause')
    var buttons = Array.from(button_elements);

    for (let j = 0; j < buttons.length; j++) {
        buttons[j].classList.remove("primary");
        buttons[j].classList.add("secondary");
        buttons[j].textContent = buttons[j].textContent.replace("Stop", "Play")
    }
}
"""

loop_region = """
function loop_region() {
    var element = document.getElementById("loop-button");
    var loop = element.textContent.includes("OFF");
    console.log(loop);

    try {
        var region = Object.values(wavesurfer.regions.list)[0];
        region.loop = loop;
    } catch {}

    if (loop) {
        element.classList.remove("secondary");
        element.classList.add("primary");
        element.textContent = "Looping ON";
    } else {
        element.classList.remove("primary");
        element.classList.add("secondary");
        element.textContent = "Looping OFF";
    }
}
"""


class Player:
    def __init__(self, app):
        self.app = app

        self.app.load(_js=load_wavesurfer_js)
        self.app.css = CUSTOM_CSS

        self.wavs = []
        self.position = 0

    def create(self):
        gr.HTML(WAVESURFER)
        gr.Markdown(
            "Click and drag on the waveform above to select a region for playback. "
            "Once created, the region can be moved around and resized. "
            "Clear the regions using the button below. Hit play on one of the buttons below to start!"
        )

        with gr.Row():
            clear = gr.Button("Clear region")
            loop = gr.Button("Looping OFF", elem_id="loop-button")

            loop.click(None, _js=loop_region)
            clear.click(None, _js=clear_regions)

        gr.HTML("<hr>")

    def add(self, name: str = "Play"):
        i = self.position
        self.wavs.append(
            {
                "audio": gr.Audio(visible=False),
                "button": gr.Button(name, elem_classes=["playpause"]),
                "position": i,
            }
        )
        self.wavs[-1]["button"].click(None, _js=play(i))
        self.position += 1
        return self.wavs[-1]

    def to_list(self):
        return [x["audio"] for x in self.wavs]


############################################################
### Keeping track of users, and CSS for the progress bar ###
############################################################

load_tracker = lambda name: """
function load_name() {
    function setCookie(name, value, exp_days) {
        var d = new Date();
        d.setTime(d.getTime() + (exp_days*24*60*60*1000));
        var expires = "expires=" + d.toGMTString();
        document.cookie = name + "=" + value + ";" + expires + ";path=/";
    }

    function getCookie(name) {
        var cname = name + "=";
        var decodedCookie = decodeURIComponent(document.cookie);
        var ca = decodedCookie.split(';');
        for(var i = 0; i < ca.length; i++){
            var c = ca[i];
            while(c.charAt(0) == ' '){
                c = c.substring(1);
            }
            if(c.indexOf(cname) == 0){
                return c.substring(cname.length, c.length);
            }
        }
        return "";
    }

    name = getCookie("{name}");
    if (name == "") {
        name = Math.random().toString(36).slice(2);
        console.log(name);
        setCookie("name", name, 30);
    }
    name = getCookie("{name}");
    return name;
}
""".replace(
    "{name}", name
)

# Progress bar

progress_template = """
<!DOCTYPE html>
<html>
  <head>
    <title>Progress Bar</title>
    <style>
      .progress-bar {
        background-color: #ddd;
        border-radius: 4px;
        height: 30px;
        width: 100%;
        position: relative;
      }

      .progress {
        background-color: #00AAFF;
        border-radius: 4px;
        height: 100%;
        width: {PROGRESS}%; /* Change this value to control the progress */
      }

      .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 18px;
        font-family: Arial, sans-serif;
        font-weight: bold;
        color: #333 !important;
        text-shadow: 1px 1px #fff;
      }
    </style>
  </head>
  <body>
    <div class="progress-bar">
      <div class="progress"></div>
      <div class="progress-text">{TEXT}</div>
    </div>
  </body>
</html>
"""


def create_tracker(app, cookie_name="name"):
    user = gr.Text(label="user", interactive=True, visible=False, elem_id="user")
    app.load(_js=load_tracker(cookie_name), outputs=user)
    return user


#################################################################
### CSS and HTML for labeling sliders for both ABX and MUSHRA ###
#################################################################

slider_abx = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>Labels Example</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }

      .labels-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        height: 40px;
        padding: 0px 12px 0px;
      }

      .label {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 33%;
        height: 100%;
        font-weight: bold;
        text-transform: uppercase;
        padding: 10px;
        font-family: Arial, sans-serif;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 1px;
        line-height: 1.5;
      }

      .label-a {
        background-color: #00AAFF;
        color: #333 !important;
      }

      .label-tie {
        background-color: #f97316;
        color: #333 !important;
      }

      .label-b {
        background-color: #00AAFF;
        color: #333 !important;
      }
    </style>
  </head>
  <body>
    <div class="labels-container">
      <div class="label label-a">Prefer A</div>
      <div class="label label-tie">Toss-up</div>
      <div class="label label-b">Prefer B</div>
    </div>
  </body>
</html>
"""

slider_mushra = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>Labels Example</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }

      .labels-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        height: 30px;
        padding: 10px;
      }

      .label {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 20%;
        height: 100%;
        font-weight: bold;
        text-transform: uppercase;
        padding: 10px;
        font-family: Arial, sans-serif;
        font-size: 13.5px;
        font-weight: 700;
        line-height: 1.5;
      }

      .label-bad {
        background-color: #ff5555;
        color: #333 !important;
      }

      .label-poor {
        background-color: #ffa500;
        color: #333 !important;
      }

      .label-fair {
        background-color: #ffd700;
        color: #333 !important;
      }

      .label-good {
        background-color: #97d997;
        color: #333 !important;
      }

      .label-excellent {
        background-color: #04c822;
        color: #333 !important;
      }
    </style>
  </head>
  <body>
    <div class="labels-container">
      <div class="label label-bad">bad</div>
      <div class="label label-poor">poor</div>
      <div class="label label-fair">fair</div>
      <div class="label label-good">good</div>
      <div class="label label-excellent">excellent</div>
    </div>
  </body>
</html>
"""

#########################################################
### Handling loading audio and tracking session state ###
#########################################################


class Samples:
    def __init__(self, folder: str, shuffle: bool = True, n_samples: int = None):
        files = find_audio(folder)
        samples = defaultdict(lambda: defaultdict())

        for f in files:
            condition = f.parent.stem
            samples[f.name][condition] = f

        self.samples = samples
        self.names = list(samples.keys())
        self.filtered = False
        self.current = 0

        if shuffle:
            random.shuffle(self.names)

        self.n_samples = len(self.names) if n_samples is None else n_samples

    def get_updates(self, idx, order):
        key = self.names[idx]
        return [gr.update(value=str(self.samples[key][o])) for o in order]

    def progress(self):
        try:
            pct = self.current / len(self) * 100
        except:  # pragma: no cover
            pct = 100
        text = f"On {self.current} / {len(self)} samples"
        pbar = (
            copy.copy(progress_template)
            .replace("{PROGRESS}", str(pct))
            .replace("{TEXT}", str(text))
        )
        return gr.update(value=pbar)

    def __len__(self):
        return self.n_samples

    def filter_completed(self, user, save_path):
        if not self.filtered:
            done = []
            if Path(save_path).exists():
                with open(save_path, "r") as f:
                    reader = csv.DictReader(f)
                    done = [r["sample"] for r in reader if r["user"] == user]
            self.names = [k for k in self.names if k not in done]
            self.names = self.names[: self.n_samples]
            self.filtered = True  # Avoid filtering more than once per session.

    def get_next_sample(self, reference, conditions):
        random.shuffle(conditions)
        if reference is not None:
            self.order = [reference] + conditions
        else:
            self.order = conditions

        try:
            updates = self.get_updates(self.current, self.order)
            self.current += 1
            done = gr.update(interactive=True)
            pbar = self.progress()
        except:
            traceback.print_exc()
            updates = [gr.update() for _ in range(len(self.order))]
            done = gr.update(value="No more samples!", interactive=False)
            self.current = len(self)
            pbar = self.progress()

        return updates, done, pbar


def save_result(result, save_path):
    with open(save_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(list(result.keys())))
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(result)
