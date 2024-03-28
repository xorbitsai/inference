# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# -*- coding:utf-8 -*-
import gradio as gr

title = """<h1 align="left" style="min-width:200px; margin-top:0;">Chat with DeepSeek-VL </h1>"""
description_top = """"""
description = """"""
CONCURRENT_COUNT = 10


ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

small_and_beautiful_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#EBFAF2",
        c100="#CFF3E1",
        c200="#A8EAC8",
        c300="#77DEA9",
        c400="#3FD086",
        c500="#02C160",
        c600="#06AE56",
        c700="#05974E",
        c800="#057F45",
        c900="#04673D",
        c950="#2E5541",
        name="small_and_beautiful",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f6f7f8",
        # c100="#f3f4f6",
        c100="#F2F2F2",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        # c900="#272727",
        c900="#2B2B2B",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    # button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_600",
    # button_primary_background_fill_hover="*primary_400",
    # button_primary_border_color="*primary_500",
    button_primary_border_color_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_50",
    button_secondary_background_fill_dark="*neutral_900",
    button_secondary_text_color="*neutral_800",
    button_secondary_text_color_dark="white",
    # background_fill_primary="#F7F7F7",
    # background_fill_primary_dark="#1F1F1F",
    # block_title_text_color="*primary_500",
    block_title_background_fill_dark="*primary_900",
    block_label_background_fill_dark="*primary_900",
    input_background_fill="#F6F6F6",
    # chatbot_code_background_color_dark="*neutral_950",
)
