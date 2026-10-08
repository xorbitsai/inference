from __future__ import annotations

from dataclasses import dataclass
from html import escape

import gradio as gr


@dataclass(frozen=True)
class EmojiPaletteItem:
    emoji: str
    label: str
    description: str


EMOJI_PALETTE_CSS = """
.emoji-palette {
    max-width: 100%;
}

.emoji-palette-grid {
    display: flex;
    gap: 3px;
    flex-wrap: wrap;
    align-items: flex-start;
    max-height: 124px;
    overflow-y: auto;
}

.emoji-palette-button {
    flex: 0 0 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
}

.emoji-palette-button button {
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    border-radius: 4px;
    font-size: 17px;
    line-height: 1;
    padding: 0 !important;
}

button.emoji-palette-button {
    width: 28px;
    min-width: 28px;
    height: 28px;
    min-height: 28px;
    border-radius: 4px;
    font-size: 17px;
    line-height: 1;
    padding: 0;
    cursor: pointer;
}
"""


EMOJI_PALETTE_ITEMS: tuple[EmojiPaletteItem, ...] = (
    EmojiPaletteItem("👂", "囁き", "耳元の音"),
    EmojiPaletteItem("😮‍💨", "吐息", "溜息、寝息"),
    EmojiPaletteItem("⏸️", "間", "沈黙"),
    EmojiPaletteItem("🤭", "笑い", "くすくす、含み笑い"),
    EmojiPaletteItem("🥵", "喘ぎ", "うめき声、唸り声"),
    EmojiPaletteItem("📢", "エコー", "リバーブ"),
    EmojiPaletteItem("😏", "からかう", "甘えるように"),
    EmojiPaletteItem("🥺", "震え声", "自信なさげに"),
    EmojiPaletteItem("🌬️", "息切れ", "荒い息遣い、呼吸音"),
    EmojiPaletteItem("😮", "息をのむ", "Gasp"),
    EmojiPaletteItem("👅", "舐める音", "咀嚼音、水音"),
    EmojiPaletteItem("💋", "リップノイズ", "Lip smack"),
    EmojiPaletteItem("🫶", "優しく", "Tenderly"),
    EmojiPaletteItem("😭", "泣き声", "嗚咽、悲しみ"),
    EmojiPaletteItem("😱", "悲鳴", "叫び、絶叫"),
    EmojiPaletteItem("😪", "眠そう", "気だるげに"),
    EmojiPaletteItem("😴", "寝言", "いびき"),
    EmojiPaletteItem("⏩", "早口", "一気に、急いで"),
    EmojiPaletteItem("📞", "電話越し", "スピーカー越し"),
    EmojiPaletteItem("🐢", "ゆっくり", "Slowly"),
    EmojiPaletteItem("🥤", "飲み込む", "唾を飲む音"),
    EmojiPaletteItem("🤧", "咳・鼻", "咳き込み、鼻すすり"),
    EmojiPaletteItem("😒", "舌打ち", "Tutting"),
    EmojiPaletteItem("😰", "慌てる", "動揺、緊張、どもり"),
    EmojiPaletteItem("😆", "喜び", "嬉しそうに"),
    EmojiPaletteItem("💥", "勢いよく", "力強い勢い"),
    EmojiPaletteItem("😠", "怒り", "不満げ、拗ねる"),
    EmojiPaletteItem("😲", "驚き", "感嘆"),
    EmojiPaletteItem("🥱", "あくび", "Yawn"),
    EmojiPaletteItem("😖", "苦しげ", "Agonizingly"),
    EmojiPaletteItem("😟", "心配", "不安そうに"),
    EmojiPaletteItem("🫣", "照れ", "恥ずかしそうに"),
    EmojiPaletteItem("🙄", "呆れ", "Exasperatedly"),
    EmojiPaletteItem("😊", "楽しげ", "嬉しそうに"),
    EmojiPaletteItem("😎", "得意げ", "自信ありげに"),
    EmojiPaletteItem("👌", "相槌", "頷く音"),
    EmojiPaletteItem("🙏", "懇願", "お願いするように"),
    EmojiPaletteItem("🥴", "酔う", "Drunkenly"),
    EmojiPaletteItem("🎵", "鼻歌", "Humming"),
    EmojiPaletteItem("🤐", "口を塞ぐ", "Muffled"),
    EmojiPaletteItem("😌", "安堵", "満足げに"),
    EmojiPaletteItem("🤔", "疑問", "Questioning"),
    EmojiPaletteItem("💪", "力強く", "力を込めて"),
    EmojiPaletteItem("👃", "嗅ぐ音", "匂いを嗅ぐ音"),
    EmojiPaletteItem("📖", "朗読", "ナレーション"),
)


_INSERT_EMOJI_ON_POINTER_DOWN = (
    "event.preventDefault();"
    "const root=this.closest('[data-irodori-emoji-palette]');"
    "const input=root?document.querySelector(root.dataset.target):null;"
    "if(!input)return;"
    "const emoji=this.dataset.emoji;"
    "const text=input.value||'';"
    "const focused=document.activeElement===input;"
    "const start=focused&&typeof input.selectionStart==='number'?input.selectionStart:text.length;"
    "const end=focused&&typeof input.selectionEnd==='number'?input.selectionEnd:text.length;"
    "const next=text.slice(0,start)+emoji+text.slice(end);"
    "const caret=start+emoji.length;"
    "input.value=next;"
    "input.focus({preventScroll:true});"
    "input.setSelectionRange(caret,caret);"
    "input.dispatchEvent(new Event('input',{bubbles:true}));"
    "input.dispatchEvent(new Event('change',{bubbles:true}));"
)


def _textbox_selector(textbox: gr.Textbox) -> str:
    elem_id = getattr(textbox, "elem_id", None)
    root_selector = f"#{elem_id}" if elem_id else f"#component-{textbox._id}"
    return f"{root_selector} textarea, {root_selector} input:not([type='hidden'])"


def _emoji_palette_html(textbox: gr.Textbox) -> str:
    target = escape(_textbox_selector(textbox), quote=True)
    handler = escape(_INSERT_EMOJI_ON_POINTER_DOWN, quote=True)
    buttons = []
    for item in EMOJI_PALETTE_ITEMS:
        emoji = escape(item.emoji, quote=True)
        title = escape(f"{item.label}: {item.description}", quote=True)
        buttons.append(
            '<button type="button" '
            'class="emoji-palette-button" '
            f'data-emoji="{emoji}" '
            f'title="{title}" '
            f'aria-label="{title}" '
            f'onpointerdown="{handler}">'
            f"{emoji}</button>"
        )
    return (
        '<div class="emoji-palette-grid" '
        'data-irodori-emoji-palette="true" '
        f'data-target="{target}">'
        f"{''.join(buttons)}</div>"
    )


def build_emoji_palette(textbox: gr.Textbox, *, open: bool = True) -> None:
    with gr.Accordion("Emoji Palette", open=open, elem_classes=["emoji-palette"]):
        gr.HTML(_emoji_palette_html(textbox))
