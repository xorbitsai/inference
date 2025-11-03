import gradio as gr


def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def html_left(text, label='p'):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def next_page(page_number,sentences):
    new_page_number = int(page_number) + 1
    update_page_number = gr.update(value=str(new_page_number))
    update_prev_page = gr.update(visible=True, interactive=True)
    if len(sentences.values) <= new_page_number * 20:
        update_next_page = gr.update(visible=False, interactive=False)
    else:
        update_next_page = gr.update(visible=True, interactive=True)
    return update_page_number, update_next_page, update_prev_page


def prev_page(page_number):
    new_page_number = int(page_number) - 1
    update_page_number = gr.update(value=str(new_page_number))
    if new_page_number == 1:
        update_prev_page = gr.update(visible=False, interactive=False)
    else:
        update_prev_page = gr.update(visible=True, interactive=True)
    update_next_page = gr.update(visible=True, interactive=True)
    return update_page_number, update_next_page, update_prev_page


def update_current_texts(page_number,sentences):
    start_index = (int(page_number) - 1) * 20
    end_index = int(page_number) * 20
    current_texts = sentences.values[start_index:end_index if end_index < len(sentences.values) else len(sentences.values)]
    return gr.update(values=current_texts)
