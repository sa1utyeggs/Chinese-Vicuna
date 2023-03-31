import time

import gradio as gr


def process(text):
    output = 'asdfasdf'
    time.sleep(1000)
    output = text


gr.Interface(
    fn=process,
    inputs=[gr.components.Textbox(
        lines=2, label="text", placeholder="Tell me about alpacas."
    )],
    outputs=[
        gr.components.Textbox(
            lines=15,
            label="output",
        )
    ],
    title="test",
    description="test",
).queue().launch(share=False)
