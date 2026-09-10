"""Gradio version of the web UI. Run `python app.py` and open http://127.0.0.1:7860."""
import re
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import rag

load_dotenv(rag.ROOT / ".env")


def _sources_md(sources: list[dict]) -> str:
    return "**Sources used**\n" + "\n".join(
        f"- [{s['game_name']}]({s['thread_url']}) by {s['author']} ({s['license']})" for s in sources
    )


def generate(prompt: str, k: float):
    if not prompt.strip():
        raise gr.Error("Enter a request, e.g. 'make a snake game'.")
    try:
        answer, code, sources = rag.ask(prompt.strip(), int(k))
    except Exception as e:  # show it as a popup in the UI instead of crashing
        raise gr.Error(f"{type(e).__name__}: {e}")
    notes = re.sub(r"```.*?```", "", answer, flags=re.S).strip()  # the model's notes without the code block
    p8 = rag.write_p8(code, Path(tempfile.mkdtemp()) / "generated.p8")
    return code, notes, _sources_md(sources), str(p8)


def find(prompt: str, k: float) -> str:
    if not prompt.strip():
        raise gr.Error("Enter a search query, e.g. 'platformer with double jump'.")
    try:
        hits = rag.search(prompt.strip(), int(k))
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}")
    return "\n\n".join(
        f"**{n}. [{h['meta']['game_name']}]({h['meta']['thread_url']})** by {h['meta']['author']} "
        f"({h['meta']['kind']}, distance {h['distance']:.3f})\n```\n{h['text'][:800]}\n```"
        for n, h in enumerate(hits, 1)
    )


with gr.Blocks(title="PICO-8 RAG") as demo:
    gr.Markdown("# PICO-8 code generator\nRetrieval over 100 recent Lexaloffle BBS carts, generation with gpt-oss-120b on Groq.")
    prompt = gr.Textbox(label="What should the cart do?", placeholder="make a snake game", lines=2)
    k = gr.Slider(2, 12, value=6, step=1, label="Snippets to retrieve")
    with gr.Row():
        gen_btn = gr.Button("Generate", variant="primary")
        find_btn = gr.Button("Search only")
    code = gr.Code(label="Generated PICO-8 code", lines=20)
    notes = gr.Markdown()
    sources = gr.Markdown()
    p8_file = gr.File(label="Download .p8 cartridge")
    results = gr.Markdown()
    gen_btn.click(generate, [prompt, k], [code, notes, sources, p8_file])
    find_btn.click(find, [prompt, k], results)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
