"""Streamlit front end for the PICO-8 RAG database. Run locally with `streamlit run streamlit_app.py`."""
import os
import re
import sys

# Chroma needs sqlite3 >= 3.35; Streamlit Community Cloud's system sqlite can be older, so use the
# bundled pysqlite3-binary when it is installed (Linux only, see requirements.txt).
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv(rag.ROOT / ".env")
try:  # Streamlit Cloud: keys come from the app's Secrets settings
    for key in ("GROQ_API_KEY", "LLM_MODEL"):
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = str(st.secrets[key])
except FileNotFoundError:  # no secrets.toml locally; .env is used instead
    pass


@st.cache_resource(show_spinner="First start: building the vector database (about 1-2 minutes)...")
def database_ready() -> bool:
    rag.ensure_built()
    return True


st.set_page_config(page_title="PICO-8 RAG", page_icon="🎮")
st.title("PICO-8 code generator")
st.caption("Retrieval over 100 recent Lexaloffle BBS carts, generation with gpt-oss-120b on Groq.")
database_ready()

prompt = st.text_area("What should the cart do?", placeholder="make a snake game")
k = st.slider("Snippets to retrieve", 2, 12, 6)
col1, col2 = st.columns(2)
generate = col1.button("Generate", type="primary", width="stretch")
search_only = col2.button("Search only", width="stretch")

if (generate or search_only) and not prompt.strip():
    st.error("Enter a request, e.g. 'make a snake game'.")
elif generate:
    with st.spinner("Retrieving snippets and generating code..."):
        try:
            answer, code, sources = rag.ask(prompt.strip(), k)
            st.session_state.result = {"kind": "generate", "answer": answer, "code": code, "sources": sources}
        except Exception as e:  # UI boundary: show the message instead of a stack trace
            st.session_state.result = {"kind": "error", "message": f"{type(e).__name__}: {e}"}
elif search_only:
    try:
        st.session_state.result = {"kind": "search", "hits": rag.search(prompt.strip(), k)}
    except Exception as e:
        st.session_state.result = {"kind": "error", "message": f"{type(e).__name__}: {e}"}

# Rendered from session state so the result survives reruns (e.g. clicking the download button).
result = st.session_state.get("result")
if result and result["kind"] == "error":
    st.error(result["message"])
elif result and result["kind"] == "generate":
    st.code(result["code"], language="lua")
    notes = re.sub(r"```.*?```", "", result["answer"], flags=re.S).strip()
    if notes:
        st.markdown(notes)
    st.markdown("**Sources used**\n" + "\n".join(
        f"- [{s['game_name']}]({s['thread_url']}) by {s['author']} ({s['license']})" for s in result["sources"]
    ))
    st.download_button("Download .p8 cartridge", rag.p8_text(result["code"]), file_name="generated.p8", mime="text/plain")
elif result and result["kind"] == "search":
    for n, h in enumerate(result["hits"], 1):
        m = h["meta"]
        st.markdown(f"**{n}. [{m['game_name']}]({m['thread_url']})** by {m['author']} ({m['kind']}, distance {h['distance']:.3f})")
        st.code(h["text"][:800], language="lua")
