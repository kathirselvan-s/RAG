#!/usr/bin/env python3
"""
Web UI for the RAG Chatbot using Gradio.
Run: python web_ui.py
"""

import inspect
import sys
from pathlib import Path

import gradio as gr

# Import from the main script
sys.path.insert(0, str(Path(__file__).parent))
from rag_chatbot import (
    DATA_DIR, LLM_MODEL, EMBED_MODEL,
    check_ollama, ensure_models, load_documents, split_documents,
    build_or_load_vectorstore, create_rag_chain, console
)

# Global chain
chain = None
retriever = None


def initialize():
    global chain, retriever
    models = check_ollama()
    ensure_models(models)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents(DATA_DIR)
    if documents:
        chunks = split_documents(documents)
        vectorstore = build_or_load_vectorstore(chunks)
    else:
        vectorstore = build_or_load_vectorstore()

    chain, retriever = create_rag_chain(vectorstore)
    return "Knowledge base ready!"


def chat(message, history):
    history = history or []
    messages = history + [{"role": "user", "content": message}]

    if chain is None:
        return messages + [{
            "role": "assistant",
            "content": "System not initialized. Please wait...",
        }]

    try:
        answer = chain.invoke(message)
    except Exception as e:
        answer = f"Error: {e}"

    return messages + [{"role": "assistant", "content": answer}]


def reindex():
    global chain, retriever
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    vectorstore = build_or_load_vectorstore(chunks, force_rebuild=True)
    chain, retriever = create_rag_chain(vectorstore)
    return "✓ Knowledge base re-indexed successfully!"


blocks_options = {"title": "TechNova RAG Chatbot"}
launch_options = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
}
if "theme" in inspect.signature(gr.Blocks).parameters:
    blocks_options["theme"] = gr.themes.Soft()
elif "theme" in inspect.signature(gr.Blocks.launch).parameters:
    launch_options["theme"] = gr.themes.Soft()

with gr.Blocks(**blocks_options) as demo:
    gr.Markdown(f"""
    # 🧠 TechNova Knowledge Chatbot
    Powered by **{LLM_MODEL}** + RAG  
    Drop files into the `data/` folder and click **Re-index** to teach the bot new knowledge.
    """)

    with gr.Row():
        status = gr.Textbox(label="Status", value="Initializing...", interactive=False)
        reindex_btn = gr.Button("🔄 Re-index Knowledge Base", variant="secondary")

    chatbot_options = {"label": "Chat", "height": 500}
    if "type" in inspect.signature(gr.Chatbot).parameters:
        chatbot_options["type"] = "messages"
    chatbot = gr.Chatbot(**chatbot_options)
    msg = gr.Textbox(label="Your question", placeholder="Ask anything about the documents...")
    clear = gr.Button("Clear")

    # Events
    demo.load(initialize, outputs=status)
    reindex_btn.click(reindex, outputs=status)
    msg.submit(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(**launch_options)
