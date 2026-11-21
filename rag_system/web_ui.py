#!/usr/bin/env python3
"""
Web UI for the RAG Chatbot using Gradio.
Run: python web_ui.py
"""

import gradio as gr
from pathlib import Path
import sys

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
    if chain is None:
        return history + [[message, "System not initialized. Please wait..."]]
    try:
        answer = chain.invoke(message)
        history.append([message, answer])
    except Exception as e:
        history.append([message, f"Error: {str(e)}"])
    return history


def reindex():
    global chain, retriever
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    vectorstore = build_or_load_vectorstore(chunks, force_rebuild=True)
    chain, retriever = create_rag_chain(vectorstore)
    return "✓ Knowledge base re-indexed successfully!"


with gr.Blocks(title="TechNova RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🧠 TechNova Knowledge Chatbot
    Powered by **{LLM_MODEL}** + RAG  
    Drop files into the `data/` folder and click **Re-index** to teach the bot new knowledge.
    """)

    with gr.Row():
        status = gr.Textbox(label="Status", value="Initializing...", interactive=False)
        reindex_btn = gr.Button("🔄 Re-index Knowledge Base", variant="secondary")

    chatbot = gr.Chatbot(label="Chat", height=500)
    msg = gr.Textbox(label="Your question", placeholder="Ask anything about the documents...")
    clear = gr.Button("Clear")

    # Events
    demo.load(initialize, outputs=status)
    reindex_btn.click(reindex, outputs=status)
    msg.submit(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
