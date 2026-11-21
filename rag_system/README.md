# RAG Chatbot with Llama 3.2 3B (Local + GPU)

A complete **Retrieval-Augmented Generation** chatbot that automatically learns from every file you put in the `data/` folder.

## Features

- 🧠 Brain: **llama3.2:3b** via Ollama (local, private, fast)
- ⚡ GPU acceleration automatic (Ollama detects NVIDIA/AMD/Apple Silicon)
- 📂 Auto-ingest: Drop any `.txt`, `.md`, `.pdf`, `.docx` into `data/` folder
- 🔄 One-click re-index
- 💾 Persistent ChromaDB vector store
- Beautiful CLI + optional Gradio Web UI

## Prerequisites

1. **Ollama** installed → https://ollama.com
2. Pull the models (once):

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

3. Python 3.10+

## Quick Start

On Windows, use the Windows section below instead of the generic install/run commands
to keep Python packages and temporary files on O:.

```bash
# 1. Install dependencies
cd rag_system
pip install -r requirements.txt

# 2. Put your documents in the data folder
#    (sample files are already there)
ls ../data/

# 3. Run CLI chatbot
python rag_chatbot.py

# OR run beautiful Web UI
python web_ui.py
# Then open http://localhost:7860
```

## How it works

1. On startup it scans `../data/` for all supported files
2. Chunks them intelligently
3. Creates embeddings with `nomic-embed-text` (runs on GPU if available)
4. Stores vectors in local ChromaDB (`chroma_db/`)
5. When you ask a question → retrieves top relevant chunks → feeds to Llama 3.2 3B → answers

## Answering behavior

- Greetings and small talk get a natural, brief reply without requiring document context.
- Questions about TechNova are answered only from retrieved company documents.
- If those documents do not contain the answer, the chatbot says it does not have that information.

## Adding new knowledge

Just drop new files into the `data/` folder and type `/reindex` in the chat  
(or click the Re-index button in the web UI).

## Commands (CLI)

- `/reindex` – rebuild knowledge base from all files in data/
- `/sources <query>` – show which documents were retrieved
- `/quit` – exit

## Performance Tips

- Ollama automatically uses GPU when available → much faster generation
- For even better quality you can change `LLM_MODEL = "llama3.1:8b"` (needs more VRAM)
- Increase `TOP_K` if answers miss context
- Decrease `CHUNK_SIZE` for more precise retrieval

## Folder Structure

```
artifacts/
├── data/                  ← put all your documents here
│   ├── company_info.txt
│   ├── products.md
│   └── hr_policy.txt
└── rag_system/
    ├── rag_chatbot.py     ← main CLI
    ├── web_ui.py          ← Gradio web interface
    ├── requirements.txt
    ├── chroma_db/         ← auto-created vector store
    └── README.md
```

Enjoy your private, GPU-accelerated knowledge chatbot!

## Windows: Keep Python dependencies on O:

The Python dependencies and pip temporary files should stay on the O: drive.
From PowerShell:

```powershell
$projectRoot = 'O:\Project\RAG'
$env:TEMP = Join-Path $projectRoot 'tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& "$projectRoot\.venv\Scripts\python.exe" -m pip install --no-cache-dir -r "$projectRoot\rag_system\requirements.txt"
```

Ollama is a separate local application and manages its own model storage. This
project connects to it at `http://127.0.0.1:11434`; it does not install Ollama
or its models into the Python virtual environment. If Ollama is already running,
run the chatbot with:

```powershell
& 'O:\Project\RAG\.venv\Scripts\python.exe' 'O:\Project\RAG\rag_system\rag_chatbot.py'
```

The existing Ollama model store is separate from this project and remains where
Ollama currently manages it.
