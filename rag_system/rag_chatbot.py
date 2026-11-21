#!/usr/bin/env python3
"""
RAG Chatbot powered by Llama 3.2 3B (via Ollama)
- Automatically indexes all files from the data/ folder
- Supports TXT, MD, PDF, DOCX
- Uses GPU automatically if available (Ollama handles it)
- Vector store: ChromaDB (persistent)
- Optimized for speed with local embeddings + local LLM
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Optional

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

console = Console()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"          # ../data
PERSIST_DIR = Path(__file__).parent / "chroma_db"        # local vector store
COLLECTION_NAME = "knowledge_base"

# Models - change if you want different ones
LLM_MODEL = "llama3.2:3b"               # The brain
EMBED_MODEL = "nomic-embed-text"        # Fast local embeddings (pull with ollama)
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# Alternative embed: "all-minilm" or use sentence-transformers

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4                               # number of chunks to retrieve

# Supported file extensions
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".py", ".html"}

SYSTEM_PROMPT = """You are a friendly, helpful assistant for TechNova Solutions.
For greetings, thanks, small talk, and other general conversation, respond naturally and briefly. These do not need to be answered from the context.
For questions about TechNova, its products, employees, HR, or policies, answer using only the provided context. Do not invent or assume company-specific facts.
If the context does not contain the answer to a company-specific question, say: "I don't have that information in the knowledge base."
Keep answers clear, concise, and professional. Cite the source file when possible.

Context:
{context}"""


def check_ollama():
    """Verify Ollama server and installed models."""
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.list()
        console.print("[green]✓ Connected to Ollama server[/green]")
        return [model.model for model in response.models]
    except Exception as e:
        console.print(f"[red]Ollama connection error: {e!r}[/red]")
        console.print(f"[yellow]Check whether Ollama is running at {OLLAMA_BASE_URL}[/yellow]")
        sys.exit(1)


def ensure_models(models: List[str]):
    """Pull required models if missing."""
    import ollama
    client = ollama.Client(host=OLLAMA_BASE_URL)
    required = [LLM_MODEL, EMBED_MODEL]
    available = {model.split(":")[0] for model in models}
    for model in required:
        base = model.split(":")[0]
        if base not in available:
            console.print(f"[yellow]Pulling model {model} ... this may take a while[/yellow]")
            client.pull(model)
            console.print(f"[green]✓ {model} ready[/green]")


def load_documents(data_dir: Path) -> List:
    """Load all supported documents from the data directory."""
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        Docx2txtLoader,
    )

    documents = []
    files = list(data_dir.rglob("*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        console.print(f"[yellow]No supported files found in {data_dir}[/yellow]")
        return []

    console.print(f"[cyan]Found {len(files)} files to index...[/cyan]")

    for file_path in files:
        try:
            ext = file_path.suffix.lower()
            if ext in {".txt", ".md", ".csv", ".json", ".py", ".html"}:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(file_path))
            else:
                continue

            docs = loader.load()
            # Add source metadata
            for d in docs:
                d.metadata["source"] = str(file_path.relative_to(data_dir))
                d.metadata["filename"] = file_path.name
            documents.extend(docs)
            console.print(f"  ✓ Loaded: {file_path.name}")
        except Exception as e:
            console.print(f"  [red]✗ Failed {file_path.name}: {e}[/red]")

    return documents


def split_documents(documents: List) -> List:
    """Split documents into chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    console.print(f"[green]Created {len(chunks)} chunks[/green]")
    return chunks


def get_embeddings():
    """Return Ollama embeddings (GPU accelerated if available)."""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def build_or_load_vectorstore(chunks: Optional[List] = None, force_rebuild: bool = False):
    """Build or load persistent Chroma vector store."""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings()

    if force_rebuild and PERSIST_DIR.exists():
        import shutil
        shutil.rmtree(PERSIST_DIR)
        console.print("[yellow]Old vector store removed (force rebuild)[/yellow]")

    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()) and not force_rebuild:
        console.print("[cyan]Loading existing vector store...[/cyan]")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        count = vectorstore._collection.count()
        console.print(f"[green]Loaded {count} existing vectors[/green]")
        return vectorstore

    if not chunks:
        console.print("[red]No documents to index and no existing store found.[/red]")
        sys.exit(1)

    console.print("[cyan]Building vector store (this uses GPU if available)...[/cyan]")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    console.print(f"[green]✓ Indexed {len(chunks)} chunks into ChromaDB[/green]")
    return vectorstore


def create_rag_chain(vectorstore):
    """Create the RAG retrieval + generation chain."""
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,          # slightly creative but grounded
        num_ctx=4096,             # context window
        # Ollama automatically uses GPU if available
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def interactive_chat(chain, retriever):
    """Simple beautiful CLI chat loop."""
    console.print(Panel.fit(
        "[bold green]RAG Chatbot Ready[/bold green]\n"
        f"Model: [cyan]{LLM_MODEL}[/cyan]  |  Embeddings: [cyan]{EMBED_MODEL}[/cyan]\n"
        "Commands: /reindex  /sources <query>  /quit",
        title="TechNova Knowledge Bot",
        border_style="green",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not question.strip():
            continue

        if question.strip().lower() in {"/quit", "/exit", "quit", "exit"}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if question.strip().lower() == "/reindex":
            console.print("[cyan]Re-indexing all documents...[/cyan]")
            docs = load_documents(DATA_DIR)
            chunks = split_documents(docs)
            vectorstore = build_or_load_vectorstore(chunks, force_rebuild=True)
            chain, retriever = create_rag_chain(vectorstore)
            console.print("[green]✓ Knowledge base updated![/green]")
            continue

        if question.strip().lower().startswith("/sources"):
            q = question[8:].strip() or " "
            docs = retriever.invoke(q)
            console.print("\n[bold]Retrieved sources:[/bold]")
            for i, d in enumerate(docs, 1):
                console.print(f"  {i}. {d.metadata.get('source')} (score relevant)")
                console.print(f"     {d.page_content[:150]}...")
            continue

        # Normal question
        start = time.time()
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            try:
                answer = chain.invoke(question)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
        elapsed = time.time() - start

        console.print(Panel(
            Markdown(answer),
            title=f"Assistant ({elapsed:.1f}s)",
            border_style="cyan",
        ))


def main():
    console.print(Panel.fit(
        "[bold]Building RAG System with Llama 3.2 3B[/bold]\n"
        "GPU will be used automatically by Ollama if available.",
        border_style="blue",
    ))

    # 1. Check Ollama
    models = check_ollama()
    ensure_models(models)

    # 2. Load & index documents
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents(DATA_DIR)

    if documents:
        chunks = split_documents(documents)
        vectorstore = build_or_load_vectorstore(chunks, force_rebuild=False)
    else:
        # Try loading existing store
        vectorstore = build_or_load_vectorstore(force_rebuild=False)

    # 3. Create chain
    chain, retriever = create_rag_chain(vectorstore)

    # 4. Start chat
    interactive_chat(chain, retriever)


if __name__ == "__main__":
    main()
