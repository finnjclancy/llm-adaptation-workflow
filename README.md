# LLM Adaptation Workflow — Enterprise AI Pipeline

A complete, end-to-end workflow for adapting open-source foundation models to proprietary knowledge domains. Built to run on Apple Silicon (M-series Mac) using public data, but designed so any team can swap in their own domain corpus and fine-tune a model to their specific use case.

---

## What this project demonstrates

Starting from a general-purpose language model and ending with a domain-adapted model that has been instruction-tuned, retrieval-augmented, aligned to preferences, and rigorously evaluated.

```
Public/Proprietary Documents
         │
         ▼
  02 · Data Preparation        ← Convert raw documents into training datasets
         │
    ┌────┴────┐
    ▼         ▼
03 · RAG   04 · Fine-Tuning    ← Two adaptation paths (choose one or combine)
    │         │
    └────┬────┘
         ▼
  05 · Alignment               ← DPO preference optimisation
         │
         ▼
  06 · Evaluation              ← Benchmark before vs after
```

---

## Notebooks

| Notebook | Topic | Key concepts |
|----------|-------|-------------|
| `00_project_overview.ipynb` | Project overview & setup | Architecture, design decisions, how to adapt |
| `01_foundation_models.ipynb` | Foundation models & transformers | Transformers, tokenisation, inference, HuggingFace |
| `01b_neural_networks.ipynb` | Neural networks from scratch | Single neuron, feedforward network, tiny LM trained on financial text, backpropagation |
| `02_data_preparation.ipynb` | Data preparation | SEC EDGAR ingestion, instruction dataset construction, synthetic data |
| `03_rag_pipeline.ipynb` | Retrieval-Augmented Generation | Embeddings, vector stores, retrieval, RAG vs fine-tuning |
| `04_fine_tuning.ipynb` | LoRA fine-tuning | Parameter-efficient fine-tuning, LoRA, Apple Silicon (MPS) |
| `05_alignment.ipynb` | Preference alignment | DPO, preference datasets, reward model concepts |
| `06_evaluation.ipynb` | Evaluation & benchmarking | ROUGE, BERTScore, hallucination detection, before/after |

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Models | HuggingFace Transformers, `TinyLlama-1.1B`, `Phi-2` |
| Training | PyTorch (MPS backend), `peft`, `trl` |
| RAG | `sentence-transformers`, `faiss-cpu`, `langchain` |
| Data | `datasets`, `sec-edgar-downloader`, `pandas` |
| Evaluation | `evaluate`, `bert_score`, `rouge_score` |

---

## Hardware requirements

Designed and tested on **Apple Silicon M-series Mac** (8 GB RAM minimum, 16 GB recommended).  
Uses the MPS (Metal Performance Shaders) backend for PyTorch — no GPU required.

For cloud alternatives: Google Colab T4 (free tier) works with minor `device` changes noted in each notebook.

---

## How to adapt this to your domain

Every notebook has a clearly labelled **"Adapt This"** section. To use your own domain:

1. **Swap the corpus** — replace SEC filings in `02_data_preparation.ipynb` with your documents (PDFs, HTML, plain text)
2. **Update the instruction template** — edit `utils/templates.py` to reflect your task type
3. **Adjust the evaluation questions** — replace the financial Q&A benchmark in `06_evaluation.ipynb` with domain-relevant questions

The rest of the pipeline runs unchanged.

---

## Setup

```bash
git clone https://github.com/clancyfi/llm-adaptation-workflow
cd llm-adaptation-workflow
pip install -r requirements.txt
jupyter lab
```

---

## Local app (no notebooks required)

For a clickable version of the workflow — prepare data, fine-tune, ground answers
with RAG, compare the base vs fine-tuned model side by side, and download the
result — there's a local **Streamlit app** in [`app/`](app/). It runs entirely on
the machine and works on **Windows, macOS, and Linux** (auto-detects CUDA / Apple
Silicon MPS / CPU).

The app adds a few things over a straight notebook run: a **base-model picker**
(Qwen2.5 0.5B / TinyLlama 1.1B / Qwen2.5 1.5B / SmolLM2 1.7B), a **live training-time
estimate** that updates with the chosen model and epochs, and a **RAG tab** that
retrieves the source passage and answers from it — so figures are correct even
when the small base model's memory is not.

**For a non-technical user** (e.g. to hand off): just double-click the launcher —
it creates its own virtual environment and installs everything on first run.

| Platform | Launcher |
|----------|----------|
| Windows  | double-click `start_windows.bat` |
| macOS    | double-click `start_mac.command` |
| Any      | `python run_app.py` |

The only prerequisite is **Python 3.10+** installed (on Windows, tick *"Add Python
to PATH"* during install). See [`app/README_FOR_MANAGER.md`](app/README_FOR_MANAGER.md)
for plain-English instructions to pass along.

The download tab exports the fine-tuned model as a `.zip` containing just the LoRA
adapter (a few MB) plus a `HOW_TO_LOAD.txt` — the base model is fetched
automatically when loaded.

---

## Context

Built to demonstrate the machine learning engineering skills required to adapt foundation models in an enterprise setting — covering the full lifecycle from raw documents through fine-tuning, alignment, and production-readiness evaluation.

Domain used: **financial services** (SEC filings, earnings reports) — representative of work performed at Data & AI consultancies, hedge funds, and AI research teams in regulated industries.
