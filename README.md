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

## Context

Built to demonstrate the machine learning engineering skills required to adapt foundation models in an enterprise setting — covering the full lifecycle from raw documents through fine-tuning, alignment, and production-readiness evaluation.

Domain used: **financial services** (SEC filings, earnings reports) — representative of work performed at Data & AI consultancies, hedge funds, and AI research teams in regulated industries.
