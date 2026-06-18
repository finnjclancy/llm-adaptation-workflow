"""
Core pipeline logic for the LLM Adaptation frontend.

A self-contained, cross-platform (Windows / macOS / Linux) distillation of the
notebooks:

    02_data_preparation  -> build_dataset()
    03_rag_pipeline       -> rag_retrieve() / rag_answer()
    04_fine_tuning       -> run_finetuning()
    inference            -> ModelBundle.chat() / generate_base() / generate_finetuned()
    06_evaluation        -> evaluate()

Everything is anchored to the project root via absolute paths, so it does not
matter what directory the app is launched from. No network call is required for
the data (a representative financial dataset is baked in); models and the
embedding model download from Hugging Face on first use.

The base model is selectable — see MODELS. LoRA adapters are stored per base
model, so fine-tuning one model doesn't clobber another.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ── Paths (anchored to the repo root, not the current working directory) ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EXAMPLES_PATH = PROCESSED_DIR / "examples.json"   # instruction dataset (fine-tuning)
CORPUS_PATH = PROCESSED_DIR / "corpus.json"        # RAG knowledge base (passages)
META_PATH = PROCESSED_DIR / "meta.json"            # where the dataset came from
MODELS_DIR = PROJECT_ROOT / "models"
ADAPTERS_DIR = MODELS_DIR / "adapters"

SYSTEM_PROMPT = "You are a financial analyst assistant."

# ── Selectable base models ────────────────────────────────────────────────────
# All are small, openly available, chat/instruction-tuned, and share the
# Llama/Qwen attention layout (q/k/v/o_proj) so the same LoRA config fits each.
MODELS: dict[str, dict] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "label": "Qwen2.5 0.5B — fastest, lightest",
        "params_b": 0.5,
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "label": "TinyLlama 1.1B — balanced (default)",
        "params_b": 1.1,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "label": "Qwen2.5 1.5B — better quality",
        "params_b": 1.5,
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "label": "SmolLM2 1.7B — strongest, slowest",
        "params_b": 1.7,
    },
}
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ── Baked-in instruction dataset ──────────────────────────────────────────────
# A compact financial Q&A set so the demo runs fully offline. The `context`
# fields double as the RAG knowledge base.
SEED_EXAMPLES = [
    {
        "instruction": "What was Apple's total net sales for fiscal year 2023?",
        "context": "Apple Inc. reported total net sales of $383.3 billion for the fiscal year ended September 30, 2023, compared to $394.3 billion for fiscal 2022, a decrease of 2.8%.",
        "response": "Apple's total net sales for fiscal year 2023 were $383.3 billion, down 2.8% from $394.3 billion in fiscal 2022.",
    },
    {
        "instruction": "How did Microsoft Azure grow in fiscal 2023?",
        "context": "Azure and other cloud services revenue grew 27% in fiscal year 2023, with the Intelligent Cloud segment generating $87.9 billion.",
        "response": "Microsoft Azure grew 27% in fiscal 2023, helping the Intelligent Cloud segment reach $87.9 billion in revenue.",
    },
    {
        "instruction": "What drove JPMorgan Chase's net income in 2023?",
        "context": "JPMorgan Chase reported net income of $49.6 billion for fiscal year 2023, driven by higher net interest income as interest rates rose.",
        "response": "JPMorgan's 2023 net income of $49.6 billion was driven primarily by higher net interest income in a rising rate environment.",
    },
    {
        "instruction": "What percentage of Apple's revenue comes from international markets?",
        "context": "International markets accounted for 58% of total Apple revenues in fiscal 2023.",
        "response": "International markets accounted for 58% of Apple's total revenues in fiscal 2023.",
    },
    {
        "instruction": "What were Apple's main revenue segments in fiscal 2023?",
        "context": "Apple's Products revenue was $298.1 billion and Services revenue was $85.2 billion in fiscal 2023, with iPhone contributing $200.6 billion.",
        "response": "Apple's revenue split into Products ($298.1B, with iPhone at $200.6B) and Services ($85.2B) in fiscal 2023.",
    },
    {
        "instruction": "How much revenue did Microsoft report for fiscal year 2023?",
        "context": "Microsoft reported revenue of $211.9 billion for fiscal year 2023, up 7% year-over-year, with net income of $72.4 billion.",
        "response": "Microsoft reported $211.9 billion in revenue for fiscal 2023, up 7% year-over-year, with net income of $72.4 billion.",
    },
    {
        "instruction": "What are the main risks Apple identifies in its annual report?",
        "context": "Apple cites global economic conditions, intense competition, rapid technological change, supplier concentration, and cybersecurity threats as key risks.",
        "response": "Apple identifies global economic conditions, intense competition, rapid technological change, supplier concentration risk, and cybersecurity threats as its principal risks.",
    },
    {
        "instruction": "How does JPMorgan Chase manage credit risk?",
        "context": "JPMorgan Chase manages credit risk through diversification, credit limits, collateral requirements, and credit derivatives, supported by internal ratings and stress testing.",
        "response": "JPMorgan manages credit risk through portfolio diversification, per-counterparty credit limits, collateral requirements, and hedging with credit derivatives, backed by internal ratings and stress testing.",
    },
    {
        "instruction": "What was Apple's Services revenue in fiscal 2023?",
        "context": "Apple's Services segment generated $85.2 billion in revenue in fiscal 2023, a 16% increase year-over-year, driven by the App Store, Apple Music, iCloud, and Apple TV+.",
        "response": "Apple's Services revenue was $85.2 billion in fiscal 2023, up 16% year-over-year, driven by the App Store, Apple Music, iCloud, and Apple TV+.",
    },
    {
        "instruction": "What was Microsoft's operating income in fiscal 2023?",
        "context": "Microsoft reported operating income of $88.5 billion for fiscal year 2023.",
        "response": "Microsoft's operating income for fiscal 2023 was $88.5 billion.",
    },
    {
        "instruction": "How did Apple's fiscal 2023 revenue compare to the prior year?",
        "context": "Apple's total net sales fell from $394.3 billion in fiscal 2022 to $383.3 billion in fiscal 2023, a decline of 2.8%.",
        "response": "Apple's revenue declined 2.8% year-over-year, from $394.3 billion in fiscal 2022 to $383.3 billion in fiscal 2023.",
    },
    {
        "instruction": "What was JPMorgan Chase's primary income driver as rates rose?",
        "context": "Higher net interest income, driven by rising interest rates throughout 2023, was the primary contributor to JPMorgan Chase's earnings growth.",
        "response": "Rising interest rates lifted JPMorgan's net interest income, which was the primary driver of its 2023 earnings growth.",
    },
    {
        "instruction": "What was Apple's iPhone revenue in fiscal 2023?",
        "context": "Apple's iPhone revenue was $200.6 billion in fiscal 2023, representing 52% of total company revenue.",
        "response": "Apple's iPhone revenue was $200.6 billion in fiscal 2023, about 52% of total company revenue.",
    },
]

# A small held-out benchmark for the evaluation tab. Each item lists keywords a
# correct answer should contain (used for a simple fact-match score).
BENCHMARK = [
    {
        "question": "What was Apple's total net sales in fiscal year 2023?",
        "reference": "Apple's total net sales for fiscal 2023 were $383.3 billion, down 2.8% from the prior year.",
        "keywords": ["383", "2023"],
    },
    {
        "question": "How fast did Microsoft Azure grow in fiscal 2023?",
        "reference": "Microsoft Azure grew 27% in fiscal 2023.",
        "keywords": ["27", "azure"],
    },
    {
        "question": "What was JPMorgan Chase's net income in 2023?",
        "reference": "JPMorgan Chase reported net income of $49.6 billion for 2023.",
        "keywords": ["49.6", "income"],
    },
    {
        "question": "What share of Apple's revenue comes from international markets?",
        "reference": "International markets accounted for 58% of Apple's revenue.",
        "keywords": ["58", "international"],
    },
]


# ── Instruction formatting (mirrors utils/templates.py) ───────────────────────
def format_instruction(instruction: str, response: str, context: Optional[str] = None) -> str:
    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


# ── Adapter location (namespaced per base model) ──────────────────────────────
def _safe(model_id: str) -> str:
    return model_id.replace("/", "__")


def adapter_dir(model_id: str) -> Path:
    return ADAPTERS_DIR / _safe(model_id) / "final_adapter"


def adapter_exists(model_id: str) -> bool:
    d = adapter_dir(model_id)
    return d.exists() and (d / "adapter_config.json").exists()


# ── Device detection (cross-platform) ─────────────────────────────────────────
def get_device() -> str:
    """Pick the best available device. Works on Windows (CUDA/CPU), Mac (MPS), Linux.

    Degrades to "cpu" if torch isn't importable yet, so the UI still renders
    (the launcher installs torch into the venv before the app is used for real).
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label() -> str:
    try:
        import torch

        dev = get_device()
        if dev == "cuda":
            return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        if dev == "mps":
            return "Apple Silicon GPU (MPS)"
        return "CPU (no GPU detected — training will be slow but works)"
    except Exception:
        return "unknown"


# ── Training-time estimate ────────────────────────────────────────────────────
# Calibrated from a measured run: TinyLlama (1.1B) on Apple Silicon MPS took
# ~14s per example per epoch. Scaled linearly by parameter count and device.
_SECONDS_PER_EXAMPLE_EPOCH = {"cuda": 1.5, "mps": 14.0, "cpu": 50.0}


def estimate_training_seconds(model_id: str, epochs: int, num_examples: int, device: Optional[str] = None) -> float:
    device = device or get_device()
    params_b = MODELS.get(model_id, {}).get("params_b", 1.1)
    per = _SECONDS_PER_EXAMPLE_EPOCH.get(device, 50.0)
    return epochs * num_examples * per * (params_b / 1.1)


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{round(seconds)}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{round(minutes)} min"
    return f"{minutes / 60:.1f} hr"


def estimate_training_label(model_id: str, epochs: int, num_examples: int, device: Optional[str] = None) -> str:
    """Human-readable rough range, e.g. '≈ 3–7 min'."""
    base = estimate_training_seconds(model_id, epochs, num_examples, device)
    return f"≈ {_fmt_duration(base * 0.65)}–{_fmt_duration(base * 1.5)}"


# ── Status / metadata helpers ─────────────────────────────────────────────────
def dataset_exists() -> bool:
    return EXAMPLES_PATH.exists()


def corpus_exists() -> bool:
    return CORPUS_PATH.exists()


def _write_meta(**fields) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta = dataset_meta()
    meta.update(fields)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dataset_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def dataset_source() -> str:
    """'sample', 'qa_upload', or 'none' — where the current Q&A dataset came from."""
    if not dataset_exists():
        return "none"
    return dataset_meta().get("qa_source", "sample")


def corpus_count() -> int:
    if not corpus_exists():
        return 0
    try:
        return len(json.loads(CORPUS_PATH.read_text(encoding="utf-8")))
    except Exception:
        return 0


# ── 1. Data preparation ───────────────────────────────────────────────────────
def build_dataset(log: Callable[[str], None] = print) -> dict:
    """Build the built-in sample instruction dataset and save it to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    examples = [dict(ex) for ex in SEED_EXAMPLES]
    for ex in examples:
        ex["text"] = format_instruction(ex["instruction"], ex["response"], ex.get("context"))

    log(f"Built {len(examples)} instruction examples (built-in sample).")
    with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    _write_meta(qa_source="sample", qa_count=len(examples))
    log(f"Saved dataset to {EXAMPLES_PATH}")
    return {"count": len(examples), "path": str(EXAMPLES_PATH)}


def load_examples() -> list[dict]:
    if dataset_exists():
        with open(EXAMPLES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return [dict(ex) for ex in SEED_EXAMPLES]


def num_examples() -> int:
    return len(load_examples())


def sample_questions(n: int = 6) -> list[str]:
    """A few example questions drawn from the current dataset (for the UI dropdowns)."""
    out: list[str] = []
    for ex in load_examples():
        q = ex.get("instruction")
        if q and q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


def reset_to_sample(log: Callable[[str], None] = print) -> dict:
    """Discard any uploaded data and restore the built-in sample dataset."""
    for p in (EXAMPLES_PATH, CORPUS_PATH, META_PATH):
        if p.exists():
            p.unlink()
    log("Cleared uploaded data.")
    return build_dataset(log=log)


# ── 1a. Upload your own Q&A dataset (CSV / JSON) ──────────────────────────────
# Accepts flexible column/key names so users don't have to match an exact schema.
_INSTRUCTION_KEYS = ("instruction", "question", "prompt", "input", "query")
_RESPONSE_KEYS = ("response", "answer", "output", "completion", "label")
_CONTEXT_KEYS = ("context", "source", "passage", "document", "reference")


def _first_present(row: dict, keys) -> Optional[str]:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return None


def parse_qa_records(filename: str, data: bytes) -> list[dict]:
    """Parse an uploaded CSV or JSON file into instruction/response(/context) records."""
    name = filename.lower()
    if name.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict):
            rows = obj.get("data") or obj.get("examples") or obj.get("rows") or []
        else:
            rows = obj
    elif name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    else:
        raise ValueError("Please upload a .csv or .json file.")

    records = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        low = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}
        instr = _first_present(low, _INSTRUCTION_KEYS)
        resp = _first_present(low, _RESPONSE_KEYS)
        ctx = _first_present(low, _CONTEXT_KEYS)
        if instr and resp:
            rec = {"instruction": instr, "response": resp}
            if ctx:
                rec["context"] = ctx
            records.append(rec)
    return records


def save_qa_dataset(records: list[dict], log: Callable[[str], None] = print) -> dict:
    """Validate and save uploaded Q&A records as the instruction dataset."""
    if not records:
        raise ValueError(
            "No usable rows found. Each row needs a question/instruction column and an "
            "answer/response column (an optional context column is also supported)."
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for ex in records:
        ex["text"] = format_instruction(ex["instruction"], ex["response"], ex.get("context"))
    with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    _write_meta(qa_source="qa_upload", qa_count=len(records))
    log(f"Saved {len(records)} uploaded Q&A examples.")
    return {"count": len(records)}


def qa_template_csv() -> str:
    """A small CSV the user can fill in and re-upload."""
    return (
        "instruction,response,context\n"
        '"What is our refund policy?","Customers can request a refund within 30 days of purchase.",'
        '"Refunds: orders may be returned within 30 days for a full refund."\n'
        '"Who is the primary contact for billing?","Billing questions go to the finance team at billing@acme.com.",""\n'
    )


# ── 1b. Upload your own documents (TXT / MD / PDF) for RAG ─────────────────────
def clean_text(raw_text: str) -> str:
    """Strip HTML, normalise whitespace, drop tiny lines (from notebook 02)."""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [line for line in lines if len(line) > 30]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max(1, chunk_size - overlap)
    return chunks


def extract_document_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md", ".markdown", ".text")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import pypdf
        except ImportError:
            raise ValueError(
                f"'{filename}' is a PDF, but PDF support isn't installed. Upload .txt/.md "
                "files instead, or reinstall the app to enable PDF reading."
            )
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {filename}. Use .txt, .md, or .pdf.")


def passages_from_documents(
    files: list[tuple[str, bytes]],
    chunk_size: int = 200,
    overlap: int = 40,
    log: Callable[[str], None] = lambda m: None,
) -> list[str]:
    """Extract, clean and chunk uploaded documents into passages (not persisted)."""
    passages: list[str] = []
    for fname, data in files:
        cleaned = clean_text(extract_document_text(fname, data))
        chunks = [c for c in chunk_text(cleaned, chunk_size, overlap) if len(c.split()) >= 10]
        passages.extend(chunks)
        log(f"{fname}: {len(chunks)} passages")
    return passages


def build_corpus_from_documents(
    files: list[tuple[str, bytes]],
    chunk_size: int = 200,
    overlap: int = 40,
    log: Callable[[str], None] = print,
) -> dict:
    """Extract, clean and chunk uploaded documents into the RAG knowledge base."""
    passages = passages_from_documents(files, chunk_size, overlap, log)
    if not passages:
        raise ValueError("No usable text could be extracted from the uploaded documents.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)
    _write_meta(corpus_source="document_upload", corpus_count=len(passages))
    log(f"Built knowledge base of {len(passages)} passages.")
    return {"count": len(passages)}


# ── 1c. Auto-draft Q&A pairs from passages (LLM-as-teacher, local model) ───────
# Uses few-shot completion: small local models follow a 'Passage → Q/A' pattern
# more reliably as a raw completion than as a chat turn. Output is rough — the UI
# has the user review and edit before saving.
_QA_FEWSHOT = [
    (
        "Acme Corp offers a 30-day money-back guarantee on all hardware purchases, "
        "no questions asked.",
        "What is Acme's refund policy on hardware?",
        "Acme offers a 30-day money-back guarantee on all hardware purchases.",
    ),
    (
        "The customer support team is available Monday to Friday, 9am to 5pm GMT, "
        "by phone and email.",
        "When is Acme's customer support available?",
        "Acme's support team is available Monday to Friday, 9am-5pm GMT, by phone and email.",
    ),
]


def _qa_completion_prompt(passage: str) -> str:
    demos = "\n\n".join(f"Passage: {p}\nQ: {q}\nA: {a}" for p, q, a in _QA_FEWSHOT)
    return (
        "Write one specific question and a concise, accurate answer based ONLY on "
        f"each passage.\n\n{demos}\n\nPassage: {passage[:600]}\nQ:"
    )


def _parse_completion_qa(text: str) -> Optional[tuple[str, str]]:
    """Parse '<question>\\nA: <answer>' out of a completion that followed 'Q:'."""
    m = re.match(r"\s*(.+?)\n\s*A:\s*(.+)", text, re.S)
    if not m:
        return None
    question = m.group(1).strip().split("\n")[0].strip()
    answer = m.group(2).strip().split("\n")[0].split("Passage:")[0].strip()
    if len(question) >= 8 and len(answer) >= 4:
        return question, answer
    return None


def draft_qa_from_passages(
    bundle: "ModelBundle",
    passages: list[str],
    max_items: int = 8,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[float], None]] = None,
) -> list[dict]:
    """Draft instruction/response pairs from passages using the local base model.

    Returns a list of {instruction, response, context} dicts. Unparseable
    generations are skipped (small models occasionally produce malformed output).
    """
    drafts: list[dict] = []
    n = min(len(passages), max_items)
    for i, passage in enumerate(passages[:n]):
        try:
            out = bundle.complete(_qa_completion_prompt(passage), max_new_tokens=96, temperature=0.7)
            parsed = _parse_completion_qa(out)
        except Exception as e:
            parsed = None
            log(f"[{i + 1}/{n}] error: {e}")
        if parsed:
            q, a = parsed
            drafts.append({"instruction": q, "response": a, "context": passage})
            log(f"[{i + 1}/{n}] ✓ {q[:60]}")
        else:
            log(f"[{i + 1}/{n}] ✗ skipped (couldn't parse a clean Q/A)")
        if progress:
            progress((i + 1) / n)
    return drafts


# ── 2. Fine-tuning (LoRA) ─────────────────────────────────────────────────────
def run_finetuning(
    model_id: str = DEFAULT_MODEL,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    log: Callable[[str], None] = print,
    progress: Optional[Callable[[float], None]] = None,
) -> dict:
    """Fine-tune the chosen base model with a LoRA adapter on the financial data."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    device = get_device()
    log(f"Base model: {model_id}")
    log(f"Device: {device_label()}")

    if not dataset_exists():
        log("No dataset found — building it first.")
        build_dataset(log=log)
    examples = load_examples()
    log(f"Loaded {len(examples)} training examples.")

    tokeniser = AutoTokenizer.from_pretrained(model_id)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    log("Loading base model (first use of a model downloads it — please wait)…")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA attached — training {trainable:,} of {total:,} params ({100 * trainable / total:.2f}%).")

    MAX_LENGTH = 512
    ds = Dataset.from_list(examples)

    def tokenise(batch):
        return tokeniser(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    tokenised = ds.map(tokenise, batched=True, remove_columns=ds.column_names)
    tokenised.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokeniser, mlm=False)
    output_dir = ADAPTERS_DIR / _safe(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    class UICallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                log(f"  step {state.global_step}: loss = {logs['loss']:.4f}")

        def on_step_end(self, args, state, control, **kwargs):
            if progress and state.max_steps:
                progress(min(1.0, state.global_step / state.max_steps))

    training_args = _build_training_args(TrainingArguments, output_dir, epochs, learning_rate, device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised,
        data_collator=data_collator,
        callbacks=[UICallback()],
    )

    log(f"Starting training: {epochs} epoch(s). Only LoRA weights update.")
    result = trainer.train()
    log(f"Training complete. Final loss: {result.training_loss:.4f}")

    out = adapter_dir(model_id)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokeniser.save_pretrained(str(out))

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    log(f"Adapter saved to {out} ({size_mb:.1f} MB).")
    if progress:
        progress(1.0)
    return {"final_loss": float(result.training_loss), "adapter_path": str(out), "size_mb": size_mb}


def _build_training_args(TrainingArguments, output_dir, epochs, learning_rate, device):
    """Construct TrainingArguments robustly across transformers versions."""
    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
    )
    try:
        return TrainingArguments(use_cpu=(device == "cpu"), **common)
    except TypeError:
        try:
            return TrainingArguments(no_cuda=(device == "cpu"), **common)
        except TypeError:
            return TrainingArguments(**common)


# ── 3. Inference / chat ───────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    model: object
    tokeniser: object
    device: str
    model_id: str
    has_adapter: bool

    def _format(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self.tokeniser, "chat_template", None):
            return self.tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Fallback for models without a chat template
        return f"{system}\n\n### Instruction:\n{user}\n\n### Response:\n"

    def chat(
        self,
        user: str,
        system: str = SYSTEM_PROMPT,
        use_adapter: bool = True,
        max_new_tokens: int = 160,
        temperature: float = 0.3,
    ) -> str:
        """Generate a reply. If use_adapter is False and an adapter is loaded,
        the adapter is temporarily disabled to get the base model's answer."""
        import torch

        disable = (
            self.has_adapter and not use_adapter and hasattr(self.model, "disable_adapter")
        )
        ctx = self.model.disable_adapter() if disable else nullcontext()
        formatted = self._format(system, user)
        inputs = self.tokeniser(formatted, return_tensors="pt").to(self.device)
        with ctx, torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokeniser.eos_token_id,
            )
        return self.tokeniser.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def generate_base(self, prompt: str, max_new_tokens: int = 160, temperature: float = 0.3) -> str:
        return self.chat(prompt, use_adapter=False, max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_finetuned(self, prompt: str, max_new_tokens: int = 160, temperature: float = 0.3) -> str:
        return self.chat(prompt, use_adapter=True, max_new_tokens=max_new_tokens, temperature=temperature)

    def complete(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """Raw text completion (no chat template). Used for few-shot data drafting,
        where small models follow a 'Passage→Q/A' pattern better than a chat turn.
        Always uses the base model (adapter disabled)."""
        import torch

        disable = self.has_adapter and hasattr(self.model, "disable_adapter")
        ctx = self.model.disable_adapter() if disable else nullcontext()
        inputs = self.tokeniser(prompt, return_tensors="pt").to(self.device)
        with ctx, torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokeniser.eos_token_id,
            )
        return self.tokeniser.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


def load_model_bundle(model_id: str = DEFAULT_MODEL, log: Callable[[str], None] = print) -> ModelBundle:
    """Load the base model once. If a compatible fine-tuned adapter exists, wrap
    it so we can switch between base and fine-tuned outputs without a second copy.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    log(f"Loading {model_id} on {device_label()} (first use downloads the model)…")
    tokeniser = AutoTokenizer.from_pretrained(model_id)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)

    has_adapter = adapter_exists(model_id)
    if has_adapter:
        from peft import PeftModel

        log("Fine-tuned adapter found for this model — loading it.")
        model = PeftModel.from_pretrained(base, str(adapter_dir(model_id))).to(device)
    else:
        log("No fine-tuned adapter for this model yet — base model only.")
        model = base

    model.eval()
    return ModelBundle(
        model=model, tokeniser=tokeniser, device=device, model_id=model_id, has_adapter=has_adapter
    )


# ── 4. RAG (retrieval-augmented generation) ───────────────────────────────────
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBED_MODEL_ID)
    return _EMBEDDER


def rag_corpus() -> list[str]:
    """The RAG knowledge base.

    Prefers an uploaded document corpus (corpus.json); otherwise falls back to the
    deduplicated context passages from the instruction dataset.
    """
    if corpus_exists():
        try:
            docs = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
            if docs:
                return docs
        except Exception:
            pass

    seen, docs = set(), []
    for ex in load_examples():
        c = ex.get("context")
        if c and c not in seen:
            seen.add(c)
            docs.append(c)
    return docs


def rag_retrieve(query: str, k: int = 3) -> list[dict]:
    """Embed the query and return the k most similar passages (cosine similarity).

    Uses brute-force numpy search — the corpus is small, so no FAISS dependency
    is needed and it stays trivially cross-platform.
    """
    import numpy as np

    docs = rag_corpus()
    if not docs:
        return []
    embedder = _get_embedder()
    doc_vecs = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = doc_vecs @ q_vec
    top = np.argsort(-sims)[: min(k, len(docs))]
    return [{"text": docs[i], "score": float(sims[i])} for i in top]


RAG_SYSTEM = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "information in the context below. If the context does not contain the answer, "
    "say you don't have enough information."
)


def rag_answer(
    bundle: ModelBundle,
    question: str,
    k: int = 3,
    use_adapter: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 0.2,
) -> tuple[str, list[dict]]:
    """Full RAG pipeline: retrieve relevant passages, then answer grounded in them."""
    retrieved = rag_retrieve(question, k=k)
    context = "\n\n".join(f"Source {i + 1}: {r['text']}" for i, r in enumerate(retrieved))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    answer = bundle.chat(
        user,
        system=RAG_SYSTEM,
        use_adapter=use_adapter and bundle.has_adapter,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return answer, retrieved


# ── 5. Evaluation ─────────────────────────────────────────────────────────────
def _keyword_score(text: str, keywords: list[str]) -> float:
    text_l = text.lower()
    hits = sum(1 for k in keywords if k.lower() in text_l)
    return hits / len(keywords) if keywords else 0.0


def _rouge_l(prediction: str, reference: str) -> Optional[float]:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    except Exception:
        return None


def evaluate(bundle: ModelBundle, include_rag: bool = True, log: Callable[[str], None] = print) -> dict:
    """Benchmark base vs fine-tuned (and optionally RAG) on the held-out questions."""
    rows = []
    for item in BENCHMARK:
        q = item["question"]
        log(f"Evaluating: {q}")
        base_ans = bundle.generate_base(q)
        row = {
            "question": q,
            "reference": item["reference"],
            "base_answer": base_ans,
            "base_factmatch": _keyword_score(base_ans, item["keywords"]),
            "base_rougeL": _rouge_l(base_ans, item["reference"]),
        }
        if bundle.has_adapter:
            ft_ans = bundle.generate_finetuned(q)
            row["ft_answer"] = ft_ans
            row["ft_factmatch"] = _keyword_score(ft_ans, item["keywords"])
            row["ft_rougeL"] = _rouge_l(ft_ans, item["reference"])
        if include_rag:
            rag_ans, _ = rag_answer(bundle, q, use_adapter=bundle.has_adapter)
            row["rag_answer"] = rag_ans
            row["rag_factmatch"] = _keyword_score(rag_ans, item["keywords"])
            row["rag_rougeL"] = _rouge_l(rag_ans, item["reference"])
        rows.append(row)

    summary = {"base_factmatch": sum(r["base_factmatch"] for r in rows) / len(rows)}
    if bundle.has_adapter:
        summary["ft_factmatch"] = sum(r["ft_factmatch"] for r in rows) / len(rows)
    if include_rag:
        summary["rag_factmatch"] = sum(r["rag_factmatch"] for r in rows) / len(rows)
    return {"rows": rows, "summary": summary}


# ── 6. Packaging for download ─────────────────────────────────────────────────
def package_adapter(model_id: str = DEFAULT_MODEL) -> Optional[Path]:
    """Zip the fine-tuned adapter (plus a load README) for download."""
    if not adapter_exists(model_id):
        return None

    out = adapter_dir(model_id)
    zip_path = MODELS_DIR / f"finetuned-{_safe(model_id)}.zip"
    (out / "HOW_TO_LOAD.txt").write_text(_load_instructions(model_id), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out.parent))
    return zip_path


def _load_instructions(model_id: str) -> str:
    return f"""How to load this fine-tuned model
==================================

This zip contains a LoRA adapter for {model_id}. It is small because it only
holds the trained adapter weights — the base model downloads automatically from
Hugging Face the first time you load it.

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained("{model_id}")
    model = PeftModel.from_pretrained(base, "final_adapter")  # this folder
    tok = AutoTokenizer.from_pretrained("final_adapter")

    messages = [
        {{"role": "system", "content": "You are a financial analyst assistant."}},
        {{"role": "user", "content": "What was Apple's net sales in fiscal 2023?"}},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    print(tok.decode(model.generate(**inputs, max_new_tokens=160)[0]))
"""
