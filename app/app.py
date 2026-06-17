"""
LLM Adaptation Workflow — local frontend.

A clickable version of the notebooks. Launch it with the double-click scripts in
the project root (start_windows.bat / start_mac.command) — no terminal needed.

Tabs:
  Overview        — what this is, machine readout, base-model picker
  1 Prepare data  — build the instruction dataset
  2 Fine-tune     — train the LoRA adapter (with a live time estimate)
  3 Chat & compare— base vs fine-tuned, side by side
  4 RAG (grounded)— retrieve from documents, answer with/without RAG
  5 Evaluate      — benchmark base vs fine-tuned vs RAG
  6 Download      — download the fine-tuned model as a zip
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402

st.set_page_config(page_title="LLM Adaptation Workflow", page_icon="🧠", layout="wide")


# ── Sidebar: base-model picker (shared across all tabs) ───────────────────────
def selected_model() -> str:
    return st.session_state.get("model_id", pl.DEFAULT_MODEL)


with st.sidebar:
    st.header("⚙️ Settings")
    model_ids = list(pl.MODELS.keys())
    st.selectbox(
        "Base model",
        options=model_ids,
        index=model_ids.index(pl.DEFAULT_MODEL),
        format_func=lambda m: pl.MODELS[m]["label"],
        key="model_id",
        help="The open-source model to adapt. Bigger = better answers but slower. "
        "Each model is fine-tuned and downloaded separately.",
    )
    _m = selected_model()
    st.caption(f"`{_m}`  ·  ~{pl.MODELS[_m]['params_b']}B params")
    if pl.adapter_exists(_m):
        st.success("Fine-tuned ✓", icon="✅")
    else:
        st.info("Not yet fine-tuned", icon="ℹ️")
    st.divider()
    st.caption(f"Compute: **{pl.get_device().upper()}**")


# ── Shared helpers ────────────────────────────────────────────────────────────
def status_badges():
    m = selected_model()
    c1, c2, c3 = st.columns(3)
    c1.metric("Machine", pl.get_device().upper())
    c2.metric("Dataset", "Ready" if pl.dataset_exists() else "Not built")
    c3.metric("Fine-tuned", "Ready" if pl.adapter_exists(m) else "Not trained")


@st.cache_resource(show_spinner=False)
def get_bundle(model_id: str, adapter_signature_key: str):
    """Load (and cache) the model bundle for a given base model. The signature key
    busts the cache when that model's adapter is created or retrained. Arg names
    must NOT start with an underscore — Streamlit excludes those from the key."""
    return pl.load_model_bundle(model_id, log=lambda m: None)


def adapter_signature(model_id: str) -> str:
    if not pl.adapter_exists(model_id):
        return "base-only"
    d = pl.adapter_dir(model_id)
    newest = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0)
    return f"adapter-{newest}"


def load_current_bundle():
    m = selected_model()
    with st.spinner(f"Loading {pl.MODELS[m]['label']}…"):
        return get_bundle(m, adapter_signature(m))


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 LLM Adaptation Workflow")
st.caption(
    "Adapt an open-source language model to the financial domain — prepare data, "
    "fine-tune with LoRA, ground answers with RAG, compare against the base model, "
    "and download the result. Everything runs locally on this machine."
)

tab_overview, tab_data, tab_train, tab_chat, tab_rag, tab_eval, tab_download = st.tabs(
    [
        "Overview",
        "1 · Prepare data",
        "2 · Fine-tune",
        "3 · Chat & compare",
        "4 · RAG (grounded)",
        "5 · Evaluate",
        "6 · Download",
    ]
)


# ── Tab: Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("What this does")
    st.markdown(
        """
        This app takes a general-purpose open-source model (pick one in the
        sidebar) and adapts it to **financial-analysis** questions, then lets you
        compare before/after and download the result.

        **Two ways to make the model domain-aware — this app does both:**
        - **Fine-tuning** (tab 2) bakes the domain *style* into the model's weights.
        - **RAG** (tab 4) retrieves the relevant source passage at question time and
          grounds the answer in it — best when you need the *exact facts and figures*.

        **Suggested order:** Prepare data → Fine-tune → Chat & compare → RAG → Evaluate → Download.
        """
    )
    st.divider()
    st.subheader("This machine")
    status_badges()
    st.info(f"Compute device: **{pl.device_label()}**", icon="🖥️")
    if pl.get_device() == "cpu":
        st.warning(
            "No GPU detected. Everything still works, but fine-tuning on CPU is slow. "
            "Pick the **Qwen2.5 0.5B** model in the sidebar for the quickest run, or skip "
            "to **RAG** (which needs no training) for accurate answers.",
            icon="⏳",
        )


# ── Tab: Prepare data ─────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Step 1 — Prepare the instruction dataset")
    st.markdown(
        "Builds the financial Q&A dataset used for both fine-tuning and as the RAG "
        "knowledge base. In the notebooks this comes from real SEC filings; here a "
        "representative sample is included so it runs fully offline."
    )

    if pl.dataset_exists():
        st.success("Dataset already built.", icon="✅")

    if st.button("Build dataset", type="primary"):
        logs = st.empty()
        buffer: list[str] = []

        def log(msg):
            buffer.append(msg)
            logs.code("\n".join(buffer))

        with st.spinner("Building dataset…"):
            result = pl.build_dataset(log=log)
        st.success(f"Built {result['count']} examples.", icon="✅")

    if pl.dataset_exists():
        st.divider()
        st.markdown("**Preview**")
        examples = pl.load_examples()
        st.caption(f"{len(examples)} examples")
        st.dataframe(
            [{"instruction": e["instruction"], "response": e["response"]} for e in examples],
            hide_index=True,
        )


# ── Tab: Fine-tune ────────────────────────────────────────────────────────────
with tab_train:
    m = selected_model()
    st.subheader("Step 2 — Fine-tune with LoRA")
    st.markdown(
        f"Trains a small **LoRA adapter** on top of **{pl.MODELS[m]['label']}** "
        "(change it in the sidebar). Only ~0.5% of the parameters are updated, which "
        "is what makes this possible on a laptop."
    )

    col_a, col_b, col_c = st.columns(3)
    epochs = col_a.slider("Epochs", 1, 8, 3, help="Full passes over the data. More = stronger adaptation, slower.")
    lora_r = col_b.select_slider("LoRA rank", options=[4, 8, 16, 32], value=16, help="Adapter capacity.")
    lr = col_c.select_slider("Learning rate", options=[1e-4, 2e-4, 3e-4], value=2e-4, format_func=lambda x: f"{x:.0e}")

    # Live training-time estimate — updates as the model/epoch sliders change.
    n_ex = pl.num_examples()
    est = pl.estimate_training_label(m, epochs, n_ex, pl.get_device())
    st.info(
        f"**Estimated training time: {est}**  "
        f"({epochs} epoch(s) × {n_ex} examples on {pl.get_device().upper()}). "
        "Rough guide; the first run also downloads the model one time.",
        icon="⏱️",
    )

    if pl.adapter_exists(m):
        st.warning("This model already has a fine-tuned adapter. Training again overwrites it.", icon="ℹ️")
    if not pl.dataset_exists():
        st.warning("No dataset yet — it will be built automatically before training.", icon="⚠️")

    if st.button("Start fine-tuning", type="primary"):
        progress_bar = st.progress(0.0, text="Starting…")
        logs = st.empty()
        buffer: list[str] = []

        def log(msg):
            buffer.append(msg)
            logs.code("\n".join(buffer[-25:]))

        def progress(frac):
            progress_bar.progress(min(1.0, frac), text=f"Training… {frac * 100:.0f}%")

        try:
            result = pl.run_finetuning(
                model_id=m, epochs=epochs, learning_rate=float(lr), lora_r=lora_r, log=log, progress=progress
            )
            progress_bar.progress(1.0, text="Done")
            st.success(
                f"Fine-tuning complete. Final loss {result['final_loss']:.4f}, "
                f"adapter size {result['size_mb']:.1f} MB.",
                icon="✅",
            )
            st.cache_resource.clear()  # force reload with the new adapter
            st.balloons()
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)


# ── Tab: Chat & compare ───────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Step 3 — Ask a question")
    st.markdown("See how the **base** model and the **fine-tuned** model answer the same prompt.")

    examples = [
        "What was Apple's total net sales in fiscal year 2023?",
        "How did Microsoft Azure perform in 2023?",
        "What drove JPMorgan Chase's net income in 2023?",
        "Summarise Apple's main business risks.",
    ]
    picked = st.selectbox("Example questions", ["— type your own —"] + examples, key="chat_pick")
    default_q = "" if picked == "— type your own —" else picked
    question = st.text_input("Question", value=default_q, placeholder="Ask a financial-analysis question…", key="chat_q")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1, key="chat_temp")

    if st.button("Generate answers", type="primary", disabled=not question.strip()):
        bundle = load_current_bundle()
        col_base, col_ft = st.columns(2)
        with col_base:
            st.markdown("#### 🟦 Base model")
            with st.spinner("Generating…"):
                st.write(bundle.generate_base(question, temperature=temperature))
        with col_ft:
            st.markdown("#### 🟩 Fine-tuned model")
            if bundle.has_adapter:
                with st.spinner("Generating…"):
                    st.write(bundle.generate_finetuned(question, temperature=temperature))
            else:
                st.info("No fine-tuned model yet — run **Fine-tune** first to compare.", icon="ℹ️")


# ── Tab: RAG ──────────────────────────────────────────────────────────────────
with tab_rag:
    st.subheader("Step 4 — RAG: answers grounded in the documents")
    st.markdown(
        "RAG retrieves the most relevant source passage for your question and makes "
        "the model answer **using only that passage**. This is how you get accurate "
        "figures — the model isn't relying on memory. Compare the two columns below."
    )

    rag_examples = [
        "What was Apple's total net sales in fiscal year 2023?",
        "How fast did Microsoft Azure grow in fiscal 2023?",
        "What was JPMorgan Chase's net income in 2023?",
        "What was Apple's iPhone revenue in fiscal 2023?",
    ]
    picked_r = st.selectbox("Example questions", ["— type your own —"] + rag_examples, key="rag_pick")
    default_r = "" if picked_r == "— type your own —" else picked_r
    rag_q = st.text_input("Question", value=default_r, placeholder="Ask about the financial documents…", key="rag_q")

    col_k, col_ft = st.columns(2)
    k = col_k.slider("Passages to retrieve (top-k)", 1, 5, 3, key="rag_k")
    use_ft = col_ft.toggle(
        "Use fine-tuned model", value=pl.adapter_exists(selected_model()), key="rag_use_ft",
        help="Combine RAG with your fine-tuned model. Off = base model.",
    )

    if st.button("Answer with RAG", type="primary", disabled=not rag_q.strip()):
        bundle = load_current_bundle()

        st.markdown("##### 🔎 Retrieved sources")
        sources = pl.rag_retrieve(rag_q, k=k)
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}]** _(similarity {s['score']:.2f})_  {s['text']}")

        st.divider()
        col_no, col_yes = st.columns(2)
        with col_no:
            st.markdown("#### 🟥 Without RAG (model memory)")
            st.caption("The model answers from what it memorised — may be wrong.")
            with st.spinner("Generating…"):
                st.write(bundle.chat(rag_q, use_adapter=use_ft and bundle.has_adapter, temperature=0.2))
        with col_yes:
            st.markdown("#### 🟩 With RAG (grounded in sources)")
            st.caption("The model answers using only the retrieved passages.")
            with st.spinner("Generating…"):
                answer, _ = pl.rag_answer(bundle, rag_q, k=k, use_adapter=use_ft and bundle.has_adapter)
                st.write(answer)


# ── Tab: Evaluate ─────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Step 5 — Evaluate")
    st.markdown(
        "Runs held-out benchmark questions through the models and scores a simple "
        "**fact-match** (did the answer contain the right figures). Compares base, "
        "fine-tuned, and RAG."
    )
    include_rag = st.toggle("Include RAG in the comparison", value=True, key="eval_rag")

    if st.button("Run evaluation", type="primary"):
        bundle = load_current_bundle()
        logs = st.empty()
        buffer: list[str] = []

        def log(msg):
            buffer.append(msg)
            logs.code("\n".join(buffer[-8:]))

        with st.spinner("Evaluating…"):
            results = pl.evaluate(bundle, include_rag=include_rag, log=log)
        logs.empty()

        summary = results["summary"]
        cols = st.columns(len(summary))
        labels = {"base_factmatch": "Base", "ft_factmatch": "Fine-tuned", "rag_factmatch": "RAG"}
        base_val = summary["base_factmatch"]
        for col, (key, val) in zip(cols, summary.items()):
            delta = None if key == "base_factmatch" else f"{(val - base_val) * 100:+.0f} pts"
            col.metric(labels.get(key, key), f"{val * 100:.0f}%", delta=delta)

        st.divider()
        for r in results["rows"]:
            with st.expander(r["question"]):
                st.caption(f"Reference: {r['reference']}")
                st.markdown("**🟦 Base**")
                st.write(r["base_answer"])
                if "ft_answer" in r:
                    st.markdown("**🟩 Fine-tuned**")
                    st.write(r["ft_answer"])
                if "rag_answer" in r:
                    st.markdown("**🟪 RAG**")
                    st.write(r["rag_answer"])


# ── Tab: Download ─────────────────────────────────────────────────────────────
with tab_download:
    m = selected_model()
    st.subheader("Step 6 — Download the fine-tuned model")
    st.markdown(
        "Packages the fine-tuned model as a single zip. It's small because it contains "
        "only the trained **LoRA adapter** — the base model downloads automatically when "
        "loaded. A `HOW_TO_LOAD.txt` with example code is included inside."
    )
    st.caption(f"Currently selected model: **{pl.MODELS[m]['label']}**")

    if not pl.adapter_exists(m):
        st.info("Nothing to download for this model yet — run **Fine-tune** first.", icon="ℹ️")
    else:
        if st.button("Package model for download", type="primary"):
            with st.spinner("Zipping…"):
                zip_path = pl.package_adapter(m)
            st.session_state["zip_path"] = str(zip_path)

        if st.session_state.get("zip_path"):
            zip_path = Path(st.session_state["zip_path"])
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / 1e6
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download fine-tuned model (.zip)",
                        data=f,
                        file_name=zip_path.name,
                        mime="application/zip",
                        type="primary",
                    )
                st.caption(f"{zip_path.name} · {size_mb:.1f} MB")
