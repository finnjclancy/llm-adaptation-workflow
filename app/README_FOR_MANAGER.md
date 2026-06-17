# LLM Adaptation Workflow — How to run the app

This is a small app that adapts an open-source AI model to financial-analysis
questions. You can run the whole process, chat with the model, and download the
finished model — all on your own laptop, with nothing sent to the internet
except the one-time model download.

---

## What you need first (one time)

**Python 3.10 or newer must be installed.**

- **Windows:** download from <https://www.python.org/downloads/windows/>.
  On the first install screen, **tick "Add Python to PATH"** before clicking Install.
- **Mac:** download from <https://www.python.org/downloads/macos/>, or it may
  already be installed.

That's the only setup step.

---

## Starting the app

- **Windows:** double-click **`start_windows.bat`**
- **Mac:** double-click **`start_mac.command`**
  - The first time, macOS may say it's from an unidentified developer.
    If so: **right-click the file → Open → Open**.

The first launch takes a few minutes — it sets itself up and downloads the AI
model (~2 GB). After that it's quick. A black window will open and stay open;
**leave it open** while you use the app. Your web browser will open
automatically with the app inside it. To stop the app, just close the black
window.

---

## Using it

In the **sidebar on the left** you can pick which base AI model to use. The
smallest one (**Qwen2.5 0.5B**) is the fastest and best if your laptop has no
graphics card; the larger ones give better answers but take longer.

The app has numbered tabs — go through them in order:

1. **Prepare data** — click *Build dataset*.
2. **Fine-tune** — click *Start fine-tuning* and watch the progress bar. Before
   you start, the app shows an **estimated training time** that updates with the
   model and number of epochs you choose. (On a laptop with no graphics card,
   pick the smallest model for the quickest run.)
3. **Chat & compare** — type a question and see the original model's answer
   next to the newly fine-tuned model's answer, side by side.
4. **RAG (grounded)** — the most accurate option. The app looks up the relevant
   source document and makes the model answer from it, so the figures are
   correct. You'll see the answer *without* RAG next to the answer *with* RAG.
   This tab needs no training, so you can use it straight away.
5. **Evaluate** — scores the models on a set of test questions.
6. **Download** — click to package the fine-tuned model, then download the
   `.zip`. Inside is a `HOW_TO_LOAD.txt` showing how to use it in code.

---

## If something goes wrong

- "Python was not found" — install Python (see above), making sure to tick
  **Add Python to PATH** on Windows, then try again.
- It's slow during fine-tuning — that's normal without a graphics card; the
  progress bar is moving even if it looks paused.
- Anything else — keep the black window open and share what it shows.
