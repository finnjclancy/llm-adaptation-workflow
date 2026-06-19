# Teaching a Language Model About Finance

This project takes a small, free, open-source AI language model and teaches it to answer questions about company finances. It starts with a general model that knows a little about everything, and ends with a model that has been trained on financial documents, given the ability to look up facts, and properly tested.

The whole thing is built as a series of Jupyter notebooks. Each notebook is one step, runs on its own, and explains what it is doing and why as it goes.

## Who this is for

You do not need a machine learning background to follow along. If you can read Python and you are curious about how tools like ChatGPT are built and customised, you will be fine. Every notebook is written to be read top to bottom like a short lesson.

## Why I built it this way

There are two normal reasons a company cannot just use ChatGPT off the shelf:

- Their data is private and cannot be sent to an outside company
- The model does not know their specific documents, products, or numbers

The fix is to take a model you are allowed to run yourself, and adapt it to your own information. This project shows the full process of doing exactly that, using public financial filings as a stand-in for a company's private documents. Swap in your own documents and the same steps apply.

## The steps, start to finish

```
Company documents (here: public SEC filings)
        |
        v
  02 - Data Preparation      turn raw documents into training material
        |
   -----+-----
   |         |
   v         v
 03 RAG    04 Fine-Tuning     two ways to adapt the model (use one or both)
   |         |
   -----+-----
        |
        v
  05 - Alignment             nudge the model toward better answers
        |
        v
  06 - Evaluation            measure whether any of it actually helped
```

## The notebooks

Run them in order the first time. After that you can jump around.

| Notebook | What it covers | Why it matters |
|----------|----------------|----------------|
| `00_project_overview` | The big picture and how to adapt it | So you know where you are going before you start |
| `01_foundation_models` | What a language model is and how to run one | You cannot adapt a model until you can load and talk to one |
| `01b_neural_networks` | A neural network built from scratch | Shows the simple idea underneath all the fancy libraries |
| `02_data_preparation` | Turning documents into training examples | A model is only as good as the data you feed it |
| `03_rag_pipeline` | Letting the model look facts up | The most reliable way to stop a model making things up |
| `04_fine_tuning` | Training the model on your data (LoRA) | Changes how the model writes and behaves |
| `05_alignment` | Teaching it which answers are better (DPO) | Two technically-correct answers are not equally good |
| `06_evaluation` | Scoring the model before and after | Without this you are just guessing whether it worked |

## What you need to run it

- A Mac with an Apple Silicon chip (M1, M2, M3 or newer). I built and tested it on one.
- 8 GB of memory at a minimum, 16 GB is more comfortable.
- No separate graphics card needed. The notebooks use Apple's built-in GPU, which keeps the cost at zero.

If you are on a different machine, the notebooks also run on Google Colab's free tier. Each one notes the small change you need to make.

## Getting set up

```bash
git clone https://github.com/finnjclancy/llm-adaptation-workflow
cd llm-adaptation-workflow
pip install -r requirements.txt
jupyter lab
```

Then open the `notebooks` folder and start with `00_project_overview`.

## Running it as a clickable app (no notebooks needed)

If you would rather not touch the notebooks, there is a small app in the `app/` folder that does the whole thing through a normal point-and-click screen. It runs entirely on your own machine and works on Windows, Mac, and Linux. It works out by itself whether to use an NVIDIA GPU, Apple Silicon, or just the CPU.

What the app lets you do:

- Pick which base model to use, from a small fast one up to a larger, slower, more capable one.
- Build the dataset, fine-tune the model, and watch a live estimate of how long training will take.
- Chat with the original model and the fine-tuned model side by side to compare them.
- Use the RAG tab to get answers grounded in the source documents, so the figures are correct.
- Download the finished model as a zip when you are done.

To start it, the only thing you need installed first is Python 3.10 or newer. Then:

- On Windows: double-click `start_windows.bat`
- On Mac: double-click `start_mac.command`
- On any system from a terminal: `python run_app.py`

The first launch sets itself up automatically and downloads the model, so it takes a few minutes. After that it is quick. For plain-English, non-technical instructions you can hand to someone else, see [`app/README_FOR_MANAGER.md`](app/README_FOR_MANAGER.md).

## The tools used, and why

- **PyTorch**: the engine that runs the model and does the training maths. It is the standard choice.
- **Hugging Face Transformers**: a library that lets you download and run free models in a few lines. The model I used, TinyLlama, comes from here.
- **peft**: handles the efficient training method (LoRA), so training fits on a laptop instead of needing a server.
- **trl**: handles the alignment step.
- **sentence-transformers**: turns text into numbers so the model can search documents by meaning, not just keywords.
- **evaluate, rouge-score, bert-score**: the scoring tools used in the testing step.

## Making this work for your own documents

Every notebook has a short section marked "Adapt This". To point it at your own area instead of finance:

1. Replace the SEC filings in `02_data_preparation` with your own files (PDFs, web pages, or plain text).
2. Edit the question templates in `utils/templates.py` to match the kind of task you care about.
3. Swap the test questions in `06_evaluation` for ones about your documents.

Everything in between runs the same way.

## A note on the data

I used public financial filings from the US SEC, things like annual reports and earnings figures. That keeps the project legal to share and easy to check, while still looking like the kind of work a bank, fund, or consultancy would actually want done on their private data.
