# How to run the app

This is a small app that teaches a free AI model to answer finance questions. You can run the whole process, chat with the model, and download the finished model, all on your own laptop. Nothing is sent over the internet except a one-time download of the model itself.

You do not need to know any coding to use it.

## What you need first (one time only)

Python 3.10 or newer has to be installed on the computer.

- On Windows: download it from https://www.python.org/downloads/windows/. On the very first install screen, tick the box that says "Add Python to PATH" before you click Install. This step matters.
- On Mac: download it from https://www.python.org/downloads/macos/. It may already be installed.

That is the only setup you have to do yourself.

## Starting the app

- On Windows: double-click the file called `start_windows.bat`.
- On Mac: double-click the file called `start_mac.command`. The first time, the Mac might warn that the file is from an unidentified developer. If it does, right-click the file, choose Open, then Open again.

A few things to expect on the first launch:

- It takes a few minutes the first time. The app sets itself up and downloads the AI model, which is about 2 GB.
- A black window will open and stay open. Leave it open the whole time you are using the app. Closing it stops the app.
- Your web browser will open by itself with the app inside it. That is where you actually use it.

After the first run, starting it again is quick.

## Using it

On the left there is a sidebar where you choose which base model to use. The smallest one (Qwen2.5 0.5B) is the fastest, and it is the best choice if the laptop has no separate graphics card. The bigger models give better answers but take longer.

The app has numbered tabs. Go through them in order:

1. Prepare data. Click "Build dataset" to get the training material ready.
2. Fine-tune. Click "Start fine-tuning" and watch the progress bar. Before you start, the app shows an estimated training time that changes with the model and the number of passes you pick. On a laptop with no graphics card, choose the smallest model for the quickest run.
3. Chat and compare. Type a question and see the original model's answer next to the freshly trained model's answer, side by side.
4. RAG (grounded). This is the most accurate option. The app looks up the relevant source document and makes the model answer from it, so the figures come out right. You will see the answer without RAG next to the answer with RAG. This tab needs no training, so you can use it straight away.
5. Evaluate. This scores the models on a set of test questions.
6. Download. Click to package up the trained model, then download the zip file. Inside it there is a short text file called HOW_TO_LOAD.txt that explains how to use the model in code.

## If something goes wrong

- "Python was not found": install Python using the steps above. On Windows, make sure you ticked "Add Python to PATH". Then try again.
- It feels slow during fine-tuning: that is normal on a laptop with no graphics card. The progress bar is still moving even when it looks stuck.
- Anything else: keep the black window open and share whatever it is showing. That text usually says what the problem is.
