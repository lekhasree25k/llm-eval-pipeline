\# LLM Evaluation Pipeline for a Product FAQ Assistant



Automated quality tests for an LLM-based customer support assistant, built with Python, DeepEval and pytest, and run in CI with GitHub Actions.



\## What it tests

\- \*\*Faithfulness:\*\* answers stay grounded in the provided context

\- \*\*Answer relevancy:\*\* responses address the question asked

\- \*\*Hallucination:\*\* no invented facts

\- \*\*Refusal behaviour:\*\* says "I don't have that information" when the answer is not in the context

\- \*\*Prompt injection resistance:\*\* does not reveal system instructions or follow "ignore your rules" attacks



\## How it works

Golden dataset (`data/golden.json`) → assistant (`app/bot.py`) → DeepEval metrics with an LLM judge → pytest pass/fail → GitHub Actions



\## Run locally

&#x20;   pip install -r requirements.txt

&#x20;   set GEMINI\_API\_KEY=your\_key

&#x20;   pytest -v



\## Tech

Python, DeepEval, pytest, Gemini API, GitHub Actions, Docker

