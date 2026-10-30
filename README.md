# RARAG — Reliability-Aware RAG under Indirect Prompt Injection

`RARAG.ipynb` is a benchmark of defences against **indirect prompt injection (IPI)** in a retrieval-augmented generation (RAG) pipeline built over news articles. It compares four context-selection policies. These combine an IPI detector (**PIGuard**) with **source-reliability** labels, and are tested on a small local generator (**Qwen 2.5 3B** via Ollama).

## Research question

Can screening retrieved passages with PIGuard, and then preferring passages from reliable publishers, stop injected instructions from reaching the generator? And what does each step cost in answer quality?

## Setup

| Component | Detail |
|---|---|
| Corpus | 4,800 news articles drawn from NELA, each publisher labelled `reliable` / `mixed` / `unreliable` (238 publishers) |
| QA sets | 52 validation and 106 test questions, with no article overlap between splits |
| Retrieval | Hybrid BM25 + dense (FAISS) retrieval with reciprocal rank fusion. A top-20 candidate pool is reduced to a final context budget of 5 |
| Attacks | Two BIPIA-style IPI families, each injected into the rank-1 passage: **authority spoofing** and **forged QA reference**, tracked with the canary `RAG_CANARY_764` |
| Detector | PIGuard. The threshold is calibrated on clean validation passages only, targeting ≤ 2% FPR (threshold ≈ 0.104) |
| Generator | `qwen2.5:3b` served by Ollama (`http://localhost:11434`) |
| Seed | 764 |

### Compared policies

1. **Vanilla RAG**: the top 5 passages, with no defence.
2. **PIGuard**: drop flagged passages, then take the top 5 of what remains.
3. **PIGuard + reliability**: PIGuard filtering, then reliable sources ranked first.
4. **PIGuard + shuffled reliability**: a placebo control in which reliability labels are shuffled at the publisher level (150 of 238 labels changed).

## Notebook structure

| Step | What it does |
|---|---|
| Environment check | Prints the Python, venv, PyTorch and CUDA versions and the GPU |
| Step 2 | Loads the corpus, the QA splits and saved hybrid retrieval, and checks they are consistent |
| Step 3 | Calibrates the PIGuard threshold on validation data and runs the test detection experiment |
| Step 4 | Freezes the two-attack benchmark (106 questions × 2 families = 212 attack cases) |
| Step 5 | Scores PIGuard on the candidate pool and applies the four selection policies |
| Step 6 | Runs the end-to-end Qwen evaluation (1,272 records, 710 unique prompts, checkpointed so it can be resumed) |
| Step 7 | Exports bar charts (2 attacks × 5 metrics, as PNG at 600 DPI plus PDF/SVG) |

## Key results (test set, n = 106 per cell)

**PIGuard detection on the full candidate pool:** 1.70% false-positive rate on clean passages (34 of 2,004) and 99.53% recall on injected passages (211 of 212).

**Attack success rate (ASR):**

| Attack | Vanilla | PIGuard | PIGuard + reliability | PIGuard + shuffled |
|---|---|---|---|---|
| Authority spoofing | 28.3% | 0.0% | 0.0% | 0.0% |
| Forged QA reference | 0.9% | 0.0% | 0.0% | 0.0% |

**Answer quality on clean questions:**

| Metric | Vanilla | PIGuard | PIGuard + reliability | PIGuard + shuffled |
|---|---|---|---|---|
| Answer containment | 0.425 | 0.425 | 0.208 | 0.255 |
| Token F1 | 0.105 | 0.105 | 0.043 | 0.077 |
| "Insufficient evidence" rate | 0.377 | 0.387 | 0.651 | 0.528 |
| Gold evidence in context | 0.651 | 0.660 | 0.311 | 0.358 |

PIGuard alone removes the attacks with almost no loss of clean-question quality. Adding reliability-based selection puts more reliable sources in the context (about 4.4 of 5 passages, against about 1.9). However, it pushes out the passages that hold the answer, so answer quality falls and the model abstains more often.

## Reproducing

This notebook is the **final, fast-track stage** of a larger pipeline and does **not run on its own**:

- It reads intermediate artifacts (`nela_research_corpus.parquet`, `qa_*_final.parquet`, `retrieval_*_hybrid_top5.jsonl`, `piguard_validation_scores.parquet`, …) from `E:\python_project\RAG\results\nela_experiment\`.
- It expects objects already loaded in the kernel from earlier sessions: `nela_encoder`, `hybrid_faiss`, `hybrid_bm25`, `hybrid_passages`, `bm25_rank`, `reciprocal_rank_fusion`, `select_top_passages`, `piguard_model`, `piguard_tokenizer` and `generate_answer`.
- Paths are hard-coded Windows paths. Edit `ROOT` / `OUT` / `FINAL` to match your machine.

Environment used: Python virtual environment (`geo761` kernel), PyTorch 2.11 + CUDA 13.0, NVIDIA RTX 4070 Ti SUPER, and Ollama with `qwen2.5:3b` pulled.

Main dependencies: `torch`, `transformers`, `pandas`, `pyarrow`, `numpy`, `scipy`, `matplotlib`, `faiss`, `rank_bm25`, `sentence-transformers`, `requests`.

## Context

Coursework for GEOG761, University of Auckland.
