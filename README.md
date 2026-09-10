# SpotifyCares AI Support Agent

An AI-powered customer support agent for SpotifyCares (Spotify's Twitter support) that classifies intents, drafts grounded replies, and makes escalation decisions.

Built for the **Hiver SDE Intern Take-Home Assignment**.

---

## Quick Start (Reproduce Results in <15 min)

### Prerequisites
- Python 3.10+
- A Groq API key ([get one free](https://console.groq.com/keys))
- ~500MB disk space (dataset + models)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Abhijith-Sura/hiver-support-agent.git
cd hiver-support-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Download & Preprocess Data (~5 min)

```bash
# Download dataset from Kaggle (requires kaggle auth — see below)
python -m src.data.download

# Preprocess: filter to SpotifyCares, reconstruct threads, clean text
python -m src.data.preprocess

# Create golden evaluation set (stratified sample of 200 examples)
python -m src.data.sample
```

**Kaggle Authentication**: You need a Kaggle account. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token. This downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows).

### Run the Pipeline (~1 min for single message)

```bash
# Single message
python -m src.pipeline --input "My Spotify keeps crashing after the latest update"

# Batch processing
python -m src.pipeline --batch evaluation/golden_set/golden_set.jsonl --output results.jsonl
```

### Run Evaluation (< 15 min)

```bash
# 1. Instant reproduction from cached predictions (< 1 min)
python -m src.evaluation.run_eval --use-cached

# 2. Live evaluation run on 50-example benchmark sample (~5-10 min)
python -m src.evaluation.run_eval --sample-size 50

# 3. Full evaluation across all 200 golden set examples
python -m src.evaluation.run_eval
```

Results and charts are saved to `evaluation/results/`.

### Run Unit Tests (< 5s)

```bash
pytest tests/ -v
```

---

## Architecture

```
Customer Message
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│   Intent     │     │  Reply Retriever │
│  Classifier  │     │  (Sentence BERT) │
│ (Qwen/Groq)  │     │  Top-5 similar   │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────────┐
│       Reply Generator (Qwen/Groq)    │
│  Grounded in retrieved examples +     │
│  brand voice + thread context         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│      Escalation Decider              │
│  Rules + LLM hybrid                  │
│  → AUTO_HANDLE or ESCALATE           │
└──────────────────────────────────────┘
```

### Components

| Component | Method | File |
|-----------|--------|------|
| Intent Classification | Qwen 2.5/3.8 via Groq with few-shot taxonomy | `src/intent/classifier.py` |
| Baseline 1 | Random (weighted by distribution) | `src/intent/classifier.py` |
| Baseline 2 | TF-IDF + Logistic Regression | `src/intent/classifier.py` |
| Reply Retrieval | Sentence-BERT (`all-MiniLM-L6-v2`) + cosine similarity | `src/reply/retriever.py` |
| Reply Generation | Qwen via Groq with RAG grounding | `src/reply/generator.py` |
| Escalation | Hybrid rule-based + LLM for borderline | `src/escalation/decider.py` |
| Evaluation | Automated metrics + LLM-as-judge | `src/evaluation/` |

---

## Project Structure

```
hiver-support-agent/
├── README.md                 # This file
├── REPORT.md                 # Detailed evaluation report (max 6 pages)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── data/
│   ├── raw/                  # twcs.csv (gitignored, ~300MB)
│   └── processed/            # Cleaned SpotifyCares conversations
├── src/
│   ├── config.py             # All configuration in one place
│   ├── pipeline.py           # End-to-end orchestration
│   ├── data/                 # Download, preprocess, sample
│   ├── intent/               # Taxonomy + classifiers (3 methods)
│   ├── reply/                # Retriever + generator
│   ├── escalation/           # Decision logic
│   └── evaluation/           # Metrics + LLM judge + runner
├── evaluation/
│   ├── golden_set/           # 200 hand-labelled examples + guide
│   └── results/              # Evaluation outputs (gitignored)
└── tests/                    # Unit tests
```

---

## Evaluation Overview

See [REPORT.md](REPORT.md) for the full analysis. Measured results from our benchmark evaluation:

| Metric | LLM Agent (Ours) | TF-IDF Baseline | Random Baseline |
|---|:---:|:---:|:---:|
| **Intent Accuracy** | **72.0%** | 100.0%* | 20.0% |
| **Intent Macro-F1** | **0.7959** | 1.0000* | 0.0919 |
| **ROUGE-L (Reply)** | **0.4596** | — | — |
| **BLEU (Reply)** | **0.1417** | — | — |
| **BERTScore F1** | **0.4691** | — | — |
| **LLM Judge (Overall)** | **4.79 / 5.0** | — | — |
| **Escalation False Negative Rate** | **0.0% (benchmark sample)**† | — | — |

*\*Note: TF-IDF baseline was fit on the test slice to demonstrate data leakage ceiling. See REPORT.md §5 for full discussion.*
*†Note: In the 25-message benchmark sample, all messages were routine support (`auto_handle`). Across the full 200-item golden set with 34 real escalations, our engine achieves a 20.6% FNR (~80% caught) and 30.1% FPR. See REPORT.md §2.*

---

## Key Files

- **Golden Set**: `evaluation/golden_set/golden_set.jsonl` — 200 hand-labelled examples
- **Labelling Guide**: `evaluation/golden_set/LABELLING_GUIDE.md` — sampling & labelling protocol
- **Report**: `REPORT.md` — problem framing, results, failure analysis, decision log

---

## Tools & Citations

- **LLM**: Groq Cloud API (`qwen/qwen3.8-27b`) via `groq` SDK
- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, ThoughtVector)
- **Embeddings**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence-Transformers)
- **Metrics & Testing**: scikit-learn, scipy, rouge-score, bert-score, pytest
- **AI Assistants**: Code written with assistance from Claude/Gemini AI coding assistants
