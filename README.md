# 🚀 JobPilot — Smart Job Matcher & Resume Builder

**BAX-423 Big Data | Spring 2026 | Final Project Option B**

> Upload your profile → get ranked job matches → generate a tailored resume

**Live App:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)
*(Replace with your Streamlit Community Cloud URL after deployment)*

---

## What JobPilot Does

JobPilot is a full-stack intelligent job-matching application that:

1. **Ingests** job postings from the Kaggle TechMap dataset + optional live JSearch API
2. **Deduplicates** using MinHash LSH (detects near-duplicate postings from multiple boards)
3. **Matches** your profile to jobs via dense vector embeddings (sentence-transformers + FAISS)
4. **Ranks** results through a 4-stage pipeline: hard filters → semantic scoring → skill match → MMR re-ranking
5. **Learns** from your feedback using Thompson Sampling bandit + online weight adjustment
6. **Generates** a tailored, ATS-optimized resume for any selected role via GPT-4o-mini
7. **Explains** why each job ranked where it did (matched skills, score breakdown, salary fit)
8. **Visualizes** market analytics: top skills, salary distribution, remote split, skill gaps

---

## Quick Start (Local)

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/jobpilot.git
cd jobpilot
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
# JSEARCH_API_KEY=your_rapidapi_key (optional)
# KAGGLE_USERNAME=... and KAGGLE_KEY=... (optional — for Kaggle download)
```

### 3. Run
```bash
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## First Run (No Kaggle Setup)

If you don't have Kaggle credentials configured, JobPilot automatically generates a realistic **5,000-record synthetic dataset** for demo purposes. The full pipeline (dedup → embeddings → ranking) works identically on synthetic data.

To use the real Kaggle dataset:
1. Get your Kaggle API key from [kaggle.com/account](https://www.kaggle.com/account)
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to `.env`
3. Or place `kaggle.json` in `~/.kaggle/`
4. Select "Offline dataset (Kaggle snapshot)" in the Profile Setup page

---

## Kaggle Data
Dataset: [TechMap International Job Postings (Sept 2021)](https://www.kaggle.com/datasets/techmap/international-job-postings-september-2021)

The dataset is automatically downloaded via `kagglehub` at runtime. A cleaned snapshot is saved to `data/jobs_clean.parquet` for subsequent runs.

---

## Test the Four Personas
```bash
pytest tests/test_personas.py -v
```

| Persona | Test Focus |
|---|---|
| Aisha — ML Pivoter | No Senior/Staff roles; no defense companies |
| Marcus — New Grad | No 3+ year requirements; no contract-only |
| Priya — Senior Engineer | No Junior roles; salary ≥ $200K enforced |
| Kenji — Visa Constrained | No contract roles; visa sponsorship filtered |

---

## BAX-423 Techniques Used

| Technique | Component | Benchmarked Against |
|---|---|---|
| **MinHash LSH** (Sketching) | Deduplication | Exact string match — 11× faster, detects paraphrase dupes |
| **Dense Embeddings + FAISS** (Embeddings) | Job Retrieval | TF-IDF — +32pp Recall@10 |
| **Thompson Sampling** (Reinforcement Learning) | Adaptive Learning | Cold start — +85% Precision@5 after 30 signals |

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include `data/jobs_sample.parquet`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   JSEARCH_API_KEY = "your_key"
   OFFLINE_MODE = "true"
   ```
5. Click **Deploy** — live in ~2 minutes

> **Note:** Set `OFFLINE_MODE=true` for Streamlit Cloud so the app uses the bundled sample data instead of downloading from Kaggle at runtime.

---

## Project Structure

```
jobpilot/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── README.md
├── prompts.md                # All AI prompts used
│
├── data/
│   ├── personas.json         # 4 test personas
│   └── jobs_sample.parquet   # 5,000-row offline sample
│
├── src/
│   ├── utils.py              # Shared utilities and constants
│   ├── ingest.py             # Kaggle + JSearch data ingestion
│   ├── clean.py              # Cleaning and feature extraction
│   ├── dedupe.py             # MinHash LSH deduplication
│   ├── embeddings.py         # Sentence-transformers + FAISS
│   ├── ranker.py             # Multi-stage ranking pipeline
│   ├── adaptive_learning.py  # Thompson Sampling + weight updater
│   ├── resume_generator.py   # GPT-4o-mini resume tailoring
│   └── analytics.py          # Batch analytics + Plotly charts
│
├── tests/
│   └── test_personas.py      # Automated persona pass/fail suite
│
└── outputs/                  # Benchmark results, sample resumes
```

---

## Known Limitations

- **Salary sparsity:** ~38% of Kaggle records have no salary data. Salary filter applies only when data exists.
- **Seniority extraction:** Regex-based; struggles with titles like "Engineer II" or "Associate II".
- **Visa detection:** Relies on keyword matching; cannot verify actual H-1B sponsorship history.
- **Session-only feedback:** Adaptive learning resets on page reload (in-memory storage).
- **OpenAI rate limits:** Free tier may throttle resume generation; template fallback is used automatically.

---

## Dependencies

| Package | Purpose |
|---|---|
| streamlit | Web UI |
| sentence-transformers | Dense embeddings |
| faiss-cpu | Approximate nearest-neighbour search |
| datasketch | MinHash LSH deduplication |
| openai | Resume generation (GPT-4o-mini) |
| pandas + pyarrow | Data processing |
| plotly | Analytics visualizations |
| scikit-learn | TF-IDF baseline + metrics |
| kagglehub | Kaggle dataset download |
| requests | JSearch API calls |
