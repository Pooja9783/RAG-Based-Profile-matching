# RAG-Based Resume Profile Matching

A **Retrieval-Augmented Generation (RAG)** system that automatically matches and ranks candidate resumes against job descriptions using semantic search and hybrid scoring — built entirely in Node.js with no external API keys required.

---

## What It Does

- Reads PDF resumes and splits them into smart chunks
- Converts each chunk into a vector embedding using a local AI model
- Stores all embeddings in a local vector database
- Takes a job description as input and finds the most relevant candidates
- Ranks candidates using a hybrid score combining semantic similarity + skill matching
- Filters out candidates who don't meet must-have skill or experience requirements
- Generates per-candidate reasoning explaining why they matched or didn't

---

## Project Structure

```
RAG-Based-Profile-matching/
│
├── resumes/                        # 30 candidate PDF resumes
│
├── job_descriptions/               # 5 job description JSON files
│   ├── jd1_fullstack_engineer.json
│   ├── jd2_frontend_developer.json
│   ├── jd3_backend_nodejs.json
│   ├── jd4_mern_junior.json
│   └── jd5_senior_fullstack_lead.json
│
├── utils/
│   ├── chunks.js                   # PDF loading + sentence-based chunking
│   └── metaDataExtractor.js        # Extracts name, skills, experience, education
│
├── charts/                         # Auto-generated analysis charts (run analysis.js)
│   ├── chart_1_dataset.png
│   ├── chart_2_embeddings.png
│   ├── chart_3_retrieval.png
│   ├── chart_4_latency.png
│   ├── chart_5_chunking.png
│   └── chart_6_hybrid.png
│
├── resume_rag.js                   # Part A — builds the vector database
├── job_matcher.js                  # Part B — semantic search + hybrid scoring
├── analysis.js                     # Generates performance metrics + charts
├── vectorDB.json                   # Generated vector store (run resume_rag.js)
├── analysis_report.json            # Generated metrics report (run analysis.js)
└── package.json
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Node.js | Runtime |
| @xenova/transformers | Local MiniLM embedding model |
| pdfjs-dist | PDF text extraction |
| chart.js + canvas | Chart generation for analysis |
| vectorDB.json | Lightweight local vector store |

**Embedding model:** `Xenova/all-MiniLM-L6-v2`
- 384 dimensions
- Runs fully locally — no API key needed
- No cost, no rate limits

---

## How It Works

```
PDF Resumes
     │
     ▼
Text Extraction  (pdfjs-dist)
     │
     ▼
Sentence-Based Chunking  (max 400 chars, 80 char overlap)
     │
     ▼
Metadata Extraction  (name, skills, experience years, education)
     │
     ▼
Embedding Generation  (MiniLM — 384 dimensions per chunk)
     │
     ▼
Vector Store  (vectorDB.json)
     │
     ▼
Job Description → Embedding
     │
     ▼
Cosine Similarity Search  (all chunks scored)
     │
     ▼
Must-Have Skill + Experience Filtering
     │
     ▼
Hybrid Scoring: 0.7 × Semantic + 0.3 × Skill Match
     │
     ▼
Ranked Candidates with Reasoning
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pooja9783/RAG-Based-Profile-matching
cd RAG-Based-Profile-matching
```

### 2. Install dependencies

```bash
npm install
```

### 3. For analysis charts only — install these extra packages

```bash
npm install canvas chart.js
```

---

## Running the Project

### Step 1 — Build the vector database

```bash
node resume_rag.js
```

This will:
- Read all PDFs from the `resumes/` folder
- Split text into sentence-aware chunks
- Extract metadata (name, skills, experience, education) from each resume
- Generate embeddings using the MiniLM model
- Save everything to `vectorDB.json`

> The first run downloads the MiniLM model (~25MB). This takes 1–2 minutes. Subsequent runs are fast.

---

### Step 2 — Match a single job description

Edit the `jobDescription` and `mustHaveSkills` at the bottom of `job_matcher.js`, then:

```bash
node job_matcher.js
```

Results are printed to the console and saved to `match_results.json`.

---

### Step 3 — Match all 5 job descriptions at once

```bash
node job_matcher.js --all
```

This loops through every file in `job_descriptions/` and saves a separate result file for each one (`results_JD001.json`, `results_JD002.json`, etc.).

---

### Step 4 — Generate analysis charts and metrics

```bash
node analysis.js
```

This generates 6 PNG charts and `analysis_report.json` in your project root.

---

## Example Output

```json
{
  "job_description": "Looking for a Senior React and Node.js developer...",
  "must_have_skills": ["React", "Node.js"],
  "total_candidates_evaluated": 30,
  "candidates_after_filter": 25,
  "top_matches": [
    {
      "candidate_name": "Arjun Mehta",
      "resume_path": "resumes/resume_1.pdf",
      "match_score": 88,
      "semantic_score": 0.81,
      "skill_match_score": 1.0,
      "experience_years": 6,
      "matched_skills": ["React", "Node.js", "MongoDB", "REST API"],
      "relevant_excerpts": ["Software Engineer with 6+ years of experience..."],
      "reasoning": "Matched 4/4 required skills: React, Node.js, MongoDB, REST API. Candidate has 6 years of experience. Meets the 5+ years requirement. Strong semantic alignment with job description (81%)."
    }
  ]
}
```

---

## Analysis & Results

### Dataset Overview
Experience distribution across 30 candidates and top skills frequency.

![Dataset Overview](charts/chart_1_dataset.png)

---

### Embedding Quality
Pairwise cosine similarity distribution across all 435 resume pairs and a 10×10 similarity heatmap.

![Embedding Quality](charts/chart_2_embeddings.png)

---

### Retrieval Accuracy
Precision@5 and Precision@10 measured across all 5 job descriptions.

![Retrieval Accuracy](charts/chart_3_retrieval.png)

---

### Performance Metrics
Average latency and throughput for DB load, similarity search, and full pipeline.

![Latency Benchmarks](charts/chart_4_latency.png)

---

### Chunking Strategy Comparison
Sentence-based chunking vs fixed-size chunking — size distribution and summary stats.

![Chunking Comparison](charts/chart_5_chunking.png)

---

### Hybrid Scoring Analysis
Per-candidate comparison of pure semantic score vs hybrid score (0.7 semantic + 0.3 skill).

![Hybrid Scoring](charts/chart_6_hybrid.png)

---

## Performance Summary

| Metric | Value |
|---|---|
| Total resumes | 30 |
| Total chunks | 180 |
| Avg chunks per resume | 6 |
| Embedding dimensions | 384 |
| Avg pairwise similarity | 0.712 |
| DB load latency | ~50ms |
| Similarity search (180 chunks) | ~8ms |
| Full pipeline latency | ~8ms |
| Embedding model | Xenova/all-MiniLM-L6-v2 |
| Scoring formula | 0.7 × semantic + 0.3 × skill match |

---

## Key Design Decisions

**Why sentence-based chunking?**
Sentence-based chunks preserve semantic context better than fixed-size chunks. A fixed 400-char split can cut mid-sentence, breaking meaning. Our chunker splits on sentence boundaries with 80-char overlap so no context is lost at chunk edges.

**Why MiniLM locally instead of OpenAI?**
`Xenova/all-MiniLM-L6-v2` runs fully locally — no API key, no cost, no rate limits. It produces 384-dimensional embeddings that are fast to compute and accurate enough for resume matching.

**Why hybrid scoring (70/30)?**
Pure semantic search can rank a candidate highly because their resume text sounds similar to the JD, even if they lack critical skills. Adding a 30% skill match component ensures candidates with the right technologies rank above those who just "sound" similar.

**Why JSON instead of ChromaDB/Pinecone?**
For 30 resumes and 180 chunks, a JSON file is fast, portable, and requires zero setup. The same cosine similarity logic works identically with a real vector DB — swapping to ChromaDB is a one-file change if needed at scale.

---

## Assignment Requirements Covered

| Requirement | Status |
|---|---|
| Document chunking and embedding | Sentence-based chunker with 80-char overlap |
| Vector database | JSON vector store with embeddings + metadata |
| Retrieval pipeline | Cosine similarity search + top-K retrieval |
| Semantic search | MiniLM embeddings + cosine similarity |
| Metadata extraction | Name, skills, experience years, education |
| Hybrid search | Semantic + keyword skill matching |
| Score 0–100 | Normalised final score |
| Match reasoning | Per-candidate dynamic reasoning |
| Must-have filtering | Excludes candidates missing required skills |
| 30+ resumes | 30 resumes in dataset |
| 5+ job descriptions | 5 JD files across different roles and levels |
| Performance metrics | Latency benchmarks + retrieval accuracy charts |
| Analysis | analysis.js with 6 charts + analysis_report.json |

---

## Future Improvements

- Replace JSON vector store with ChromaDB or Pinecone for production scale
- Use an LLM to generate richer match reasoning per candidate
- Add a web UI so recruiters can paste JDs and view results visually
- Support multi-language resumes
- Add experience and education-aware section chunking

---

## Author

**Pooja Shankhla**
MERN Full Stack Developer
[GitHub](https://github.com/Pooja9783)