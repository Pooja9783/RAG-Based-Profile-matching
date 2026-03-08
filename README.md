# RAG-Based Resume Profile Matching

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) based resume matching system** that automatically ranks candidate resumes against a given job description.

The system processes PDF resumes, converts them into vector embeddings, and performs **semantic similarity search** combined with **skill-based scoring** to identify the most relevant candidates.

This approach mimics the behavior of modern **AI-powered Applicant Tracking Systems (ATS)** used in recruitment platforms.

---

## Features

* Resume ingestion from PDF files
* Automatic text chunking
* Semantic embeddings using **MiniLM**
* Vector-based similarity search
* Hybrid scoring (semantic similarity + skill match)
* Candidate ranking on a **0–100 scale**
* Relevant resume excerpts for reasoning
* Support for must-have skill filtering

---

## Project Architecture

```
Job Description
      │
      ▼
Embedding Model (MiniLM)
      │
      ▼
Vector Similarity Search
      │
      ▼
Hybrid Scoring
(Semantic + Skill Match)
      │
      ▼
Ranked Candidate List
```

---

## Tech Stack

* **Node.js**
* **@xenova/transformers** – local embedding model
* **pdf-parse** – resume text extraction
* **JavaScript** – similarity scoring and ranking

Embedding Model Used:

```
Xenova/all-MiniLM-L6-v2
```

---

## Project Structure

```
RAG-Based-Profile-matching
│
├── resumes/
│   ├── Pooja-Shankhla-Resume.pdf
│   ├── abc.pdf
│
├── resume_rag.js
├── job_matcher.js
├── vectorDB.json
├── package.json
└── README.md
```

---

## How the System Works

### 1. Resume Ingestion

PDF resumes are parsed using **pdf-parse** to extract text content.

### 2. Text Chunking

Resume text is split into smaller chunks to improve retrieval accuracy.

### 3. Embedding Generation

Each chunk is converted into vector embeddings using the **MiniLM sentence transformer model**.

### 4. Vector Database

Embeddings are stored locally in a JSON-based vector database (`vectorDB.json`).

### 5. Query Processing

The job description is also converted into an embedding.

### 6. Semantic Similarity Search

Cosine similarity is used to find the most relevant resume chunks.

### 7. Hybrid Scoring

Candidate ranking combines:

```
Final Score =
0.7 × Semantic Similarity
+ 0.3 × Skill Match Score
```

### 8. Candidate Ranking

Candidates are ranked and returned with:

* Match score (0–100)
* Matched skills
* Relevant resume excerpts
* Match reasoning

---

## Example Output

```
{
  "job_description": "Looking for a React Node developer with 3 years experience",
  "top_matches": [
    {
      "candidate_name": "Pooja-Shankhla-Resume",
      "resume_path": "resumes/Pooja-Shankhla-Resume.pdf",
      "match_score": 72,
      "semantic_score": "0.60",
      "skill_match_score": "1.00",
      "matched_skills": [
        "React",
        "Node.js",
        "MongoDB"
      ],
      "reasoning": "Semantic similarity between job description and resume"
    }
  ]
}
```

---

## Setup Instructions

### 1. Clone the Repository

```
git clone <repository-url>
cd RAG-Based-Profile-matching
```

### 2. Install Dependencies

```
npm install
```

### 3. Add Resumes

Place resume PDFs inside the `resumes/` folder.

---

## Running the Project

### Step 1 – Generate Resume Embeddings

```
node resume_rag.js
```

This will:

* Parse resumes
* Generate embeddings
* Create `vectorDB.json`

---

### Step 2 – Run Resume Matching

```
node job_matcher.js
```

This will:

* Embed the job description
* Perform similarity search
* Rank candidates

---

## Assignment Requirements Covered

* Resume ingestion
* RAG-based retrieval system
* Candidate ranking
* Match scoring (0–100)
* Match reasoning
* Relevant resume excerpts
* Must-have skill filtering

---

## Future Improvements

Possible enhancements for production systems:

* Experience extraction from resumes
* LLM-based reasoning generation
* Real vector database (ChromaDB / Pinecone)
* Web interface for recruiters
* Support for multiple job queries

---

## Author

**Pooja Shankhla**

MERN Full Stack Developer
Experience in building scalable web applications using React, Node.js, and MongoDB.
