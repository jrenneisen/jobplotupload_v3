# prompts.md — AI Prompts Used in JobPilot

All significant prompts used during the development of JobPilot, with explanations
of their purpose and key modifications made to initial outputs.

---

## Prompt 1: Resume Tailoring (Core Feature)

**Location:** `src/resume_generator.py` — `RESUME_SYSTEM_PROMPT` + `RESUME_USER_TEMPLATE`

**Purpose:** Generate an ATS-optimized, tailored resume for a specific job posting using the
candidate's profile. This is the signature deliverable of JobPilot.

```
SYSTEM:
You are an expert resume editor and career coach specializing in tech and data roles.
Your task is to rewrite a candidate's resume to be tailored for a specific job posting.

STRICT RULES:
1. Do NOT invent, fabricate, or add any experience, employers, degrees, certifications,
   or tools not present in the candidate profile.
2. Do NOT change dates, company names, or job titles from the original.
3. ONLY reorder, rephrase, and emphasize content already in the profile.
4. Use keywords from the job description naturally throughout.
5. Be specific and quantitative where the original data supports it.
6. Keep to one page for <5 years experience, two pages for 5+ years.
7. Produce output in clean Markdown.

WARNING: This tool tailors wording and prioritization only. It does not invent credentials.

USER: [Candidate profile JSON + target job details + matched/missing skills]
```

**Key modification:** Added the explicit anti-hallucination rule (Rule 1) after initial testing
showed the model adding a "TensorFlow certification" that wasn't in the profile. The warning
line was added to the UI as a persistent disclaimer.

---

## Prompt 2: Job Match Explanation

**Location:** `src/resume_generator.py` — `EXPLANATION_PROMPT`

**Purpose:** Generate a friendly, specific 2-3 sentence explanation of why a job matches
a candidate's profile, shown in the "Why Ranked Here?" section of each job card.

```
You are a career advisor. Explain in 2-3 friendly, specific sentences why this job
is a good match for this candidate. Be honest about both strengths and gaps.

Candidate skills: {candidate_skills}
Job title: {job_title}
Matched skills: {matched_skills}
Missing skills: {missing_skills}
Semantic similarity: {similarity_pct}%

Write a 2-3 sentence explanation starting with "This role matches your profile because..."
```

**Key modification:** Added "Be honest about both strengths and gaps" after initial outputs
were overly positive and didn't mention missing skills. Also capped at 200 tokens to keep
responses concise.

---

## Prompt 3: Skill Extraction from Job Descriptions

**Location:** `src/resume_generator.py` — `SKILL_EXTRACTION_PROMPT`

**Purpose:** Extract structured skill lists from unstructured job description text,
supplementing the regex-based extraction in `src/utils.py`.

```
Extract the required technical skills from this job description.
Return a JSON array of strings. Include only specific technical skills,
tools, and technologies. Limit to 15 items. Do not include soft skills.

Job Description: {description}

Return only valid JSON like: ["Python", "SQL", "AWS"]
```

**Key modification:** Added "Return only valid JSON" after the model was initially
returning markdown code blocks that broke the JSON parser. Also added "Do not include
soft skills" to avoid "communication" and "teamwork" being returned.

---

## Prompt 4: Build Prompt Generation (Development)

**Purpose:** Used Claude to generate the comprehensive AI build prompt (this project's
design document) from the BAX-423 project specification.

```
You are building JobPilot, a full-stack intelligent job-matching web application.
[Full specification including all 6 capabilities, tech stack, personas, rubric]

Generate a complete, detailed AI build prompt covering:
- All 6 required capabilities with implementation code
- Two BAX-423 techniques (MinHash LSH and Dense Embeddings + FAISS)
- Thompson Sampling as the third technique
- Full directory structure
- Deployment to GCP Cloud Run
- All 4 test personas with pass criteria
```

**Key modification:** Iterated 3 times — first output lacked the MMR re-ranking stage,
second lacked the JSearch API integration, third pass added the benchmark comparison tables.

---

## Prompt 5: Streamlit UI Architecture

**Purpose:** Used Claude to design the multi-page Streamlit application structure
with session state management and the pipeline execution flow.

```
Design a Streamlit application for JobPilot with:
- Sidebar navigation between 5 pages
- Session state for: profile, jobs_df, FAISS index, ranked jobs, feedback, adaptive learner
- A pipeline execution function that runs all 5 steps with a progress bar
- Job card components with score badges, skill pills, and feedback buttons
- Full custom CSS for a professional blue-themed design
```

**Key modification:** Added the `_run_full_pipeline()` function as a separate helper
to prevent re-running the pipeline on every Streamlit rerun. Added `st.rerun()` after
pipeline completion to navigate to the Job Matches page automatically.

---

## Prompt 6: Thompson Sampling Implementation

**Purpose:** Used Claude to implement the Thompson Sampling bandit for adaptive learning,
ensuring it satisfies the BAX-423 Reinforcement Learning requirement.

```
Implement a Thompson Sampling multi-armed bandit for a job recommendation system.
Arms should be defined by (seniority × industry × location_type) combinations.
Beta distributions should be updated with:
- reward = 1.0 for "good fit" / "save"
- reward = 0.4 for "skip"
- reward = 0.0 for "not for me"
The bandit score should blend into the final ranking score with weight
that increases as more feedback is collected (exploration → exploitation).
```

**Key modification:** Changed arm definition from (company × title) to
(seniority × industry × location_type) after realizing company-level arms would
never accumulate enough data to learn. The industry inference function was added
separately to map job titles to 5 broad categories.

---

## Prompt 7: Debugging — FAISS Index Build

**Purpose:** Resolved a shape mismatch error when building the FAISS index with
the `all-MiniLM-L6-v2` model on the Kaggle dataset.

```
I'm getting a faiss.swigfaiss.InternalError when adding vectors to a FAISS
IndexFlatIP index. The embeddings array has shape (22847, 384) and I'm using
faiss.IndexFlatIP(384). The error is: "not normalized vectors". Fix this.
```

**Key modification:** The fix was to add `normalize_embeddings=True` to the
`SentenceTransformer.encode()` call, which L2-normalizes vectors so that inner
product equals cosine similarity. Added a unit test to verify normalization.

---

## Prompt 8: MinHash LSH Parameter Tuning

**Purpose:** Determined optimal threshold and num_perm parameters for the
MinHash LSH deduplication step.

```
I'm using datasketch MinHashLSH to deduplicate job postings. What threshold
and num_perm values should I use to:
1. Detect paraphrased reposts from the same company (Jaccard ~0.80-0.90)
2. Not accidentally merge genuinely different jobs (Jaccard ~0.40-0.60)
3. Process 20,000 records in under 30 seconds on CPU

Provide a benchmark comparison between threshold=0.80, 0.85, and 0.90
with expected false positive and false negative rates.
```

**Key modification:** Selected threshold=0.85, num_perm=128 as the best tradeoff.
Initial recommendation was threshold=0.80 which had too many false positives
(merging different ML Engineer roles at the same company). Switched to company-prefixed
LSH keys to only deduplicate within the same company.
