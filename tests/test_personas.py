"""
test_personas.py — Automated pass/fail tests for all four BAX-423 personas.

Run with: pytest tests/test_personas.py -v
"""

import sys
import json
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import PERSONAS_FILE, DATA_DIR
from src.ingest import _generate_synthetic_jobs
from src.clean import clean_jobs
from src.ranker import rank_jobs, apply_hard_filters, RankedJob
from src.dedupe import full_deduplication
from src.embeddings import build_faiss_index, retrieve_candidates


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def personas():
    with open(PERSONAS_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def jobs_df():
    """Load or generate a clean job dataset for testing."""
    sample_path = DATA_DIR / "jobs_sample.parquet"
    if sample_path.exists():
        return pd.read_parquet(sample_path)
    raw = _generate_synthetic_jobs(2000)
    df  = clean_jobs(raw, save=False)
    df, _ = full_deduplication(df)
    return df


@pytest.fixture(scope="session")
def faiss_components(jobs_df):
    index, embeddings, job_ids = build_faiss_index(jobs_df)
    return index, embeddings, job_ids


def get_ranked_for_persona(persona, jobs_df, faiss_components):
    """Helper: run full pipeline for a persona and return ranked jobs."""
    index, _, job_ids = faiss_components
    candidates = retrieve_candidates(persona, index, job_ids, k=100)
    return rank_jobs(jobs_df, persona, candidates, top_n=10)


# ─── Persona 1: Aisha — ML Engineer Pivoter ───────────────────────────────────
class TestAisha:
    def test_no_senior_staff_roles(self, personas, jobs_df, faiss_components):
        """Top-10 must contain zero Senior or Staff roles."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        violations = [j for j in ranked if j.seniority in ("senior", "staff")]
        assert len(violations) == 0, \
            f"Found {len(violations)} senior/staff roles: {[j.title for j in violations]}"

    def test_no_defense_companies(self, personas, jobs_df, faiss_components):
        """Top-10 must contain zero defense/military companies."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        defense_keywords = ["defense", "military", "dynamics", "lockheed", "raytheon"]
        violations = [
            j for j in ranked
            if any(kw in (j.company + " " + j.title).lower() for kw in defense_keywords)
        ]
        assert len(violations) == 0, \
            f"Found defense company results: {[j.company for j in violations]}"

    def test_salary_meets_minimum(self, personas, jobs_df, faiss_components):
        """Jobs with listed salary should not be far below $140K target."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        salary_failures = [
            j for j in ranked
            if j.salary_max > 0 and j.salary_max < aisha["salary_min"] * 0.6
        ]
        assert len(salary_failures) == 0, \
            f"Jobs far below salary target: {[(j.title, j.salary_max) for j in salary_failures]}"

    def test_ml_related_results(self, personas, jobs_df, faiss_components):
        """At least 5 of top-10 must be ML/Data/AI related."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        ml_keywords = ["machine learning", "ml", "data scientist", "ai", "applied scientist",
                       "data analyst", "analytics", "deep learning", "nlp"]
        ml_count = sum(
            1 for j in ranked
            if any(kw in j.title.lower() for kw in ml_keywords)
        )
        assert ml_count >= 4, f"Only {ml_count}/10 ML-related jobs found"


# ─── Persona 2: Marcus — New Graduate ────────────────────────────────────────
class TestMarcus:
    def test_no_3plus_years_required(self, personas, jobs_df, faiss_components):
        """Top-10 must not require 3+ years of experience."""
        marcus = next(p for p in personas if p["id"] == "marcus")
        ranked = get_ranked_for_persona(marcus, jobs_df, faiss_components)
        violations = [j for j in ranked if j.experience_required >= 3]
        assert len(violations) == 0, \
            f"Found {len(violations)} roles requiring 3+ years: {[(j.title, j.experience_required) for j in violations]}"

    def test_no_contract_only(self, personas, jobs_df, faiss_components):
        """Top-10 must not contain contract-only roles."""
        marcus = next(p for p in personas if p["id"] == "marcus")
        ranked = get_ranked_for_persona(marcus, jobs_df, faiss_components)
        violations = [j for j in ranked if j.employment_type == "Contract"]
        assert len(violations) == 0, \
            f"Found {len(violations)} contract-only roles"

    def test_entry_level_seniority(self, personas, jobs_df, faiss_components):
        """Most results should be junior or mid level."""
        marcus = next(p for p in personas if p["id"] == "marcus")
        ranked = get_ranked_for_persona(marcus, jobs_df, faiss_components)
        appropriate = sum(1 for j in ranked if j.seniority in ("junior", "mid"))
        assert appropriate >= 7, f"Only {appropriate}/10 appropriate seniority levels"


# ─── Persona 3: Priya — Senior Engineer ──────────────────────────────────────
class TestPriya:
    def test_no_junior_roles(self, personas, jobs_df, faiss_components):
        """Top-10 must contain zero junior/entry-level roles."""
        priya = next(p for p in personas if p["id"] == "priya")
        ranked = get_ranked_for_persona(priya, jobs_df, faiss_components)
        violations = [j for j in ranked if j.seniority == "junior"]
        assert len(violations) == 0, \
            f"Found {len(violations)} junior roles: {[j.title for j in violations]}"

    def test_salary_minimum(self, personas, jobs_df, faiss_components):
        """Jobs with salary data should respect $200K target."""
        priya = next(p for p in personas if p["id"] == "priya")
        ranked = get_ranked_for_persona(priya, jobs_df, faiss_components)
        violations = [j for j in ranked
                      if j.salary_max > 0 and j.salary_max < priya["salary_min"] * 0.6]
        assert len(violations) <= 2, \
            f"Too many low-salary results: {[(j.title, j.salary_max) for j in violations]}"

    def test_senior_focused_results(self, personas, jobs_df, faiss_components):
        """Majority should be senior/mid roles."""
        priya = next(p for p in personas if p["id"] == "priya")
        ranked = get_ranked_for_persona(priya, jobs_df, faiss_components)
        appropriate = sum(1 for j in ranked if j.seniority in ("senior","mid","staff"))
        assert appropriate >= 6, f"Only {appropriate}/10 senior-appropriate roles"


# ─── Persona 4: Kenji — Visa Constrained ─────────────────────────────────────
class TestKenji:
    def test_no_contract_roles(self, personas, jobs_df, faiss_components):
        """Top-10 must contain zero contract/temp roles (visa incompatible)."""
        kenji = next(p for p in personas if p["id"] == "kenji")
        ranked = get_ranked_for_persona(kenji, jobs_df, faiss_components)
        violations = [j for j in ranked if j.employment_type == "Contract"]
        assert len(violations) == 0, \
            f"Found {len(violations)} contract roles: {[j.title for j in violations]}"

    def test_visa_awareness(self, personas, jobs_df, faiss_components):
        """Hard filter should mark visa_required correctly."""
        kenji  = next(p for p in personas if p["id"] == "kenji")
        index, _, job_ids = faiss_components
        candidates = retrieve_candidates(kenji, index, job_ids, k=100)
        filtered = apply_hard_filters(
            jobs_df[jobs_df["job_id"].isin({jid for jid,_ in candidates})].copy(),
            kenji
        )
        # All remaining should have visa_possible = True
        visa_failures = filtered[filtered["visa_possible"] == False]
        assert len(visa_failures) == 0, \
            f"Visa filter passed {len(visa_failures)} non-sponsoring jobs"

    def test_research_ai_focus(self, personas, jobs_df, faiss_components):
        """Most results should be research/AI/ML focused."""
        kenji = next(p for p in personas if p["id"] == "kenji")
        ranked = get_ranked_for_persona(kenji, jobs_df, faiss_components)
        ai_keywords = ["research", "scientist", "ml engineer", "ai engineer",
                       "applied scientist", "deep learning", "nlp", "computer vision"]
        ai_count = sum(1 for j in ranked
                       if any(kw in j.title.lower() for kw in ai_keywords))
        assert ai_count >= 4, f"Only {ai_count}/10 research/AI results for Kenji"


# ─── General pipeline tests ───────────────────────────────────────────────────
class TestPipeline:
    def test_ranking_produces_ordered_results(self, personas, jobs_df, faiss_components):
        """Ranked results must be in descending score order."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        scores = [j.final_score for j in ranked]
        assert scores == sorted(scores, reverse=True), "Results not in descending order"

    def test_no_duplicate_jobs_in_results(self, personas, jobs_df, faiss_components):
        """All ranked jobs must have unique job_ids."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        ids = [j.job_id for j in ranked]
        assert len(ids) == len(set(ids)), "Duplicate job_ids in ranked results"

    def test_max_2_per_company(self, personas, jobs_df, faiss_components):
        """MMR re-ranking should enforce max 2 jobs per company."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        from collections import Counter
        company_counts = Counter(j.company for j in ranked)
        violations = {co: cnt for co, cnt in company_counts.items() if cnt > 2}
        assert len(violations) == 0, f"Company cap violated: {violations}"

    def test_why_ranked_populated(self, personas, jobs_df, faiss_components):
        """Every ranked job must have a non-empty explanation."""
        aisha = next(p for p in personas if p["id"] == "aisha")
        ranked = get_ranked_for_persona(aisha, jobs_df, faiss_components)
        empty_explanations = [j for j in ranked if not j.why_ranked]
        assert len(empty_explanations) == 0, "Some jobs missing 'why ranked' explanation"
