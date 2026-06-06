"""
resume_generator.py — AI-powered tailored resume generation using OpenAI GPT-4o-mini.

Takes a user profile + selected job description and generates an ATS-optimized,
tailored resume that emphasizes relevant skills without fabricating credentials.
"""

import os
import json
import logging
from openai import OpenAI

from src.utils import OPENAI_API_KEY, OPENAI_MODEL, logger
from src.ranker import RankedJob

# ─── Prompts (also exported to prompts.md) ────────────────────────────────────

RESUME_SYSTEM_PROMPT = """You are an expert resume editor and career coach specializing in tech and data roles.
Your task is to rewrite a candidate's resume to be tailored for a specific job posting.

STRICT RULES — these are non-negotiable:
1. Do NOT invent, fabricate, or add any experience, employers, degrees, certifications, or tools not present in the candidate profile.
2. Do NOT change dates, company names, or job titles from the original.
3. ONLY reorder, rephrase, and emphasize content already in the profile.
4. Use keywords from the job description naturally throughout.
5. Be specific and quantitative where the original data supports it.
6. Keep to one page for <5 years experience, two pages for 5+ years.
7. Produce output in clean Markdown.

WARNING: This tool tailors wording and prioritization only. It does not invent credentials."""

RESUME_USER_TEMPLATE = """
## CANDIDATE PROFILE
{profile_json}

## TARGET JOB
**Title:** {job_title}
**Company:** {company}
**Location:** {location}

**Job Description (first 1500 chars):**
{job_description}

**Required Skills from Posting:** {required_skills}
**Skills You Have (matched):** {matched_skills}
**Skills You're Missing:** {missing_skills}

## YOUR TASK
Generate a complete, ATS-optimized resume tailored to this specific role.

Structure the resume exactly as:
# [Candidate Name]
[email] | [location] | [LinkedIn if available]

## Professional Summary
(3 sentences max — tailored to THIS job, using job keywords naturally)

## Skills
(Prioritize skills that appear in the job description first)

## Experience
(Rewrite bullets to use job description language; quantify where possible)

## Education

## Projects (if any)

---
After the resume, include a brief section titled:
## What Was Changed & Why
(2-4 bullet points explaining your key editorial decisions)
"""

SKILL_EXTRACTION_PROMPT = """Extract the required technical skills from this job description.
Return a JSON array of strings. Include only specific technical skills, tools, and technologies.
Limit to 15 items. Do not include soft skills.

Job Description:
{description}

Return only valid JSON like: ["Python", "SQL", "AWS"]"""

EXPLANATION_PROMPT = """You are a career advisor. Explain in 2-3 friendly, specific sentences why this job
is a good match for this candidate. Be honest about both strengths and gaps.

Candidate skills: {candidate_skills}
Job title: {job_title}
Matched skills: {matched_skills}
Missing skills: {missing_skills}
Semantic similarity: {similarity_pct}%

Write a 2-3 sentence explanation starting with "This role matches your profile because..." """


# ─── OpenAI client ────────────────────────────────────────────────────────────
def _get_client() -> OpenAI | None:
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — LLM features disabled")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_llm(
    system: str,
    user: str,
    model: str = OPENAI_MODEL,
    max_tokens: int = 1800,
    temperature: float = 0.3,
) -> str | None:
    """Call OpenAI API with error handling and rate limit awareness."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


# ─── Resume generation ────────────────────────────────────────────────────────
def generate_resume(profile: dict, job: "RankedJob") -> dict:
    """
    Generate a tailored resume for a selected job.

    Returns:
        dict with keys:
          - 'markdown': the full resume as markdown string
          - 'method':   'ai' or 'template'
          - 'matched_skills': list of matched skills
          - 'missing_skills': list of missing skills
          - 'warning': fabrication warning string
    """
    matched = job.matched_skills or []
    missing = job.missing_skills or []

    user_prompt = RESUME_USER_TEMPLATE.format(
        profile_json     = json.dumps({
            k: v for k, v in profile.items()
            if k not in ["id", "pass_criteria"]
        }, indent=2),
        job_title        = job.title,
        company          = job.company,
        location         = job.location,
        job_description  = job.description[:1500],
        required_skills  = ", ".join(job.skills_extracted[:12]),
        matched_skills   = ", ".join(matched[:8]),
        missing_skills   = ", ".join(missing[:5]),
    )

    markdown = _call_llm(RESUME_SYSTEM_PROMPT, user_prompt, max_tokens=2000)

    if markdown is None:
        markdown = _template_resume(profile, job)
        method = "template"
    else:
        method = "ai"

    return {
        "markdown":      markdown,
        "method":        method,
        "matched_skills": matched,
        "missing_skills": missing,
        "warning": (
            "⚠️ JobPilot tailors wording and prioritization only. "
            "It does not invent degrees, certifications, employers, or experience. "
            "Always review before submitting."
        ),
    }


def generate_match_explanation(profile: dict, job: "RankedJob") -> str:
    """Generate a friendly 2-3 sentence explanation of why this job matches."""
    prompt = EXPLANATION_PROMPT.format(
        candidate_skills = ", ".join(profile.get("skills", [])[:10]),
        job_title        = job.title,
        matched_skills   = ", ".join(job.matched_skills[:5]),
        missing_skills   = ", ".join(job.missing_skills[:3]),
        similarity_pct   = int(job.embedding_score * 100),
    )

    result = _call_llm(
        "You are a helpful career advisor. Be concise and specific.",
        prompt,
        max_tokens=200,
        temperature=0.5,
    )

    if result:
        return result.strip()

    # Fallback
    return (
        f"This role matches your profile because you share {len(job.matched_skills)} key skills "
        f"including {', '.join(job.matched_skills[:3])}. "
        f"The {int(job.embedding_score * 100)}% semantic similarity score indicates strong overall alignment. "
        + (f"You'll want to build skills in {', '.join(job.missing_skills[:2])} to be fully competitive."
           if job.missing_skills else "Your skill set covers most requirements well.")
    )


def extract_skills_with_llm(job_description: str) -> list[str]:
    """Use LLM to extract required skills from a job description."""
    prompt = SKILL_EXTRACTION_PROMPT.format(description=job_description[:1200])
    result = _call_llm(
        "You are a skill extractor. Return only valid JSON arrays.",
        prompt,
        max_tokens=200,
        temperature=0.1,
    )
    if result:
        try:
            import re
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
    return []


# ─── Template fallback (when no API key) ─────────────────────────────────────
def _template_resume(profile: dict, job: "RankedJob") -> str:
    """Generate a structured resume template without LLM (fallback)."""
    name     = profile.get("name", "Candidate")
    title    = profile.get("current_title", "Professional")
    skills   = profile.get("skills", [])
    resume   = profile.get("resume_text", "")
    edu      = profile.get("education", "")
    exp      = profile.get("years_experience", 0)
    target   = job.title
    company  = job.company

    # Prioritize matched skills first
    matched  = job.matched_skills or []
    all_skills = matched + [s for s in skills if s not in matched]

    return f"""# {name}
{profile.get('location_preference', 'United States')}

---

## Professional Summary
Results-driven {title} with {exp}+ years of experience seeking a {target} role at {company}.
Bringing expertise in {', '.join(all_skills[:4])} with a track record of delivering data-driven insights.
Eager to apply technical skills to {job.description[:100].split('.')[0].lower()}.

---

## Skills

**Core Technical:** {' • '.join(all_skills[:10])}

**Additional:** {' • '.join(all_skills[10:16])}

---

## Experience
*(Based on your profile — tailor bullet points to match {company}'s job description)*

**{title}**
Previous Employer | [Duration]
- Applied {', '.join(matched[:3])} to solve business problems
- Delivered data analysis projects with measurable impact
- Collaborated cross-functionally to drive analytical initiatives

---

## Education
{edu}

---

## Notes
*This resume was generated from your profile using JobPilot's template engine (no AI). *
*For best results, add your OpenAI API key to enable AI-powered resume tailoring.*
*Missing skills to consider highlighting: {', '.join(job.missing_skills[:4])}*
"""
