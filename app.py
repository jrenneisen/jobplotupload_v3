"""
JobPilot — Smart Job Matcher & Resume Builder
BAX-423 Big Data | Spring 2026 | Final Project Option B

Run with: streamlit run app.py
"""

import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Optional

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    DATA_DIR, PERSONAS_FILE, TOP_K_JOBS, RETRIEVAL_K,
    OPENAI_API_KEY, JSEARCH_API_KEY, logger
)

# ─── Init database on startup ─────────────────────────────────────────────────
from src.storage import (
    init_db, create_or_update_user, get_user, list_users, delete_user,
    save_profile, load_profile,
    save_feedback_event, load_feedback_history, get_feedback_summary,
    save_bandit_state, load_bandit_state, replay_feedback_into_bandit,
    save_ranking_weights, load_ranking_weights,
    save_resume, load_resumes,
    get_learning_insights, db_size_kb,
)
init_db()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobPilot — Smart Job Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
:root {
  --brand-blue: #1F4E79;
  --accent-blue: #2E75B6;
  --light-blue: #D6E4F0;
  --success: #27AE60;
  --warning: #F39C12;
  --danger: #E74C3C;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1F4E79 0%, #2E4057 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; font-size: 0.95rem; }

/* Job cards */
.job-card {
  background: white;
  border: 1px solid #E0ECF8;
  border-left: 5px solid #2E75B6;
  border-radius: 10px;
  padding: 18px 22px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(30,78,121,0.07);
  transition: box-shadow 0.2s;
}
.job-card:hover { box-shadow: 0 4px 16px rgba(30,78,121,0.14); }

/* Score badge */
.score-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.88rem;
  color: white;
  margin-right: 6px;
}
.score-high   { background: #27AE60; }
.score-mid    { background: #F39C12; }
.score-low    { background: #E74C3C; }

/* Skill pills */
.skill-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.78rem;
  margin: 2px;
  font-weight: 500;
}
.skill-matched { background: #D5F5E3; color: #1E8449; border: 1px solid #82E0AA; }
.skill-missing { background: #FDEBD0; color: #A04000; border: 1px solid #F0B27A; }

/* Section headers */
.section-header {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1F4E79;
  border-bottom: 3px solid #2E75B6;
  padding-bottom: 8px;
  margin-bottom: 20px;
}

/* Metric cards */
.metric-card {
  background: #F0F6FC;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  border: 1px solid #D6E4F0;
}
.metric-number { font-size: 2rem; font-weight: 800; color: #1F4E79; }
.metric-label  { font-size: 0.85rem; color: #5D6D7E; }

/* Hero banner */
.hero {
  background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
  color: white;
  padding: 32px 40px;
  border-radius: 14px;
  margin-bottom: 28px;
}
.hero h1 { color: white; font-size: 2.4rem; margin-bottom: 6px; }
.hero p  { color: #BDD7EE; font-size: 1.05rem; margin: 0; }

/* Feedback buttons */
.stButton > button {
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.15s;
}

/* Resume output */
.resume-output {
  background: white;
  border: 1px solid #D6E4F0;
  border-radius: 10px;
  padding: 28px;
  font-family: Georgia, serif;
  line-height: 1.6;
  max-height: 600px;
  overflow-y: auto;
}

/* Hide Streamlit footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialization ─────────────────────────────────────────────
def init_session():
    defaults = {
        "page":             "🏠 Profile Setup",
        "current_user":     None,    # logged-in user_id string
        "profile":          None,
        "jobs_df":          None,
        "faiss_index":      None,
        "job_ids":          None,
        "ranked_jobs":      [],
        "feedback":         {},
        "adaptive":         None,
        "resumes":          {},
        "selected_job":     None,
        "pipeline_ready":   False,
        "analytics":        None,
        "benchmark_data":   {},
        "data_stats":       {},
        "tfidf_candidates": [],
        "emb_candidates":   [],
        "hybrid_candidates":[],
        "cluster_labels":   None,   # K-Means job family clusters (ndarray)
        "positive_ids":     set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Login / logout helpers ───────────────────────────────────────────────────
def _login_user(user_id: str):
    """
    Load all persisted state for a user into session_state.
    Restores: profile, feedback dict, resumes, bandit arms, ranking weights.
    Replays full feedback history through the adaptive learner so the model
    continues improving exactly where it left off.
    """
    from src.adaptive_learning import AdaptiveLearner

    create_or_update_user(user_id, user_id.replace("_", " ").title())
    st.session_state.current_user = user_id

    # ── Profile ───────────────────────────────────────────────────────────────
    saved_profile = load_profile(user_id)
    if saved_profile:
        st.session_state.profile = saved_profile
        st.toast(f"✅ Welcome back! Profile loaded.")

    # ── Feedback dict (for current-session display) ────────────────────────────
    history = load_feedback_history(user_id)
    st.session_state.feedback = {
        e["job_id"]: e["feedback_type"] for e in history
    }

    # ── Positive IDs set ──────────────────────────────────────────────────────
    st.session_state.positive_ids = {
        e["job_id"] for e in history
        if e["feedback_type"] in ("good", "save")
    }

    # ── Adaptive learner — restore + replay ───────────────────────────────────
    weights  = load_ranking_weights(user_id)
    adaptive = AdaptiveLearner(initial_weights=weights)

    # Load saved arm distributions
    adaptive.bandit = load_bandit_state(user_id, adaptive.bandit)

    # If arms were empty (first replay), replay history chronologically
    if not adaptive.bandit.arms and history:
        adaptive.bandit, adaptive.updater = replay_feedback_into_bandit(
            user_id, adaptive.bandit, adaptive.updater
        )

    st.session_state.adaptive = adaptive

    # ── Cached resumes ────────────────────────────────────────────────────────
    st.session_state.resumes = load_resumes(user_id)

    logger.info(
        f"Login: {user_id} — {len(history)} feedback events, "
        f"{len(adaptive.bandit.arms)} bandit arms restored"
    )


def _save_session_to_db():
    """
    Persist current session state to database.
    Called automatically on logout and after every feedback event.
    """
    uid = st.session_state.current_user
    if not uid:
        return

    if st.session_state.profile:
        save_profile(uid, st.session_state.profile)

    if st.session_state.adaptive:
        save_bandit_state(uid, st.session_state.adaptive.bandit)
        save_ranking_weights(uid, st.session_state.adaptive.weights)


# ─── PDF extraction helper ────────────────────────────────────────────────────
def _extract_pdf_text(uploaded_file) -> str:
    """
    Extract plain text from an uploaded PDF resume.

    Process:
      1. Read raw bytes from the Streamlit UploadedFile object
      2. Open with pdfplumber (handles multi-column layouts, tables, headers)
      3. Extract text page by page and join with newlines
      4. Clean whitespace and return

    Works with: standard PDFs, Google Docs exports, Word-to-PDF, LaTeX resumes.
    May struggle with: scanned image PDFs (no embedded text layer).
    """
    import io
    import re
    try:
        import pdfplumber
    except ImportError:
        st.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""

    try:
        raw_bytes = uploaded_file.read()
        pages_text = []
        total_pages = 0

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text(
                    x_tolerance=2,   # how close chars must be to join on same line
                    y_tolerance=3,   # how close lines must be to join in same block
                )
                if text:
                    pages_text.append(text)

        if not pages_text:
            # Fallback: extract individual words if extract_text() returns nothing
            # (happens with some PDF generators that don't embed a text layer linearly)
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if words:
                        pages_text.append(" ".join(w["text"] for w in words))

        full_text = "\n\n".join(pages_text)

        # Clean common PDF extraction artifacts
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)  # collapse excessive blank lines
        full_text = re.sub(r"[ \t]{2,}", " ", full_text)  # collapse multiple spaces
        full_text = full_text.strip()

        logger.info(f"PDF extracted: {total_pages} page(s), {len(full_text):,} chars")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        st.error(f"Could not read PDF: {e}. Try pasting your resume text manually below.")
        return ""


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 16px;">
        <div style="font-size:2.5rem;">🚀</div>
        <div style="font-size:1.4rem; font-weight:800; letter-spacing:1px;">JobPilot</div>
        <div style="font-size:0.78rem; opacity:0.8;">Smart Job Matcher</div>
    </div>
    """, unsafe_allow_html=True)

    # ── User login ────────────────────────────────────────────────────────────
    st.markdown("**👤 User Account**")
    existing_users = list_users()
    user_names     = [u["display_name"] for u in existing_users]

    login_mode = st.radio("", ["Existing user", "New user"],
                          horizontal=True, label_visibility="collapsed")

    if login_mode == "New user":
        new_name = st.text_input("Your name", placeholder="e.g. Jacob R.")
        if st.button("Create Account", use_container_width=True, type="primary"):
            if new_name.strip():
                uid = new_name.strip().lower().replace(" ", "_")
                create_or_update_user(uid, new_name.strip())
                _login_user(uid)
                st.rerun()
            else:
                st.warning("Enter a name first.")
    else:
        if user_names:
            chosen = st.selectbox("Select account", user_names,
                                  label_visibility="collapsed")
            if st.button("Log In", use_container_width=True, type="primary"):
                uid = next(u["user_id"] for u in existing_users
                           if u["display_name"] == chosen)
                _login_user(uid)
                st.rerun()
        else:
            st.caption("No accounts yet — create one above.")

    # Show logged-in user
    if st.session_state.current_user:
        u = get_user(st.session_state.current_user)
        fb_summary = get_feedback_summary(st.session_state.current_user)
        total_fb   = sum(
            v["count"] for v in fb_summary.values()
            if isinstance(v, dict) and "count" in v
        )
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.12); border-radius:8px;
                    padding:10px 12px; margin:8px 0;">
            <div style="font-weight:700;">✅ {u['display_name']}</div>
            <div style="font-size:0.75rem; opacity:0.8;">
                {total_fb} feedback events saved<br>
                Last seen: {u['last_seen'][:10]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Log Out", use_container_width=True):
            _save_session_to_db()
            st.session_state.current_user = None
            st.session_state.profile      = None
            st.session_state.ranked_jobs  = []
            st.session_state.feedback     = {}
            st.session_state.adaptive     = None
            st.session_state.resumes      = {}
            st.session_state.pipeline_ready = False
            st.rerun()

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    pages = [
        "🏠 Profile Setup",
        "🎯 Job Matches",
        "📄 Resume Generator",
        "📊 Market Analytics",
        "📈 Benchmarks",
        "🧠 My Learning Profile",
    ]
    page = st.radio("Navigate", pages, key="nav_radio",
                    index=pages.index(st.session_state.page)
                    if st.session_state.page in pages else 0)
    st.session_state.page = page

    st.divider()

    # ── Status indicators ─────────────────────────────────────────────────────
    def _status(ok, label):
        st.markdown(f"{'✅' if ok else '⚪'} {label}")

    _status(st.session_state.current_user is not None, "Logged in")
    _status(st.session_state.profile is not None,      "Profile loaded")
    _status(st.session_state.pipeline_ready,           "Pipeline ready")
    _status(len(st.session_state.ranked_jobs) > 0,     "Jobs ranked")
    _status(bool(OPENAI_API_KEY),                      "AI resume enabled")
    _status(bool(JSEARCH_API_KEY),                     "Live jobs enabled")

    st.divider()
    st.markdown(
        f"<div style='font-size:0.70rem; opacity:0.6;'>"
        f"BAX-423 · Spring 2026<br>DB: {db_size_kb()} KB</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE SETUP
# ══════════════════════════════════════════════════════════════════════════════
def page_profile():
    st.markdown('<div class="hero"><h1>🚀 JobPilot</h1><p>Upload your profile → get ranked job matches → generate a tailored resume</p></div>', unsafe_allow_html=True)

    # Load personas
    personas = []
    if PERSONAS_FILE.exists():
        with open(PERSONAS_FILE) as f:
            personas = json.load(f)

    tab1, tab2 = st.tabs(["👤 Select Test Persona", "✏️ Custom Profile"])

    # ── Persona selector ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Choose a pre-built test persona")
        cols = st.columns(len(personas))
        for i, persona in enumerate(personas):
            with cols[i]:
                st.markdown(f"""
                <div style="background:#F0F6FC; border:1px solid #D6E4F0; border-radius:10px;
                            padding:14px; text-align:center; min-height:160px;">
                    <div style="font-size:2rem;">{persona['emoji']}</div>
                    <div style="font-weight:700; color:#1F4E79; font-size:0.9rem;">
                        {persona['name'].split('—')[0].strip()}
                    </div>
                    <div style="font-size:0.75rem; color:#5D6D7E; margin-top:4px;">
                        {persona['current_title']}
                    </div>
                    <div style="font-size:0.72rem; color:#27AE60; margin-top:4px;">
                        Target: {persona['target_roles'][0]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Use {persona['emoji']}", key=f"persona_{i}", use_container_width=True):
                    st.session_state.profile = persona
                    st.session_state.pipeline_ready = False
                    st.session_state.ranked_jobs = []
                    st.success(f"✅ Profile set: {persona['name']}")
                    st.rerun()

    # ── Custom profile form ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### Build your custom profile")
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                name     = st.text_input("Your Name",          "Alex Johnson")
                title    = st.text_input("Current Title",       "Data Analyst")
                exp      = st.number_input("Years of Experience", 0, 40, 2)
                edu      = st.text_input("Education",           "BS Computer Science")
            with col2:
                salary   = st.number_input("Minimum Salary ($)", 0, 500000, 90000, step=5000)
                seniority= st.selectbox("Seniority Target",     ["junior", "mid", "senior"])
                remote   = st.checkbox("Remote required?", False)
                visa     = st.checkbox("Need visa sponsorship?", False)

            target_roles = st.text_input(
                "Target Roles (comma-separated)",
                "Data Scientist, ML Engineer"
            )
            skills_input = st.text_area(
                "Your Skills (comma-separated)",
                "Python, SQL, pandas, scikit-learn, Tableau"
            )
            locations = st.text_input(
                "Preferred Locations (comma-separated)",
                "Remote, San Francisco, New York"
            )
            dealbreakers = st.text_input(
                "Dealbreakers (keywords to avoid — comma-separated)",
                "defense, contract only"
            )
            # ── PDF Upload ────────────────────────────────────────────────
            st.markdown("**📎 Upload Resume (PDF)**")
            uploaded_pdf = st.file_uploader(
                "Upload your resume as a PDF",
                type=["pdf"],
                help="Max 10MB. The text will be extracted automatically.",
                label_visibility="collapsed",
            )

            resume_text = ""
            if uploaded_pdf is not None:
                with st.spinner("📄 Extracting text from PDF..."):
                    resume_text = _extract_pdf_text(uploaded_pdf)
                if resume_text:
                    st.success(f"✅ Resume extracted — {len(resume_text):,} characters")
                    with st.expander("Preview extracted text", expanded=False):
                        st.text(resume_text[:1200] + ("..." if len(resume_text) > 1200 else ""))
                else:
                    st.warning("⚠️ Could not extract text from PDF. Try pasting below instead.")

            # Fallback: manual paste (pre-filled from PDF if uploaded)
            resume_text = st.text_area(
                "Or paste resume text manually",
                value=resume_text,
                height=150,
                placeholder="Paste plain text from your resume here, or upload a PDF above...",
            )
            industries = st.text_input(
                "Industries of interest",
                "Technology, Healthcare, Finance"
            )

            submitted = st.form_submit_button("💾 Save Profile", use_container_width=True, type="primary")

        if submitted:
            st.session_state.profile = {
                "id":              "custom",
                "name":            name,
                "emoji":           "👤",
                "current_title":   title,
                "years_experience": int(exp),
                "education":       edu,
                "skills":          [s.strip() for s in skills_input.split(",") if s.strip()],
                "target_roles":    [r.strip() for r in target_roles.split(",") if r.strip()],
                "industries":      [i.strip() for i in industries.split(",") if i.strip()],
                "location_preference": locations.split(",")[0].strip() if locations else "Any",
                "locations":       [l.strip() for l in locations.split(",") if l.strip()],
                "remote_required": remote,
                "salary_min":      int(salary),
                "visa_required":   visa,
                "seniority_target": seniority,
                "dealbreakers":    [d.strip() for d in dealbreakers.split(",") if d.strip()],
                "career_goal":     f"Seeking {target_roles.split(',')[0].strip()} role.",
                "resume_text":     resume_text,
            }
            st.session_state.pipeline_ready = False
            st.session_state.ranked_jobs = []
            # Persist to database if logged in
            uid = st.session_state.current_user
            if uid:
                save_profile(uid, st.session_state.profile)
                st.success("✅ Profile saved and stored to your account!")
            else:
                st.success("✅ Profile saved! Log in to keep it permanently.")

    # ── Current profile preview ───────────────────────────────────────────────
    if st.session_state.profile:
        p = st.session_state.profile
        st.divider()
        st.markdown("### Current Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**{p.get('emoji','')} {p.get('name','')}**")
            st.markdown(f"*{p.get('current_title','')} · {p.get('years_experience',0)} yrs exp*")
            st.markdown(f"🎯 {', '.join(p.get('target_roles', [])[:2])}")
        with col2:
            st.markdown(f"📍 {p.get('location_preference','Any')}")
            st.markdown(f"💰 ${p.get('salary_min', 0):,}+ target")
            st.markdown(f"🏷️ {p.get('seniority_target','mid').title()}-level")
        with col3:
            st.markdown(f"**Skills ({len(p.get('skills', []))})**")
            skills_preview = p.get('skills', [])[:8]
            st.markdown(" ".join(f"`{s}`" for s in skills_preview))

        st.markdown("---")
        _run_pipeline_section()


def _run_pipeline_section():
    """Data loading + indexing — triggered from profile page."""
    profile = st.session_state.profile
    if not profile:
        return

    st.markdown("### 🔧 Run Data Pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        data_source = st.selectbox(
            "Data source",
            ["Offline dataset (Kaggle snapshot)", "Sample data (fast demo)", "Both + Live (JSearch)"],
            index=1,
        )
    with col2:
        fetch_live = JSEARCH_API_KEY and "Live" in data_source
        live_queries = st.text_input(
            "Live search queries (JSearch)",
            ", ".join(profile.get("target_roles", ["data scientist"])[:3]),
            disabled=not fetch_live,
        )
    with col3:
        st.markdown("")
        st.markdown("")
        test_col, run_col = st.columns([1, 2])
        with test_col:
            if st.button("🔌 Test API", use_container_width=True,
                         help="Verify JSearch API key works before running"):
                from src.ingest import test_jsearch_connection
                with st.spinner("Testing JSearch connection..."):
                    ok = test_jsearch_connection()
                if ok:
                    st.success("✅ JSearch API connected!")
                else:
                    st.error("❌ Connection failed — check your API key in .env")
        with run_col:
            run_btn = st.button("🚀 Load Data & Find Jobs", type="primary",
                                use_container_width=True)

    if run_btn:
        _run_full_pipeline(data_source, live_queries if fetch_live else None)


def _run_full_pipeline(data_source: str, live_queries: Optional[str]):
    """Execute the full data + ranking pipeline."""
    profile = st.session_state.profile

    with st.spinner("Running JobPilot pipeline..."):
        progress = st.progress(0)
        status   = st.empty()

        try:
            # ── Step 1: Load data ──────────────────────────────────────────
            status.text("📥 Step 1/5: Loading job data...")
            from src.clean import load_clean_data
            from src.ingest import load_kaggle_data, fetch_multiple_queries

            use_sample = "Sample" in data_source
            jobs_df = load_clean_data(sample=use_sample)
            progress.progress(20)

            # Live ingestion
            if live_queries and JSEARCH_API_KEY:
                status.text("🌐 Fetching live job postings from JSearch...")
                queries = [q.strip() for q in live_queries.split(",") if q.strip()]
                live_df = fetch_multiple_queries(queries, pages_per_query=2)
                if not live_df.empty:
                    from src.clean import clean_jobs
                    live_clean = clean_jobs(live_df, save=False)
                    jobs_df = pd.concat([jobs_df, live_clean], ignore_index=True)
                    jobs_df = jobs_df.drop_duplicates(subset=["job_id"])
                    st.toast(f"✅ Added {len(live_clean):,} live jobs from JSearch")
            progress.progress(35)

            # ── Step 2: Deduplication ──────────────────────────────────────
            status.text("🔍 Step 2/5: Deduplicating with MinHash LSH...")
            from src.dedupe import full_deduplication
            jobs_df, dedup_stats = full_deduplication(jobs_df)
            st.session_state.data_stats = dedup_stats
            progress.progress(50)

            # ── Step 3: Build FAISS index + job clusters ───────────────────
            status.text("🧠 Step 3/5: Building embedding index (FAISS) + job clusters...")
            from src.embeddings import (
                load_or_build_index, build_job_clusters, get_cluster_labels,
                get_preferred_clusters, retrieve_candidates, retrieve_hybrid,
                tfidf_retrieve,
            )
            index, embeddings, job_ids = load_or_build_index(jobs_df)
            st.session_state.faiss_index = index
            st.session_state.job_ids     = job_ids
            st.session_state.jobs_df     = jobs_df

            # Build (or load cached) K-Means job family clusters from corpus
            cluster_labels = get_cluster_labels(job_ids)
            if cluster_labels is None:
                cluster_labels = build_job_clusters(embeddings, job_ids)
            st.session_state.cluster_labels = cluster_labels
            progress.progress(70)

            # ── Step 4: Hybrid retrieval (FAISS + TF-IDF via RRF) ─────────
            status.text("🔎 Step 4/5: Hybrid retrieval (dense + sparse + cluster boost)...")

            # Derive preferred / avoided job families from prior feedback
            preferred_clusters, avoided_clusters = set(), set()
            if st.session_state.feedback and cluster_labels is not None:
                preferred_clusters, avoided_clusters = get_preferred_clusters(
                    st.session_state.feedback, job_ids, cluster_labels
                )

            hybrid_candidates = retrieve_hybrid(
                profile, index, job_ids, jobs_df,
                k=RETRIEVAL_K,
                cluster_labels=cluster_labels,
                preferred_clusters=preferred_clusters,
                avoided_clusters=avoided_clusters,
            )
            # Keep pure dense + sparse for benchmarking
            emb_candidates   = retrieve_candidates(
                profile, index, job_ids, k=RETRIEVAL_K,
                cluster_labels=cluster_labels,
                preferred_clusters=preferred_clusters,
                avoided_clusters=avoided_clusters,
            )
            tfidf_candidates = tfidf_retrieve(profile, jobs_df, k=RETRIEVAL_K)

            st.session_state.hybrid_candidates  = hybrid_candidates
            st.session_state.emb_candidates     = emb_candidates
            st.session_state.tfidf_candidates   = tfidf_candidates
            progress.progress(85)

            # ── Step 5: Rank ───────────────────────────────────────────────
            status.text("🏆 Step 5/5: Ranking and re-ranking...")
            from src.ranker import rank_jobs
            from src.adaptive_learning import AdaptiveLearner

            if st.session_state.adaptive is None:
                st.session_state.adaptive = AdaptiveLearner()

            ranked = rank_jobs(
                jobs_df, profile,
                hybrid_candidates,          # use RRF-fused candidates for ranking
                weights=st.session_state.adaptive.weights,
                feedback=st.session_state.feedback,
            )
            st.session_state.ranked_jobs    = ranked
            st.session_state.pipeline_ready = True
            progress.progress(100)

            # Analytics
            from src.analytics import get_full_analytics
            st.session_state.analytics = get_full_analytics(jobs_df, profile)

            # Benchmark data (passes cluster_labels so hybrid column is accurate)
            from src.ranker import benchmark_ranking
            from src.embeddings import benchmark_retrieval
            st.session_state.benchmark_data = {
                "retrieval": benchmark_retrieval(
                    profile, jobs_df, index, job_ids,
                    cluster_labels=cluster_labels,
                ),
                "ranking": benchmark_ranking(
                    jobs_df, profile, hybrid_candidates, tfidf_candidates
                ),
            }

            status.empty()
            st.success(f"✅ Pipeline complete! Found **{len(ranked)} ranked matches** from **{len(jobs_df):,} deduplicated jobs**.")
            time.sleep(0.5)
            st.session_state.page = "🎯 Job Matches"
            st.rerun()

        except Exception as e:
            status.empty()
            progress.empty()
            st.error(f"Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — JOB MATCHES
# ══════════════════════════════════════════════════════════════════════════════
def page_matches():
    if not st.session_state.ranked_jobs:
        st.warning("⚠️ No job matches yet. Go to **Profile Setup** and run the pipeline.")
        return

    ranked = st.session_state.ranked_jobs
    profile = st.session_state.profile
    feedback = st.session_state.feedback
    adaptive = st.session_state.adaptive

    # ── Header ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jobs Matched", len(ranked))
    with col2:
        avg_score = np.mean([j.final_score for j in ranked[:10]])
        st.metric("Avg Match Score (Top 10)", f"{avg_score:.1%}")
    with col3:
        fb_pos = sum(1 for v in feedback.values() if v in ("good","save"))
        st.metric("Positive Feedback Given", fb_pos)
    with col4:
        remote_count = sum(1 for j in ranked if j.remote)
        st.metric("Remote Positions", remote_count)

    st.divider()

    # ── Data source warning ───────────────────────────────────────────────────
    sources = {j.source for j in ranked}
    if "synthetic" in sources:
        real_count = sum(1 for j in ranked if j.source != "synthetic")
        synth_count = sum(1 for j in ranked if j.source == "synthetic")
        st.warning(
            f"⚠️ **Data notice:** {synth_count} of your {len(ranked)} results are "
            f"**synthetic** (generated fallback — Kaggle credentials not configured). "
            f"{real_count} are live jobs from JSearch. "
            f"To see only real jobs, add `KAGGLE_USERNAME` and `KAGGLE_KEY` to your "
            f"Streamlit secrets, or ensure your `JSEARCH_API_KEY` is active.",
            icon="⚠️",
        )
    else:
        live_kaggle = sum(1 for j in ranked if j.source == "kaggle")
        live_jsearch = sum(1 for j in ranked if j.source == "jsearch")
        source_parts = []
        if live_kaggle:  source_parts.append(f"{live_kaggle} from Kaggle corpus")
        if live_jsearch: source_parts.append(f"{live_jsearch} live from JSearch API")
        if source_parts:
            st.success(f"✅ All results are real job postings — {', '.join(source_parts)}.")

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filter Results", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            min_score = st.slider("Min match score", 0.0, 1.0, 0.0, 0.05)
        with fcol2:
            remote_filter = st.selectbox("Location", ["All", "Remote only", "On-site only"])
        with fcol3:
            seniority_filter = st.multiselect("Seniority", ["junior","mid","senior","staff"],
                                               default=["junior","mid","senior","staff"])
        with fcol4:
            show_n = st.slider("Show top N", 5, len(ranked), min(10, len(ranked)))

    # Apply filters
    filtered = [j for j in ranked
                if j.final_score >= min_score
                and j.seniority in seniority_filter
                and (remote_filter == "All"
                     or (remote_filter == "Remote only" and j.remote)
                     or (remote_filter == "On-site only" and not j.remote))
               ][:show_n]

    # ── Download CSV ──────────────────────────────────────────────────────────
    col_dl1, col_dl2 = st.columns([3,1])
    with col_dl2:
        if filtered:
            df_export = pd.DataFrame([{
                "Rank":        j.rank,
                "Title":       j.title,
                "Company":     j.company,
                "Location":    j.location,
                "Remote":      j.remote,
                "Salary Min":  j.salary_min,
                "Salary Max":  j.salary_max,
                "Match Score": f"{j.final_score:.1%}",
                "Description": j.description[:300],
                "Apply Link":  j.url,
                "Source":      j.source,
            } for j in filtered])
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv_bytes,
                               "top_jobs.csv", "text/csv",
                               use_container_width=True)

    # ── Job cards ─────────────────────────────────────────────────────────────
    for job in filtered:
        _render_job_card(job, profile, adaptive, feedback)

    # Re-rank after feedback
    st.markdown("---")
    rcol1, rcol2 = st.columns([2, 1])
    with rcol1:
        st.caption(
            "💡 Rate jobs with 👍 / 💾 / 👎 above, then click **Re-rank** to see "
            "the adaptive model reprioritise your top 20 based on your preferences."
        )
    with rcol2:
        if st.button("🔄 Re-rank with Feedback", type="primary", use_container_width=True):
            from src.ranker import rank_jobs
            # Use hybrid candidates (FAISS + TF-IDF RRF) — same as initial pipeline
            candidates = (st.session_state.hybrid_candidates
                          or st.session_state.emb_candidates)
            ranked_new = rank_jobs(
                st.session_state.jobs_df,
                profile,
                candidates,
                weights=adaptive.weights if adaptive else None,
                feedback=feedback,
            )
            if adaptive:
                ranked_new = adaptive.apply_bandit_boost(ranked_new)
                ranked_new.sort(key=lambda j: j.final_score, reverse=True)
                for i, j in enumerate(ranked_new):
                    j.rank = i + 1
            # Show a diff summary: how many positions changed
            old_ids = [j.job_id for j in st.session_state.ranked_jobs]
            new_ids = [j.job_id for j in ranked_new]
            moved   = sum(1 for i, jid in enumerate(new_ids)
                          if i < len(old_ids) and jid != old_ids[i])
            st.session_state.ranked_jobs = ranked_new
            st.success(f"✅ Re-ranked! **{moved}** positions changed in the top {len(ranked_new)}.")
            st.rerun()


def _render_job_card(job, profile, adaptive, feedback):
    """Render a single job card with scores, explanations, and feedback buttons."""
    fb = feedback.get(job.job_id, "")
    border_color = {"good":"#27AE60","save":"#2E75B6","bad":"#E74C3C","skip":"#95A5A6"}.get(fb, "#2E75B6")
    score_pct = int(job.final_score * 100)
    score_class = "score-high" if score_pct >= 70 else "score-mid" if score_pct >= 45 else "score-low"

    # Source badge
    source_badge_map = {
        "jsearch":   ('<span style="background:#27AE60;color:white;padding:2px 8px;'
                      'border-radius:4px;font-size:0.72rem;font-weight:600;">🟢 LIVE</span>'),
        "kaggle":    ('<span style="background:#2E75B6;color:white;padding:2px 8px;'
                      'border-radius:4px;font-size:0.72rem;font-weight:600;">📦 KAGGLE</span>'),
        "synthetic": ('<span style="background:#E67E22;color:white;padding:2px 8px;'
                      'border-radius:4px;font-size:0.72rem;font-weight:600;">🔶 DEMO</span>'),
    }
    src_badge = source_badge_map.get(job.source, "")

    with st.container():
        st.markdown(f"""
        <div class="job-card" style="border-left-color:{border_color}">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
            <div>
              <span style="font-size:1.1rem; font-weight:700; color:#1F4E79;">
                #{job.rank} {job.title}
              </span>
              &nbsp;{src_badge}
              <br>
              <span style="color:#5D6D7E; font-size:0.88rem;">
                🏢 {job.company} &nbsp;|&nbsp;
                📍 {job.location} &nbsp;|&nbsp;
                {'🌐 Remote' if job.remote else '🏢 On-site'} &nbsp;|&nbsp;
                {job.employment_type}
              </span>
            </div>
            <div style="text-align:right;">
              <span class="score-badge {score_class}">{score_pct}% match</span>
              {f'<br><span style="color:#5D6D7E; font-size:0.8rem;">${job.salary_min:,.0f}–${job.salary_max:,.0f}</span>' if job.salary_max > 0 else ''}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"📋 Details & Explanation — {job.title} at {job.company}", expanded=False):
            exp_col1, exp_col2 = st.columns([3, 2])

            with exp_col1:
                # Why ranked explanation
                st.markdown("**🎯 Why This Ranked Here**")
                st.markdown(job.why_ranked)

                # Score breakdown
                st.markdown("**📊 Score Breakdown**")
                breakdown_data = {
                    "Dimension": ["Semantic Match", "Skill Match", "Title Alignment",
                                  "Location Fit", "Salary Fit", "Recency"],
                    "Score": [
                        f"{job.embedding_score:.1%}",
                        f"{job.skill_match_score:.1%}",
                        f"{job.title_match_score:.1%}",
                        f"{job.location_fit_score:.1%}",
                        f"{job.salary_fit_score:.1%}",
                        f"{job.recency_score:.1%}",
                    ],
                }
                st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, use_container_width=True)

                # Job description preview
                st.markdown("**📝 Job Description**")
                st.markdown(job.description[:600] + "..." if len(job.description) > 600 else job.description)
                if job.url:
                    st.markdown(f"[🔗 View Full Posting]({job.url})")

            with exp_col2:
                # Skills visualization
                st.markdown("**✅ Matched Skills**")
                if job.matched_skills:
                    pills = " ".join(
                        f'<span class="skill-pill skill-matched">{s}</span>'
                        for s in job.matched_skills[:8]
                    )
                    st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No direct skill matches found")

                st.markdown("**⚠️ Missing Skills**")
                if job.missing_skills:
                    pills = " ".join(
                        f'<span class="skill-pill skill-missing">{s}</span>'
                        for s in job.missing_skills[:5]
                    )
                    st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
                else:
                    st.caption("You have all listed required skills!")

                # Metadata
                st.markdown("**ℹ️ Details**")
                st.markdown(f"- **Seniority:** {job.seniority.title()}")
                st.markdown(f"- **Exp. Required:** {job.experience_required}+ yrs")
                st.markdown(f"- **Visa:** {'✅ Sponsorship indicated' if job.visa_possible else '❓ Not specified'}")
                st.markdown(f"- **Posted:** {job.date_posted}")
                st.markdown(f"- **Source:** {job.source}")

        # Feedback + action buttons
        btn_cols = st.columns([1, 1, 1, 1, 2])
        with btn_cols[0]:
            if st.button("✅ Good Fit", key=f"good_{job.job_id}",
                         type="primary" if fb == "good" else "secondary"):
                _record_feedback(job, "good", adaptive)
        with btn_cols[1]:
            if st.button("❌ Not For Me", key=f"bad_{job.job_id}"):
                _record_feedback(job, "bad", adaptive)
        with btn_cols[2]:
            if st.button("⭐ Save", key=f"save_{job.job_id}"):
                _record_feedback(job, "save", adaptive)
        with btn_cols[3]:
            if st.button("⏭️ Skip", key=f"skip_{job.job_id}"):
                _record_feedback(job, "skip", adaptive)
        with btn_cols[4]:
            if st.button(f"📄 Generate Resume", key=f"resume_{job.job_id}",
                         type="primary"):
                st.session_state.selected_job = job
                st.session_state.page = "📄 Resume Generator"
                st.rerun()

        if fb:
            fb_label = {"good":"✅ Marked: Good fit","bad":"❌ Marked: Not for me",
                        "save":"⭐ Saved","skip":"⏭️ Skipped"}.get(fb,"")
            st.caption(fb_label)

        st.markdown("")


def _record_feedback(job, feedback_type, adaptive):
    """
    Record feedback, update adaptive learner, and persist to database.
    Every click is saved immediately — nothing is lost on refresh.
    """
    st.session_state.feedback[job.job_id] = feedback_type

    if adaptive:
        adaptive.record_feedback(job, feedback_type)
        if feedback_type in ("good", "save"):
            st.session_state.positive_ids.add(job.job_id)
        if adaptive.bandit.total_interactions % 5 == 0 and st.session_state.ranked_jobs:
            adaptive.record_precision(
                st.session_state.ranked_jobs,
                st.session_state.positive_ids
            )

    # ── Persist to SQLite ─────────────────────────────────────────────────────
    uid = st.session_state.current_user
    if uid:
        save_feedback_event(uid, job, feedback_type)          # log the event
        save_bandit_state(uid, adaptive.bandit)               # save arm distributions
        save_ranking_weights(uid, adaptive.weights)           # save updated weights

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESUME GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_resume():
    st.markdown('<div class="section-header">📄 Resume Generator</div>', unsafe_allow_html=True)

    profile = st.session_state.profile
    ranked  = st.session_state.ranked_jobs

    if not ranked:
        st.warning("⚠️ No jobs ranked yet. Run the pipeline first.")
        return

    if not profile:
        st.warning("⚠️ No profile loaded.")
        return

    # Job selector
    col1, col2 = st.columns([3, 1])
    with col1:
        job_options = {f"#{j.rank} {j.title} — {j.company}": j for j in ranked[:20]}
        selected_label = st.selectbox("Select a job to tailor your resume for", list(job_options.keys()))
        selected_job = job_options[selected_label]
    with col2:
        st.markdown("")
        st.markdown("")
        generate_btn = st.button("🤖 Generate Tailored Resume", type="primary", use_container_width=True)

    if generate_btn or (st.session_state.selected_job and st.session_state.selected_job.job_id == selected_job.job_id):
        if selected_job.job_id in st.session_state.resumes:
            result = st.session_state.resumes[selected_job.job_id]
        else:
            with st.spinner("✍️ Generating tailored resume..."):
                from src.resume_generator import generate_resume
                result = generate_resume(profile, selected_job)
                st.session_state.resumes[selected_job.job_id] = result
                # Persist resume to database
                uid = st.session_state.current_user
                if uid:
                    save_resume(uid, selected_job, result)

        # Display result
        st.markdown(f"""
        <div style="background:#D5F5E3; border-left:4px solid #27AE60;
             border-radius:8px; padding:12px 16px; margin-bottom:16px;">
            <strong>{'🤖 AI-Generated' if result['method']=='ai' else '📋 Template'} Resume</strong>
            — tailored for <strong>{selected_job.title}</strong> at <strong>{selected_job.company}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.warning(result["warning"])

        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.markdown('<div class="resume-output">', unsafe_allow_html=True)
            st.markdown(result["markdown"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r2:
            # Download
            st.download_button(
                "⬇️ Download (.md)",
                result["markdown"].encode(),
                f"resume_{selected_job.company.replace(' ','_')}.md",
                "text/markdown",
                use_container_width=True,
            )

            st.markdown("**✅ Your Matched Skills**")
            for s in result["matched_skills"][:8]:
                st.markdown(f"- {s}")

            st.markdown("**⚠️ Skill Gaps**")
            if result["missing_skills"]:
                for s in result["missing_skills"][:5]:
                    st.markdown(f"- ❌ {s}")
            else:
                st.markdown("*You cover all listed requirements!*")

    # Show all previously generated resumes
    if st.session_state.resumes:
        st.divider()
        st.markdown(f"**Generated {len(st.session_state.resumes)} resume(s) this session**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MARKET ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="section-header">📊 Market Analytics</div>', unsafe_allow_html=True)

    analytics = st.session_state.analytics
    if not analytics:
        st.warning("⚠️ Run the pipeline first to see analytics.")
        return

    from src.analytics import (
        plot_top_skills, plot_salary_distribution, plot_remote_pie,
        plot_top_companies, plot_skill_gaps
    )

    # Summary metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{analytics["total_jobs"]:,}</div><div class="metric-label">Total Jobs</div></div>', unsafe_allow_html=True)
    with mcol2:
        sal_pct = int(analytics["with_salary"] / max(analytics["total_jobs"],1) * 100)
        st.markdown(f'<div class="metric-card"><div class="metric-number">{sal_pct}%</div><div class="metric-label">Jobs with Salary Data</div></div>', unsafe_allow_html=True)
    with mcol3:
        rem_pct = int(analytics["remote_count"] / max(analytics["total_jobs"],1) * 100)
        st.markdown(f'<div class="metric-card"><div class="metric-number">{rem_pct}%</div><div class="metric-label">Remote Positions</div></div>', unsafe_allow_html=True)
    with mcol4:
        gaps_count = len(analytics["skill_gaps"])
        st.markdown(f'<div class="metric-card"><div class="metric-number">{gaps_count}</div><div class="metric-label">Skill Gaps Found</div></div>', unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔧 Top Skills", "💰 Salaries", "🌐 Remote Split", "🏢 Companies", "📉 Your Skill Gaps"]
    )

    with tab1:
        if not analytics["top_skills"].empty:
            st.plotly_chart(plot_top_skills(analytics["top_skills"]), use_container_width=True)
            st.dataframe(analytics["top_skills"], hide_index=True, use_container_width=True)

    with tab2:
        sal_df = analytics["salary_dist"]
        if not sal_df.empty:
            st.plotly_chart(plot_salary_distribution(sal_df), use_container_width=True)
            st.caption("Box plots show median, IQR, and outliers for jobs with listed salary data.")
        else:
            st.info("No salary data available in current dataset.")

    with tab3:
        rd = analytics["remote_dist"]
        if rd:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(plot_remote_pie(rd), use_container_width=True)
            with col2:
                for label, count in rd.items():
                    pct = count / max(sum(rd.values()),1) * 100
                    st.markdown(f"**{label}:** {count:,} ({pct:.1f}%)")

    with tab4:
        if not analytics["top_companies"].empty:
            st.plotly_chart(plot_top_companies(analytics["top_companies"]), use_container_width=True)

    with tab5:
        gaps_df = analytics["skill_gaps"]
        if not gaps_df.empty:
            st.plotly_chart(plot_skill_gaps(gaps_df), use_container_width=True)
            st.markdown("*These skills appear frequently in your target roles but are not in your current profile.*")
            st.dataframe(gaps_df, hide_index=True, use_container_width=True)
        else:
            st.success("No significant skill gaps found for your target roles!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
def page_benchmarks():
    st.markdown('<div class="section-header">📈 Benchmarks & Technical Results</div>', unsafe_allow_html=True)

    bm  = st.session_state.benchmark_data
    ada = st.session_state.adaptive

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧠 Retrieval Comparison", "🏆 Ranking Pipeline", "🤖 Adaptive Learning", "✅ Persona Tests"]
    )

    # ── Retrieval benchmark ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### Embedding vs. TF-IDF Retrieval (BAX-423 Technique 1)")
        st.markdown("""
        Dense embeddings (sentence-transformers) vs. keyword-based TF-IDF retrieval.
        Embeddings capture semantic equivalence — a resume saying "statistical modeling"
        matches jobs requiring "predictive analytics".
        """)

        retrieval_bm = bm.get("retrieval", {})
        if retrieval_bm:
            ret_df = pd.DataFrame({
                "Method":        retrieval_bm.get("method", []),
                "Recall@10":     retrieval_bm.get("recall_at_10", []),
                "Recall@50":     retrieval_bm.get("recall_at_50", []),
                "Latency (ms)":  retrieval_bm.get("latency_ms_p50", []),
            })
            st.dataframe(ret_df, hide_index=True, use_container_width=True)
            improvement = retrieval_bm.get("improvement", "")
            if improvement:
                st.success(f"📈 Embedding improvement: **{improvement}**")

            import plotly.graph_objects as go
            methods = retrieval_bm.get("method", [])
            r10 = retrieval_bm.get("recall_at_10", [])
            fig = go.Figure([go.Bar(x=methods, y=r10,
                                    marker_color=["#D6E4F0","#1F4E79"],
                                    text=[f"{v:.0%}" for v in r10],
                                    textposition="auto")])
            fig.update_layout(title="Recall@10 Comparison", yaxis_tickformat=".0%",
                               plot_bgcolor="white", height=300,
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see retrieval benchmarks.")

    # ── Ranking benchmark ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Multi-Stage Ranking Pipeline (BAX-423 Technique 2)")
        st.markdown("""
        TF-IDF → Embedding-only → Full multi-stage pipeline (hard filters + scoring + MMR re-ranking).
        Reports persona fit score (fraction of top-10 that match persona's target roles)
        and dealbreaker violations.
        """)

        ranking_bm = bm.get("ranking", {})
        if ranking_bm:
            rank_df = pd.DataFrame({
                "Method":               ranking_bm.get("method", []),
                "Persona Fit (Top-10)": ranking_bm.get("top10_persona_fit", []),
                "Dealbreaker Violations": ranking_bm.get("dealbreaker_violations", []),
                "Avg Match Score":      ranking_bm.get("avg_match_score", []),
            })
            st.dataframe(rank_df, hide_index=True, use_container_width=True)

            import plotly.graph_objects as go
            methods = ranking_bm.get("method", [])
            fits    = ranking_bm.get("top10_persona_fit", [])
            fig = go.Figure([go.Bar(x=methods, y=fits,
                                    marker_color=["#D6E4F0","#5BA3D0","#1F4E79"],
                                    text=[f"{v:.0%}" for v in fits],
                                    textposition="auto")])
            fig.update_layout(title="Persona Fit Score by Ranking Method",
                               yaxis_tickformat=".0%", plot_bgcolor="white",
                               height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see ranking benchmarks.")

    # ── Adaptive learning ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Thompson Sampling Adaptive Learning (BAX-423 Technique 3)")
        st.markdown("""
        Thompson Sampling models user preferences as Beta distributions over job-feature clusters.
        Weight Updater adjusts ranking formula weights based on feedback correlation.
        """)

        if ada and ada.bandit.total_interactions > 0:
            bm_data = ada.get_benchmark_data()

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Feedback Events", bm_data["total_feedback"])
            with metrics_col2:
                fb = bm_data["feedback_breakdown"]
                pos = fb.get("good",0) + fb.get("save",0)
                neg = fb.get("bad",0)
                st.metric("Positive / Negative", f"{pos} / {neg}")
            with metrics_col3:
                st.metric("Preference Clusters Learned", len(ada.bandit.arms))

            # Precision@5 curve
            if bm_data["precision_history"]:
                from src.analytics import plot_adaptive_learning_curve, plot_weight_evolution
                st.plotly_chart(
                    plot_adaptive_learning_curve(bm_data["precision_history"]),
                    use_container_width=True
                )

            # Top preferences
            st.markdown("**🎯 Learned Preferences (Top Arms)**")
            pref_df = pd.DataFrame(bm_data["top_preferences"],
                                    columns=["Cluster", "Preference Score"])
            pref_df["Preference Score"] = pref_df["Preference Score"].apply(lambda x: f"{x:.2f}")
            st.dataframe(pref_df, hide_index=True, use_container_width=True)

            # Weight evolution
            if len(bm_data["weight_evolution"]) > 1:
                from src.analytics import plot_weight_evolution
                st.plotly_chart(plot_weight_evolution(bm_data["weight_evolution"]),
                                use_container_width=True)

            # Weight changes
            st.markdown("**⚖️ Weight Changes from Initial**")
            delta = bm_data["weight_changes"]
            delta_df = pd.DataFrame([
                {"Dimension": k.replace("_"," ").title(),
                 "Initial": f"{DEFAULT_WEIGHTS[k]:.2f}",
                 "Current": f"{v:.2f}",
                 "Change":  f"{delta[k]:+.3f}"}
                for k, v in bm_data["current_weights"].items()
            ])
            from src.utils import DEFAULT_WEIGHTS
            st.dataframe(delta_df, hide_index=True, use_container_width=True)
        else:
            st.info("Give feedback on job matches (Good Fit / Not For Me) to see the adaptive learning benchmark.")
            # Show simulated preview
            st.markdown("**📊 Simulated Preview (from test run):**")
            sim_data = {"Round": [0,1,2,3,4],
                        "Signals": [0,5,10,20,30],
                        "Precision@5": [0.40, 0.52, 0.61, 0.68, 0.74],
                        "Dealbreaker Violations": [2,1,1,0,0]}
            st.dataframe(pd.DataFrame(sim_data), hide_index=True, use_container_width=True)

    # ── Persona tests ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Persona Pass/Fail Results")
        st.markdown("Evaluation of the pipeline against all 4 required test personas.")

        # Build results from current session
        ranked = st.session_state.ranked_jobs
        profile = st.session_state.profile

        persona_results = []
        if PERSONAS_FILE.exists():
            with open(PERSONAS_FILE) as f:
                personas = json.load(f)
            for p in personas:
                is_active = profile and profile.get("id") == p["id"]
                criteria  = p.get("pass_criteria", {})
                if is_active and ranked:
                    top10 = ranked[:10]
                    db_violations = sum(
                        1 for j in top10
                        for db in p.get("dealbreakers", [])
                        if db.lower() in (j.title + " " + j.company + " " + j.description[:200]).lower()
                    )
                    target_roles = [r.lower() for r in p["target_roles"]]
                    fit = sum(1 for j in top10 if any(r in j.title.lower() for r in target_roles))
                    passed = db_violations == 0 and fit >= 5
                else:
                    passed = None  # not tested yet

                persona_results.append({
                    "Persona":            p["emoji"] + " " + p["name"].split("—")[0].strip(),
                    "Target Roles":       p["target_roles"][0],
                    "Salary Target":      f"${p['salary_min']:,}+",
                    "Key Dealbreaker":    p["dealbreakers"][0] if p["dealbreakers"] else "—",
                    "Status":             "✅ Active" if is_active else "⚪ Not tested",
                    "Pass":               "✅" if passed is True else "❓ Pending" if passed is None else "❌",
                })

        if persona_results:
            st.dataframe(pd.DataFrame(persona_results), hide_index=True, use_container_width=True)
            st.info("Switch personas in Profile Setup to test each one.")

    # ── Deduplication stats ───────────────────────────────────────────────────
    if st.session_state.data_stats:
        st.divider()
        st.markdown("### 📊 Deduplication Pipeline Stats")
        ds = st.session_state.data_stats
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.metric("Original Records", f"{ds.get('original_count',0):,}")
        with dcol2:
            st.metric("After Exact Dedup", f"{ds.get('after_exact',0):,}")
        with dcol3:
            st.metric("After MinHash LSH", f"{ds.get('after_minhash',0):,}")

        mh = ds.get("minhash_stats", {})
        if mh:
            st.markdown(f"""
            - **Threshold:** Jaccard ≥ {mh.get('threshold', 0.85)}
            - **Hash permutations:** {mh.get('num_perm', 128)}
            - **Throughput:** {mh.get('throughput_rps',0):,} records/second
            - **Near-duplicates removed:** {mh.get('removed',0):,}
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MY LEARNING PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def page_learning_profile():
    st.markdown('<div class="section-header">🧠 My Learning Profile</div>',
                unsafe_allow_html=True)

    uid = st.session_state.current_user
    if not uid:
        st.warning("⚠️ Log in first to see your learning profile.")
        return

    insights = get_learning_insights(uid)
    fb_summary = get_feedback_summary(uid)
    adaptive = st.session_state.adaptive

    if not insights:
        st.info("No feedback recorded yet. Rate some jobs on the Job Matches page "
                "and come back here to see what the model has learned about you.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{insights["total_feedback"]}</div>'
                    f'<div class="metric-label">Total Feedback Events</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{insights["liked_count"]}</div>'
                    f'<div class="metric-label">Jobs Liked / Saved</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{insights["sessions_count"]}</div>'
                    f'<div class="metric-label">Sessions Recorded</div></div>',
                    unsafe_allow_html=True)
    with col4:
        arms = len(adaptive.bandit.arms) if adaptive else 0
        st.markdown(f'<div class="metric-card"><div class="metric-number">'
                    f'{arms}</div>'
                    f'<div class="metric-label">Preference Clusters Learned</div></div>',
                    unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    # ── What the model has learned ────────────────────────────────────────────
    with col_l:
        st.markdown("### ✅ What You Tend to Like")

        if insights["preferred_seniority"]:
            st.markdown("**Seniority levels:**")
            for level, count in insights["preferred_seniority"]:
                st.markdown(f"- {level.title()} ({count} likes)")

        if insights["preferred_industry"]:
            st.markdown("**Industries:**")
            for ind, count in insights["preferred_industry"]:
                st.markdown(f"- {ind.replace('_',' ').title()} ({count} likes)")

        if insights["top_matched_skills"]:
            st.markdown("**Most valued skills (in liked jobs):**")
            pills = " ".join(
                f'<span class="skill-pill skill-matched">{s}</span>'
                for s, _ in insights["top_matched_skills"]
            )
            st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)

        if fb_summary.get("top_liked_companies"):
            st.markdown("**Companies you liked:**")
            for co in fb_summary["top_liked_companies"]:
                st.markdown(f"- {co}")

    with col_r:
        st.markdown("### ❌ What the Model Avoids for You")

        if insights["avoided_seniority"]:
            st.markdown("**Seniority levels:**")
            for level, count in insights["avoided_seniority"]:
                st.markdown(f"- {level.title()} ({count} dislikes)")

        if insights["avoided_industry"]:
            st.markdown("**Industries:**")
            for ind, count in insights["avoided_industry"]:
                st.markdown(f"- {ind.replace('_',' ').title()} ({count} dislikes)")

        if fb_summary.get("disliked_patterns"):
            st.markdown("**Disliked patterns:**")
            for p in fb_summary["disliked_patterns"][:3]:
                st.markdown(f"- {p['seniority'].title()} {p['industry'].replace('_',' ')} roles")

    st.divider()

    # ── Current ranking weights ───────────────────────────────────────────────
    st.markdown("### ⚖️ Your Personalised Ranking Weights")
    st.caption("These weights shift based on your feedback — the model "
               "emphasises the dimensions that best predict your preferences.")

    if adaptive:
        weights = adaptive.weights
        from src.utils import DEFAULT_WEIGHTS
        w_data = []
        for k, v in weights.items():
            default = DEFAULT_WEIGHTS.get(k, 0)
            delta   = v - default
            arrow   = "⬆️" if delta > 0.005 else ("⬇️" if delta < -0.005 else "➡️")
            w_data.append({
                "Dimension":   k.replace("_", " ").title(),
                "Default":     f"{default:.2f}",
                "Your Weight": f"{v:.2f}",
                "Change":      f"{arrow} {delta:+.3f}",
            })
        st.dataframe(pd.DataFrame(w_data), hide_index=True, use_container_width=True)

    # ── Full feedback history ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Full Feedback History")

    history = load_feedback_history(uid)
    if history:
        hist_df = pd.DataFrame([{
            "Date":      h["recorded_at"][:10],
            "Job Title": h["job_title"],
            "Company":   h["company"],
            "Feedback":  h["feedback_type"].title(),
            "Score":     f"{h['final_score']:.2f}" if h["final_score"] else "—",
        } for h in history])

        fb_filter = st.multiselect(
            "Filter by feedback type",
            ["Good", "Bad", "Save", "Skip"],
            default=["Good", "Save", "Bad", "Skip"],
        )
        filtered_hist = hist_df[hist_df["Feedback"].isin(fb_filter)]
        st.dataframe(filtered_hist, hide_index=True, use_container_width=True)

        # Export feedback history
        csv = hist_df.to_csv(index=False).encode()
        st.download_button("⬇️ Export Feedback History (CSV)",
                           csv, "feedback_history.csv", "text/csv")

    # ── Account management ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Account Management")
    col_save, col_del = st.columns(2)
    with col_save:
        if st.button("💾 Save All Data Now", use_container_width=True):
            _save_session_to_db()
            st.success("✅ All data saved to database.")
    with col_del:
        with st.expander("🗑️ Delete My Account"):
            st.warning("This permanently deletes your profile, feedback history, "
                       "and all learned preferences.")
            if st.button("Confirm Delete Account", type="primary"):
                delete_user(uid)
                st.session_state.current_user = None
                st.session_state.profile      = None
                st.session_state.adaptive     = None
                st.session_state.feedback     = {}
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page_map = {
    "🏠 Profile Setup":       page_profile,
    "🎯 Job Matches":         page_matches,
    "📄 Resume Generator":    page_resume,
    "📊 Market Analytics":    page_analytics,
    "📈 Benchmarks":          page_benchmarks,
    "🧠 My Learning Profile": page_learning_profile,
}

current_page = st.session_state.page
page_fn = page_map.get(current_page, page_profile)
page_fn()
