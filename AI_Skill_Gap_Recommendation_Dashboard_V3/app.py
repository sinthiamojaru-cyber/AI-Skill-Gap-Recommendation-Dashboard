
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# APP CONFIG
# ============================================================

APP_TITLE = "AI-Based Skill Gap Analysis and Recommendation for Technical Training in Bangladesh IT Industry"

DATA_DIR = Path(__file__).parent / "data"
CURRICULUM_FILE = DATA_DIR / "Dataset_01_Technical_Training_Curriculum_BD_IT.csv"
INDUSTRY_FILE = DATA_DIR / "Dataset_02_BD_IT_Industry_Skill_Requirements.csv"

FULL_MATCH_THRESHOLD = 0.75
PARTIAL_MATCH_THRESHOLD = 0.45

SKILL_ALIASES = {
    "js": "javascript",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node js": "node.js",
    "restful api": "rest api",
    "restful apis": "rest api",
    "structured query language": "sql",
    "amazon web services": "aws",
    "quality assurance": "software testing",
    "qa": "software testing",
    "problem-solving": "problem solving",
    "team work": "teamwork",
    "cyber security": "cybersecurity",
}

DOMAIN_TO_TITLES = {
    "Software Development": [
        "Junior Software Engineer",
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
    ],
    "Cloud & DevOps": [
        "Cloud Engineer",
        "DevOps Engineer",
    ],
    "Cybersecurity": [
        "Cybersecurity Analyst",
    ],
    "Network & IT Support": [
        "Network Engineer",
        "IT Support Engineer",
    ],
    "Database & QA": [
        "Database Engineer",
        "QA Engineer",
    ],
}

LEVEL_TO_JOB_LEVEL = {
    "Beginner": "Entry",
    "Intermediate": "Junior",
    "Advanced": "Mid",
}

# Category rules reduce unrealistic combinations caused by synthetic data generation.
ROLE_SKILL_FAMILIES = {
    "Junior Software Engineer": {"programming", "web development", "database", "version control", "soft skill"},
    "Software Engineer": {"programming", "web development", "database", "version control", "testing", "soft skill"},
    "Backend Developer": {"programming", "web development", "database", "version control", "devops", "cloud", "soft skill"},
    "Frontend Developer": {"programming", "web development", "version control", "testing", "soft skill"},
    "Full Stack Developer": {"programming", "web development", "database", "version control", "devops", "testing", "soft skill"},
    "QA Engineer": {"testing", "programming", "version control", "soft skill"},
    "Database Engineer": {"database", "programming", "operating system", "soft skill"},
    "Network Engineer": {"networking", "operating system", "cybersecurity", "soft skill"},
    "IT Support Engineer": {"operating system", "networking", "cybersecurity", "soft skill"},
    "DevOps Engineer": {"devops", "cloud", "operating system", "version control", "programming", "soft skill"},
    "Cloud Engineer": {"cloud", "devops", "operating system", "networking", "cybersecurity", "soft skill"},
    "Cybersecurity Analyst": {"cybersecurity", "networking", "operating system", "soft skill"},
}

# Skill-to-training recommendations are limited to course names present in Dataset 01.
SKILL_TO_COURSE = {'python': 'Python Programming', 'java': 'Java Programming', 'javascript': 'Web Development', 'html': 'Web Development', 'css': 'Web Development', 'react': 'Web Development', 'node.js': 'Full Stack Development', 'sql': 'Database Management', 'mysql': 'Database Management', 'mongodb': 'Database Management', 'database design': 'Database Management', 'git': 'DevOps Essentials', 'docker': 'DevOps Essentials', 'jenkins': 'DevOps Essentials', 'aws': 'Cloud Computing', 'linux': 'IT Support', 'networking': 'Networking', 'cybersecurity': 'Cybersecurity', 'selenium': 'Software Testing', 'software testing': 'Software Testing', 'rest api': 'Full Stack Development', 'communication': 'IT Support', 'problem solving': 'Python Programming', 'teamwork': 'Full Stack Development'}

# ============================================================
# STREAMLIT STYLE
# ============================================================

st.set_page_config(
    page_title="Skill Gap Recommendation System",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2.8rem;
        max-width: 1450px;
    }
    .hero {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0;
        opacity: .72;
    }
    .step-title {
        font-size: 1.08rem;
        font-weight: 750;
        margin: 1rem 0 .55rem 0;
    }
    .skill-chip {
        display: inline-block;
        padding: 8px 12px;
        margin: 4px 6px 4px 0;
        border: 1px solid rgba(128,128,128,.34);
        border-radius: 999px;
        font-weight: 650;
    }
    .agent-box {
        padding: 12px 15px;
        border-radius: 13px;
        background: rgba(128,128,128,.08);
        line-height: 1.55;
        margin: .6rem 0;
    }
    .recommend-box {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 16px;
        padding: 15px 17px;
        margin-bottom: 10px;
    }
    .gap-chart {
        width: 100%;
        margin: 12px 0 20px 0;
    }
    .gap-row {
        display: grid;
        grid-template-columns: minmax(150px, 220px) 1fr 42px;
        gap: 12px;
        align-items: center;
        margin: 10px 0;
    }
    .gap-label {
        font-size: .92rem;
        font-weight: 650;
    }
    .gap-track {
        height: 22px;
        border-radius: 999px;
        background: rgba(128,128,128,.14);
        overflow: hidden;
    }
    .gap-fill {
        height: 100%;
        border-radius: 999px;
        background: currentColor;
        opacity: .72;
    }
    .gap-value {
        text-align: right;
        font-weight: 700;
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero {
            padding: 18px 18px;
        }
        .hero h1 {
            font-size: 1.55rem;
        }
        .gap-row {
            grid-template-columns: 1fr;
            gap: 5px;
        }
        .gap-value {
            text-align: left;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA HELPERS
# ============================================================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s+#./-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_skill(value) -> str:
    text = clean_text(value)
    return SKILL_ALIASES.get(text, text)

def normalize_category(value) -> str:
    text = clean_text(value)
    aliases = {
        "security": "cybersecurity",
        "web": "web development",
        "soft skill": "soft skill",
        "programming": "programming",
        "database": "database",
        "cloud": "cloud",
        "devops": "devops",
        "networking": "networking",
        "version control": "version control",
        "operating system": "operating system",
        "web development": "web development",
        "cybersecurity": "cybersecurity",
        "testing": "testing",
    }
    return aliases.get(text, text)

@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    curriculum = pd.read_csv(CURRICULUM_FILE, encoding="utf-8-sig")
    industry = pd.read_csv(INDUSTRY_FILE, encoding="utf-8-sig")

    curriculum["Normalized_Skill"] = curriculum["Skill_Name"].apply(normalize_skill)
    curriculum["Normalized_Category"] = curriculum["Skill_Category"].apply(normalize_category)
    curriculum["Analysis_Text"] = (
        curriculum["Skill_Name"].fillna("").astype(str) + ". " +
        curriculum["Skill_Category"].fillna("").astype(str) + ". " +
        curriculum["Module_Description"].fillna("").astype(str) + ". " +
        curriculum["Learning_Outcome"].fillna("").astype(str)
    )

    industry["Normalized_Skill"] = industry["Required_Skill"].apply(normalize_skill)
    industry["Normalized_Category"] = industry["Skill_Category"].apply(normalize_category)
    industry["Analysis_Text"] = (
        industry["Job_Title"].fillna("").astype(str) + ". " +
        industry["Required_Skill"].fillna("").astype(str) + ". " +
        industry["Skill_Category"].fillna("").astype(str) + ". " +
        industry["Preferred_Skill"].fillna("").astype(str) + ". " +
        industry["Job_Description"].fillna("").astype(str)
    )
    return curriculum, industry

@st.cache_resource(show_spinner=False)
def get_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

# ============================================================
# AGENTIC SEARCH
# ============================================================

def parse_query(query: str, domain: str, title: str, level: str):
    q = clean_text(query)
    resolved_domain = domain
    resolved_title = title
    resolved_level = level

    level_words = {
        "beginner": "Beginner",
        "entry": "Beginner",
        "entry level": "Beginner",
        "intermediate": "Intermediate",
        "junior": "Intermediate",
        "advanced": "Advanced",
        "mid": "Advanced",
        "mid level": "Advanced",
    }
    for token, mapped in level_words.items():
        if token in q:
            resolved_level = mapped
            break

    all_titles = [t for titles in DOMAIN_TO_TITLES.values() for t in titles]
    for candidate in all_titles:
        if clean_text(candidate) in q:
            resolved_title = candidate
            for d, titles in DOMAIN_TO_TITLES.items():
                if candidate in titles:
                    resolved_domain = d
                    break
            return resolved_domain, resolved_title, resolved_level

    domain_keywords = {
        "Software Development": ["software", "backend", "frontend", "full stack", "developer", "programming"],
        "Cloud & DevOps": ["cloud", "devops", "docker", "jenkins", "aws"],
        "Cybersecurity": ["cybersecurity", "cyber", "security"],
        "Network & IT Support": ["network", "networking", "it support", "support engineer", "linux support"],
        "Database & QA": ["database", "qa", "quality assurance", "testing", "selenium", "mysql", "mongodb", "sql"],
    }

    for d, words in domain_keywords.items():
        if any(word in q for word in words):
            resolved_domain = d
            if resolved_title not in DOMAIN_TO_TITLES[d]:
                resolved_title = DOMAIN_TO_TITLES[d][0]
            break

    return resolved_domain, resolved_title, resolved_level

def semantic_retrieve(query: str, rows: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if rows.empty or not query.strip():
        return rows, "Structured filtering"

    texts = rows["Analysis_Text"].tolist()
    model = get_model()

    if model is not None:
        q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        d_vec = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        scores = cosine_similarity(q_vec, d_vec)[0]
        method = "Sentence-BERT semantic retrieval"
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([query] + texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
        method = "TF-IDF retrieval fallback"

    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    result = rows.copy()
    result["Agent_Search_Score"] = scores

    keep = min(max(8, len(result) // 2), len(result))
    return result.sort_values("Agent_Search_Score", ascending=False).head(keep), method

def role_rows(industry: pd.DataFrame, title: str, level: str) -> pd.DataFrame:
    job_level = LEVEL_TO_JOB_LEVEL[level]

    rows = industry[
        (industry["Job_Title"] == title) &
        (industry["Job_Level"] == job_level)
    ].copy()

    if len(rows) < 3:
        rows = industry[industry["Job_Title"] == title].copy()

    allowed_categories = ROLE_SKILL_FAMILIES.get(title, set())
    if allowed_categories:
        filtered = rows[rows["Normalized_Category"].isin(allowed_categories)].copy()
        if not filtered.empty:
            rows = filtered

    return rows

def required_skills(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["Skill", "Category", "Frequency"])

    return (
        rows.groupby("Normalized_Skill", as_index=False)
        .agg(
            Category=("Normalized_Category", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            Frequency=("Normalized_Skill", "size"),
        )
        .rename(columns={"Normalized_Skill": "Skill"})
        .sort_values(["Frequency", "Skill"], ascending=[False, True])
        .reset_index(drop=True)
    )

# ============================================================
# SKILL GAP ENGINE
# ============================================================

def curriculum_profiles(curriculum: pd.DataFrame) -> pd.DataFrame:
    return (
        curriculum.groupby("Normalized_Skill", as_index=False)
        .agg(
            Category=("Normalized_Category", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            Profile_Text=("Analysis_Text", lambda s: " ".join(s.astype(str))),
            Frequency=("Normalized_Skill", "size"),
            Max_Level=("Skill_Level", lambda s: _highest_level(s.astype(str).tolist())),
        )
        .rename(columns={"Normalized_Skill": "Skill"})
    )

def _highest_level(levels: List[str]) -> str:
    rank = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    valid = [x for x in levels if x in rank]
    return max(valid, key=lambda x: rank[x]) if valid else "Beginner"

def _level_score(curriculum_level: str, requested_level: str) -> float:
    rank = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    c = rank.get(curriculum_level, 1)
    r = rank.get(requested_level, 1)
    if c >= r:
        return 1.0
    if c == r - 1:
        return 0.65
    return 0.35

def analyze_gaps(
    required: pd.DataFrame,
    curriculum: pd.DataFrame,
    requested_level: str,
) -> Tuple[pd.DataFrame, str]:
    if required.empty:
        return pd.DataFrame(), "N/A"

    profiles = curriculum_profiles(curriculum)
    demand_texts = [
        f"{row.Skill}. {row.Category}. IT industry required competency."
        for row in required.itertuples()
    ]
    curr_texts = profiles["Profile_Text"].tolist()

    model = get_model()
    if model is not None:
        demand_vecs = model.encode(
            demand_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        curriculum_vecs = model.encode(
            curr_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        semantic_method = "Sentence-BERT (all-MiniLM-L6-v2)"
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(demand_texts + curr_texts).toarray()
        demand_vecs = matrix[:len(demand_texts)]
        curriculum_vecs = matrix[len(demand_texts):]
        semantic_method = "TF-IDF fallback"

    sim = cosine_similarity(demand_vecs, curriculum_vecs)
    sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)

    curriculum_skill_set = set(profiles["Skill"])
    out = []

    for i, row in required.iterrows():
        req_skill = row["Skill"]
        exact_match = req_skill in curriculum_skill_set

        best_idx = int(np.argmax(sim[i]))
        best_profile = profiles.iloc[best_idx]
        best_skill = best_profile["Skill"]
        semantic_score = float(sim[i, best_idx])

        if exact_match:
            exact_idx = int(profiles.index[profiles["Skill"] == req_skill][0])
            exact_profile = profiles.loc[exact_idx]
            semantic_score = max(float(sim[i, exact_idx]), 0.90)
            best_skill = req_skill
            best_profile = exact_profile

        curriculum_level = str(best_profile["Max_Level"])
        level_score = _level_score(curriculum_level, requested_level)

        # V3 combines semantic alignment with training-level adequacy.
        # This prevents all exact matches from automatically appearing equally strong.
        combined_score = (0.75 * semantic_score) + (0.25 * level_score)
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        if combined_score >= FULL_MATCH_THRESHOLD:
            status = "Fully Matched"
        elif combined_score >= PARTIAL_MATCH_THRESHOLD:
            status = "Partially Matched"
        else:
            status = "Missing / Low Alignment"

        if status == "Fully Matched":
            gap_priority = "No Additional Training"
        elif status == "Partially Matched":
            gap_priority = "Medium"
        else:
            gap_priority = "High"

        if status == "Fully Matched":
            recommended_training = "No Additional Training Required"
            recommendation = (
                f"Curriculum coverage is adequate for {req_skill.title()} at the selected "
                f"{requested_level} level. Continue practical reinforcement and periodic review."
            )
        else:
            recommended_training = SKILL_TO_COURSE.get(req_skill, "Curriculum Enhancement Module")
            if status == "Partially Matched":
                recommendation = (
                    f"Strengthen {req_skill.title()} within {recommended_training} with more "
                    f"practical activities appropriate to the {requested_level} level."
                )
            else:
                recommendation = (
                    f"Add focused {req_skill.title()} content to {recommended_training} because "
                    f"the current curriculum shows low alignment with the selected job requirement."
                )

        out.append({
            "Required Skill": req_skill.upper() if req_skill in {"sql", "aws"} else req_skill.title(),
            "Industry Frequency": int(row["Frequency"]),
            "Best Curriculum Match": best_skill.title(),
            "Curriculum Max Level": curriculum_level,
            "Semantic Similarity": round(semantic_score, 3),
            "Level Adequacy": round(level_score, 2),
            "Alignment Score": round(combined_score, 3),
            "Gap Status": status,
            "Recommended Training": recommended_training,
            "Priority": gap_priority,
            "Recommendation": recommendation,
        })

    result = pd.DataFrame(out)

    order = {
        "Missing / Low Alignment": 0,
        "Partially Matched": 1,
        "Fully Matched": 2,
    }
    result["_sort"] = result["Gap Status"].map(order)
    result = result.sort_values(
        ["_sort", "Industry Frequency"],
        ascending=[True, False],
    ).drop(columns="_sort")

    return result.reset_index(drop=True), semantic_method

# ============================================================
# DISPLAY HELPERS
# ============================================================

def skill_chips(df: pd.DataFrame) -> str:
    html = ""
    for row in df.itertuples():
        name = row.Skill.upper() if row.Skill in {"sql", "aws"} else row.Skill.title()
        html += f'<span class="skill-chip">{name} · {row.Frequency}</span>'
    return html

# ============================================================
# APP
# ============================================================

curriculum, industry = load_data()

st.markdown(
    f"""
    <div class="hero">
        <h1>Skill Gap Analysis & Recommendation System</h1>
        <p>{APP_TITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="step-title">1. Select Career Context</div>', unsafe_allow_html=True)

domain = st.selectbox("Industry / IT Domain", list(DOMAIN_TO_TITLES.keys()))
job_title = st.selectbox("Job Title", DOMAIN_TO_TITLES[domain])
skill_level = st.selectbox(
    "Skill Level",
    ["Beginner", "Intermediate", "Advanced"],
)

st.markdown('<div class="step-title">2. Agentic AI Search</div>', unsafe_allow_html=True)

query = st.text_input(
    "Ask the system about a job, level, or skill",
    placeholder="Example: What skills are required for an intermediate backend developer?",
)

run = st.button("Search & Analyze", type="primary", use_container_width=True)

if run:
    resolved_domain, resolved_title, resolved_level = parse_query(
        query,
        domain,
        job_title,
        skill_level,
    )

    rows = role_rows(industry, resolved_title, resolved_level)
    rows, retrieval_method = semantic_retrieve(query, rows)
    req = required_skills(rows)
    gap_report, semantic_method = analyze_gaps(req, curriculum, resolved_level)

    st.divider()

    st.markdown('<div class="step-title">3. AI Interpretation</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric("IT Domain", resolved_domain)
    b.metric("Job Title", resolved_title)

    c, d = st.columns(2)
    c.metric("Level", resolved_level)
    d.metric("Matched Job Records", len(rows))

    st.markdown(
        f"""
        <div class="agent-box">
        <b>Agent workflow:</b>
        interpreted the user context → selected role and level → retrieved matching records
        from Dataset 02 → removed clearly role-incompatible synthetic combinations →
        extracted required skills → compared them with Dataset 01 → evaluated semantic
        alignment and curriculum level adequacy → generated technical training recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-title">4. Required Skills</div>', unsafe_allow_html=True)

    if req.empty:
        st.warning("No relevant records were found for the selected profile.")
    else:
        st.markdown(skill_chips(req), unsafe_allow_html=True)

        st.markdown('<div class="step-title">5. Skill Gap Analysis</div>', unsafe_allow_html=True)

        fully = int((gap_report["Gap Status"] == "Fully Matched").sum())
        partial = int((gap_report["Gap Status"] == "Partially Matched").sum())
        missing = int((gap_report["Gap Status"] == "Missing / Low Alignment").sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Fully Matched", fully)
        m2.metric("Partially Matched", partial)
        m3.metric("Missing / Low Alignment", missing)

        # Pure HTML/CSS chart (does not depend on Altair).
        # This is more robust on Streamlit Cloud / newer Python runtimes.
        max_count = max(fully, partial, missing, 1)
        chart_items = [
            ("Fully Matched", fully),
            ("Partially Matched", partial),
            ("Missing / Low Alignment", missing),
        ]
        chart_html = '<div class="gap-chart">'
        for label, value in chart_items:
            width = max(2, int((value / max_count) * 100)) if value > 0 else 0
            chart_html += f"""
            <div class="gap-row">
                <div class="gap-label">{label}</div>
                <div class="gap-track">
                    <div class="gap-fill" style="width:{width}%"></div>
                </div>
                <div class="gap-value">{value}</div>
            </div>
            """
        chart_html += '</div>'
        st.markdown(chart_html, unsafe_allow_html=True)

        st.dataframe(
            gap_report[[
                "Required Skill",
                "Industry Frequency",
                "Best Curriculum Match",
                "Curriculum Max Level",
                "Semantic Similarity",
                "Level Adequacy",
                "Alignment Score",
                "Gap Status",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="step-title">6. Recommended Technical Training</div>', unsafe_allow_html=True)

        gaps = gap_report[gap_report["Gap Status"] != "Fully Matched"].copy()

        if gaps.empty:
            st.success(
                "All identified required skills are adequately covered for the selected profile. "
                "No additional training is required; continue practical reinforcement and periodic curriculum review."
            )
        else:
            st.dataframe(
                gaps[[
                    "Required Skill",
                    "Gap Status",
                    "Recommended Training",
                    "Priority",
                    "Recommendation",
                ]],
                use_container_width=True,
                hide_index=True,
            )

            high = gaps[gaps["Priority"] == "High"]
            medium = gaps[gaps["Priority"] == "Medium"]

            if not high.empty:
                st.error(
                    "High-priority training recommendation: "
                    + ", ".join(high["Required Skill"].tolist())
                )

            if not medium.empty:
                st.warning(
                    "Medium-priority strengthening area: "
                    + ", ".join(medium["Required Skill"].tolist())
                )

        with st.expander("Show all skills, including adequately covered skills"):
            st.dataframe(
                gap_report[[
                    "Required Skill",
                    "Gap Status",
                    "Recommended Training",
                    "Priority",
                    "Recommendation",
                ]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="step-title">7. Dataset Evidence</div>', unsafe_allow_html=True)

        with st.expander("Industry evidence — Dataset 02"):
            cols = [
                "Job_ID",
                "Job_Title",
                "Job_Level",
                "Required_Skill",
                "Skill_Category",
                "Preferred_Skill",
                "Experience",
                "Education",
            ]
            st.dataframe(rows[cols], use_container_width=True, hide_index=True)

        with st.expander("Curriculum evidence — Dataset 01"):
            gap_courses = gaps["Recommended Training"].dropna().unique().tolist()
            gap_courses = [x for x in gap_courses if x != "No Additional Training Required"]

            if gap_courses:
                evidence = curriculum[curriculum["Course_Name"].isin(gap_courses)][[
                    "Curriculum_ID",
                    "Course_Name",
                    "Module_Name",
                    "Skill_Name",
                    "Skill_Category",
                    "Skill_Level",
                    "Theory_Hours",
                    "Practical_Hours",
                    "Assessment_Method",
                ]]
                st.dataframe(evidence.head(80), use_container_width=True, hide_index=True)
            else:
                st.info("No additional training course evidence is required because no skill gap was identified.")

        with st.expander("Methods and thresholds"):
            st.write(f"Retrieval method: {retrieval_method}")
            st.write(f"Semantic matching method: {semantic_method}")
            st.write("Alignment score = 75% semantic similarity + 25% curriculum level adequacy.")
            st.write(f"Fully Matched threshold: ≥ {FULL_MATCH_THRESHOLD}")
            st.write(f"Partially Matched threshold: {PARTIAL_MATCH_THRESHOLD} to < {FULL_MATCH_THRESHOLD}")
            st.write(f"Missing / Low Alignment threshold: < {PARTIAL_MATCH_THRESHOLD}")
            st.write("Training recommendations are selected only from course names present in Dataset 01.")

        st.caption(
            "Version 3: recommendations are shown prominently only for partially matched or "
            "missing skills. Fully matched skills are treated as adequately covered and do not "
            "receive unrelated training recommendations."
        )

else:
    st.info(
        "Select an IT domain, job title and skill level. Optionally enter a natural-language "
        "query, then click Search & Analyze."
    )
