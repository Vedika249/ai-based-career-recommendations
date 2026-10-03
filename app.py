"""
AI-Based Career Recommendation System
--------------------------------------
Approach: Pretrained SBERT (Sentence-BERT) embeddings for semantic matching
between the user's stated interests and a curated career knowledge base,
combined with a transparent rule-based academic-fit score derived from the
user's subject-wise marks.

No model training or labeled dataset is required — all "learning" comes
from a pretrained transformer (all-MiniLM-L6-v2) used purely for inference.

Project stage: Week 6 prototype — core recommendation pipeline (SBERT
interest matching + rule-based academic fit) is functional. Larger features
planned for later months (resume upload, authentication, deployment) are
intentionally NOT included yet — see Work Plan in the project report.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util

from career_data import CAREERS

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom styling (CSS injected once)
# ----------------------------------------------------------------------
STREAM_COLORS = {
    "Science": {"bg": "#E8F1FB", "text": "#1E5C97", "badge": "#4A90D9"},
    "Commerce": {"bg": "#FFF3E8", "text": "#B5591A", "badge": "#DD8047"},
    "Arts": {"bg": "#F1EEFB", "text": "#5B3E9E", "badge": "#8B6BC7"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, .hero-title {
    font-family: 'Poppins', sans-serif;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #5C86A8 0%, #7BA79D 100%);
    padding: 2.2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 24px rgba(92, 134, 168, 0.25);
}
.hero-title {
    color: white;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    color: rgba(255,255,255,0.92);
    font-size: 1.02rem;
    font-weight: 400;
    line-height: 1.5;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.8rem;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Section cards */
.section-card {
    background: #FFFFFF;
    border: 1px solid #EAEAEA;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.section-heading {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: #3A332E;
    margin-bottom: 0.6rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #DD8047 0%, #C96A30 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.6rem;
    font-size: 1rem;
    box-shadow: 0 4px 14px rgba(221, 128, 71, 0.35);
    transition: transform 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(221, 128, 71, 0.45);
    color: white;
}

/* Recommendation cards */
.rec-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #EDEDED;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.rec-rank {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #3A332E;
}
.stream-badge {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
.match-score {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
}
.desc-text {
    color: #5A5450;
    font-size: 0.92rem;
    line-height: 1.5;
    margin: 0.6rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FAF8F5;
}
.sidebar-badge {
    background: #EBDDC3;
    color: #775F55;
    padding: 0.5rem 0.8rem;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 About")
    st.markdown(
        '<div class="sidebar-badge">📍 Project Stage: Week 6 Prototype</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This tool combines a **pretrained NLP model (SBERT)** with a "
        "**transparent rule-based scoring layer** to recommend careers — "
        "no labeled training dataset required."
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Your interests → semantic vector (SBERT)\n"
        "2. Compared against career descriptions (cosine similarity)\n"
        "3. Combined with your academic-fit score\n"
        "4. Ranked recommendations shown"
    )
    st.markdown("---")
    st.caption(f"Career knowledge base: **{len(CAREERS)} draft profiles** across Science, Commerce & Arts streams.")
    st.caption("Full 35+ profile base planned for Month 2 (see project Work Plan).")
    st.caption("Model: `all-MiniLM-L6-v2` (pretrained, inference-only)")

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🎯 AI-Based Career Recommendation Platform</div>
    <div class="hero-subtitle">
        Get personalized career suggestions powered by a pretrained language model —
        matched to both your academic strengths and your personal interests.
    </div>
    <div class="hero-badge">✨ No training data needed — pretrained AI + transparent scoring</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load pretrained model (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pretrained language model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def embed_career_descriptions(_model):
    descriptions = [c["description"] for c in CAREERS]
    embeddings = _model.encode(descriptions, convert_to_tensor=True)
    return embeddings


model = load_model()
career_embeddings = embed_career_descriptions(model)

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
SUBJECTS = ["Math", "Physics", "Chemistry", "Biology", "Computer Science", "English"]

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Academic Scores</div>', unsafe_allow_html=True)
    st.caption("Enter your percentage marks (0–100). Leave subjects you haven't studied at 0.")
    scores = {}
    score_cols = st.columns(2)
    for i, subject in enumerate(SUBJECTS):
        with score_cols[i % 2]:
            scores[subject] = st.slider(subject, 0, 100, 0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">💡 Your Interests</div>', unsafe_allow_html=True)
    st.caption("Describe what you enjoy — subjects, hobbies, or the kind of work that excites you.")
    interest_text = st.text_area(
        "Interests",
        height=160,
        placeholder="e.g. I enjoy solving puzzles, working with numbers, and building things with computers...",
        label_visibility="collapsed",
    )
    top_n = st.slider("Number of suggestions to show", 3, 10, 5)
    st.markdown('</div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 1.2, 1])
with btn_col:
    submit = st.button("🔍  Get Career Recommendations", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Recommendation logic
# ----------------------------------------------------------------------
def academic_fit_score(career, scores):
    """
    Returns a 0-1 score representing how well the user's academic scores
    align with a career's key subjects. Careers with no key subject
    requirements get a neutral score of 0.6 so they aren't unfairly
    penalized against interest matching.
    """
    key_subjects = career["key_subjects"]
    min_scores = career["min_scores"]

    if not key_subjects:
        return 0.6

    fit_values = []
    for subject in key_subjects:
        user_score = scores.get(subject, 0)
        threshold = min_scores.get(subject, 50)
        if user_score <= 0:
            fit_values.append(0.4)
        else:
            fit = min(user_score / threshold, 1.2) if threshold > 0 else 1.0
            fit_values.append(min(fit, 1.2) / 1.2)

    return float(np.mean(fit_values))


def get_recommendations(interest_text, scores, top_n=5, interest_weight=0.65):
    if not interest_text.strip():
        interest_text = "no specific interest provided"

    user_embedding = model.encode(interest_text, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, career_embeddings)[0].cpu().numpy()

    results = []
    for i, career in enumerate(CAREERS):
        sim_score = float(similarities[i])
        sim_score_norm = max(sim_score, 0)
        fit_score = academic_fit_score(career, scores)
        final_score = (interest_weight * sim_score_norm) + ((1 - interest_weight) * fit_score)

        results.append({
            "name": career["name"],
            "stream": career["stream"],
            "description": career["description"],
            "similarity": sim_score_norm,
            "academic_fit": fit_score,
            "final_score": final_score,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_n]


# ----------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------
RANK_MEDALS = ["🥇", "🥈", "🥉"]

if submit:
    if not interest_text.strip() and all(v == 0 for v in scores.values()):
        st.warning("Please enter at least your interests or some academic scores.")
    else:
        with st.spinner("Analyzing your profile..."):
            recommendations = get_recommendations(interest_text, scores, top_n=top_n)

        st.markdown("### ✨ Your Top Career Matches")

        chart_data = {
            "Career": [r["name"] for r in recommendations],
            "Match Score (%)": [round(r["final_score"] * 100, 1) for r in recommendations],
        }
        st.bar_chart(data=chart_data, x="Career", y="Match Score (%)", horizontal=True, color="#5C86A8")

        st.markdown("")

        for rank, rec in enumerate(recommendations, start=1):
            colors = STREAM_COLORS.get(rec["stream"], STREAM_COLORS["Science"])
            medal = RANK_MEDALS[rank - 1] if rank <= 3 else f"#{rank}"

            st.markdown(f"""
            <div class="rec-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span class="rec-rank">{medal} {rec['name']}</span>
                        <span class="stream-badge" style="background:{colors['bg']}; color:{colors['text']};">
                            {rec['stream']}
                        </span>
                    </div>
                    <div class="match-score" style="color:{colors['badge']};">{rec['final_score']*100:.0f}%</div>
                </div>
                <div class="desc-text">{rec['description']}</div>
            </div>
            """, unsafe_allow_html=True)

            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.progress(rec["similarity"], text=f"💡 Interest match: {rec['similarity']*100:.0f}%")
            with detail_col2:
                st.progress(rec["academic_fit"], text=f"📊 Academic fit: {rec['academic_fit']*100:.0f}%")
            st.markdown("")

        with st.expander("ℹ️ How this recommendation was generated"):
            st.write(
                "1. Your interest text was converted into a numeric vector using a "
                "pretrained transformer model (SBERT - all-MiniLM-L6-v2).\n\n"
                "2. Each career's description was already converted into a vector "
                "the same way.\n\n"
                "3. Cosine similarity was computed between your interest vector and "
                "every career vector, giving an 'interest match' score.\n\n"
                "4. Your academic scores were compared against each career's typical "
                "subject requirements to compute an 'academic fit' score.\n\n"
                "5. Both scores were combined (65% interest, 35% academic fit) to "
                "produce the final ranking."
            )
else:
    st.info("👈 Fill in the form above and click **Get Career Recommendations** to see results.")

st.markdown("---")
st.caption(
    "Built with a pretrained Sentence-BERT model — no labeled training dataset required. "
    "Academic + Interest hybrid scoring is fully explainable and rule-transparent. "
    "· Week 6 prototype stage"
)