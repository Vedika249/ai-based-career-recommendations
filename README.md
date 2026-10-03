# AI-Based Career Recommendation System

A Streamlit app that recommends careers based on a user's academic scores
and stated interests, using a **pretrained** Sentence-BERT model — no
training data or model training required.

**Project stage: Week 6 prototype.** The core recommendation pipeline
(interest matching + academic-fit scoring) is functional. Larger features
planned for later months — expanded knowledge base, resume upload,
authentication, cloud deployment — are intentionally not included yet
(see the project Work Plan).

## How it works

1. **Interest matching (Deep Learning / NLP)**
   The user's free-text interest description is converted into a semantic
   vector using the pretrained `all-MiniLM-L6-v2` Sentence-BERT model. The
   same model is used (offline, at startup) to convert each career
   description in `career_data.py` into vectors. Cosine similarity between
   the user vector and each career vector gives an "interest match" score.

2. **Academic fit (Rule-based layer)**
   Each career profile lists its typical key subjects and score thresholds.
   The user's entered marks are compared against these to produce a
   transparent "academic fit" score.

3. **Final ranking**
   `final_score = 0.65 * interest_match + 0.35 * academic_fit`
   Careers are ranked by this combined score and the top N are shown with
   a bar chart and per-career breakdown.

## Interface

The UI uses a custom-styled layout (gradient header, card-based sections,
color-coded stream badges, medal icons for top 3 matches) built with
Streamlit + injected CSS — no extra frontend framework required.

## Files

- `app.py` — Streamlit frontend + recommendation logic (with custom styling)
- `career_data.py` — Career knowledge base (10 draft profiles — name,
  description, subject requirements) used for semantic matching
- `requirements.txt` — Python dependencies

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run will download the pretrained SBERT model (~80MB) from
Hugging Face — an internet connection is required for that one-time
download. After that, it runs fully offline.

## Roadmap (per project Work Plan)

- **Weeks 1–6 (current):** Requirement gathering, literature review,
  architecture/UML design, and this functional prototype with a 10-profile
  draft knowledge base (5 Science, 2 Commerce, 3 Arts).
- **Month 2 onward:** Expand knowledge base to 35+ profiles, add resume/PDF
  upload with NER-based interest extraction, testing with real users, and
  deployment to a public cloud platform.

## Extending this project

- Add more careers to `career_data.py` to broaden coverage.
- Swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` for higher accuracy
  (slower, larger model) — one-line change in `app.py`.
- Add resume/PDF upload and extract interests automatically using spaCy
  NER instead of manual text entry.
- Log user inputs + chosen career (if they pick one) to build a real
  dataset over time — enabling a future supervised model.