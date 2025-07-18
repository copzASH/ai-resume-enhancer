import streamlit as st
import pdfplumber
import spacy
import os
import re
import nltk
from nltk.corpus import stopwords
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer

# Setup
st.set_page_config(page_title="AI Resume Enhancer", layout="centered")
st.title("📄 AI Resume Enhancer")

# Preload NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load Spacy model
@st.cache_resource
def load_spacy_model():
    import en_core_web_sm
    nlp = en_core_web_sm.load()

nlp = load_spacy_model()

# OpenAI API (Groq) setup
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Upload & JD Input
uploaded_resume = st.file_uploader("Upload your Resume (PDF, ≤10MB)", type=["pdf"])
job_description = st.text_area("Paste the Job Description", height=200)

# ----------------------------
# Keyword Extraction
# ----------------------------
def extract_keywords(text, max_keywords=25):
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    vectorizer = CountVectorizer(max_features=max_keywords)
    word_counts = vectorizer.fit_transform([' '.join(words)])
    return set(vectorizer.get_feature_names_out())

def get_keyword_analysis(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    matched = jd_keywords & resume_keywords
    missing = jd_keywords - resume_keywords
    score = int(len(matched) / len(jd_keywords) * 100) if jd_keywords else 0
    return matched, missing, score

# ----------------------------
# Resume Analysis Logic
# ----------------------------
def analyze_resume(resume_text, jd):
    matched, missing, score = get_keyword_analysis(resume_text, jd)

    st.subheader("📈 Match Score:")
    st.progress(score / 100)
    st.write(f"**{score}% match** between your resume and the job description.")

    with st.expander("✅ Matched Keywords"):
        st.write(", ".join(sorted(matched)))
    with st.expander("⚠️ Missing Keywords"):
        st.write(", ".join(sorted(missing)))

    prompt = f"""
You are an expert resume reviewer. Analyze the resume below in the context of the job description.
Provide missing skills, matched terms, and improvements in bullet points.

Resume:
{resume_text}

Job Description:
{jd}
"""
    with st.spinner("Analyzing with LLaMA..."):
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        st.subheader("🧠 GPT Feedback:")
        st.write(response.choices[0].message.content)

# ----------------------------
# Trigger Analysis
# ----------------------------
if st.button("Analyze Resume"):
    if not uploaded_resume or not job_description:
        st.warning("Please upload your resume and paste the job description.")
    elif uploaded_resume.size > 10 * 1024 * 1024:
        st.error("Resume file must be smaller than 10MB.")
    else:
        with pdfplumber.open(uploaded_resume) as pdf:
            resume_text = "\n".join(
                [page.extract_text() for page in pdf.pages if page.extract_text()]
            )
        analyze_resume(resume_text, job_description)

