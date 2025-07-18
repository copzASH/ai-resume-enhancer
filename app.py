import streamlit as st
import pdfplumber
import spacy
from openai import OpenAI
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# ----------------------------
# Setup
# ----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.set_page_config(page_title="AI Resume Enhancer", layout="centered")
st.title("📄 AI Resume Enhancer")

# OpenAI API (Groq) setup
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # ✅ For Groq
)

uploaded_resume = st.file_uploader("Upload your Resume (PDF, ≤10MB)", type=["pdf"])
job_description = st.text_area("Paste the Job Description", height=200)

# ----------------------------
# Keyword Extraction (Cleaned)
# ----------------------------
def extract_keywords(text, max_keywords=25):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation & lowercase
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    vectorizer = CountVectorizer(max_features=max_keywords)
    word_counts = vectorizer.fit_transform([' '.join(words)])
    keywords = vectorizer.get_feature_names_out()
    return set(keywords)

def get_keyword_analysis(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)

    matched_keywords = jd_keywords & resume_keywords
    missing_keywords = jd_keywords - resume_keywords

    return {
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "match_score": int(len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords else 0
    }

# ----------------------------
# Resume Analyzer
# ----------------------------
if st.button("Analyze Resume"):
    if uploaded_resume and job_description:
        if uploaded_resume.size > 10 * 1024 * 1024:
            st.error("Please upload a resume smaller than 10MB.")
        else:
            with pdfplumber.open(uploaded_resume) as pdf:
                resume_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

            analysis = get_keyword_analysis(resume_text, job_description)
            match_score = analysis["match_score"]

            st.subheader("📈 Match Score:")
            st.progress(match_score / 100)
            st.write(f"**{match_score}% match** between your resume and the job description.")

            with st.expander("✅ Matched Keywords"):
                st.write(", ".join(sorted(analysis["matched_keywords"])))
            with st.expander("⚠️ Missing Keywords"):
                st.write(", ".join(sorted(analysis["missing_keywords"])))

            prompt = f"""
You are an expert resume reviewer. Analyze the following resume against the job description. 
Identify matching skills, missing keywords, and suggest bullet point improvements. 
Resume:\n{resume_text}\n
Job Description:\n{job_description}\n
Give detailed feedback in points.
"""
            with st.spinner("Analyzing with GPT..."):
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6
                )
            st.subheader("🧠 GPT Feedback:")
            st.write(response.choices[0].message.content)
    else:
        st.warning("Please upload a resume and enter the job description.")

