import streamlit as st
import pdfplumber
import re
from openai import OpenAI

# ----------------------------
# Streamlit UI & API Setup
# ----------------------------
st.set_page_config(page_title="AI Resume Enhancer", layout="centered")
st.title("📄 AI Resume Enhancer")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

uploaded_resume = st.file_uploader("Upload your Resume (PDF, ≤10MB)", type=["pdf"])
job_description = st.text_area("Paste the Job Description", height=200)

# ----------------------------
# Pure‑Python Keyword + Phrase Extraction
# ----------------------------
def extract_phrases(text, min_len=4, max_n=3):
    # tokenize on words of length ≥ min_len
    tokens = re.findall(rf'\b\w{{{min_len},}}\b', text.lower())
    keywords = set(tokens)
    # build n‑grams up to max_n
    for n in range(2, max_n+1):
        for i in range(len(tokens)-n+1):
            phrase = " ".join(tokens[i:i+n])
            keywords.add(phrase)
    return keywords

def analyze_keywords(resume_text, jd_text):
    jd_phrases = extract_phrases(jd_text)
    resume_phrases = extract_phrases(resume_text)
    matched = jd_phrases & resume_phrases
    missing = jd_phrases - resume_phrases
    score = int(len(matched)/len(jd_phrases)*100) if jd_phrases else 0
    return matched, missing, score

# ----------------------------
# Main Logic
# ----------------------------
if st.button("Analyze Resume"):
    if not uploaded_resume or not job_description.strip():
        st.warning("Please upload a resume and enter the job description.")
    else:
        if uploaded_resume.size > 10 * 1024 * 1024:
            st.error("Please upload a PDF smaller than 10MB.")
        else:
            # Extract text
            with pdfplumber.open(uploaded_resume) as pdf:
                resume_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

            # Keyword/Phrase analysis
            matched, missing, score = analyze_keywords(resume_text, job_description)

            # Display results
            st.subheader("📈 Match Score")
            st.progress(score/100)
            st.write(f"**{score}%** match")

            with st.expander("✅ Matched Phrases"):
                st.write(", ".join(sorted(matched)) or "— none —")

            with st.expander("⚠️ Missing Phrases"):
                st.write(", ".join(sorted(missing)) or "— none —")

            # GPT feedback
            prompt = f"""
You are an expert resume reviewer. Given this resume and job description, 
1. Highlight what matches.
2. Point out what’s missing.
3. Suggest concise bullet‑point improvements.

Resume:
{resume_text}

Job Description:
{job_description}
"""
            with st.spinner("🤖 Generating AI feedback..."):
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6
                )
            st.subheader("🧠 AI Suggestions")
            st.markdown(response.choices[0].message.content)

