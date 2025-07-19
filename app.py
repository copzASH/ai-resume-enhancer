import streamlit as st
import pdfplumber
import re
from openai import OpenAI
import plotly.graph_objects as go

# Setup OpenAI (Groq) client
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Page config
st.set_page_config(page_title="AI Resume Enhancer", layout="centered")
st.title("📄 AI Resume Enhancer")
st.markdown("Enhance your resume to better match job descriptions using AI + keyword matching.")

# Upload Resume
uploaded_resume = st.file_uploader("📎 Upload Your Resume (PDF, <10MB)", type=["pdf"])

if uploaded_resume is not None and uploaded_resume.size > 10 * 1024 * 1024:
    st.error("File too large. Please upload a PDF smaller than 10MB.")
    uploaded_resume = None

# Job Description input
job_description = st.text_area("💼 Paste the Job Description", height=200)

# --- Helper Functions ---
def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+\-#\.]{1,}\b', text.lower())
    stopwords = {"the", "and", "for", "to", "with", "a", "in", "on", "of", "at", "by", "an", "is", "it", "as", "this", "that", "from"}
    return set(word for word in words if word not in stopwords and len(word) > 2)

def extract_sections(text):
    section_titles = ['education', 'experience', 'projects', 'skills', 'certifications', 'achievements', 'summary', 'objective']
    sections = {}
    current_section = 'General'
    sections[current_section] = []

    for line in text.splitlines():
        line_strip = line.strip().lower()
        matched = [title for title in section_titles if title in line_strip]
        if matched:
            current_section = matched[0].capitalize()
            sections[current_section] = []
        sections[current_section].append(line.strip())

    return {k: "\n".join(v).strip() for k, v in sections.items()}

def analyze_resume(resume_text, job_description):
    jd_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)

    matched = jd_keywords & resume_keywords
    unmatched = jd_keywords - resume_keywords
    score = int(len(matched) / len(jd_keywords) * 100) if jd_keywords else 0

    # GPT Feedback
    prompt = f"""
You are a professional resume reviewer. Analyze the following resume in comparison to the job description. 
Suggest improvements and identify missing skills or keywords.

Resume:
{resume_text}

Job Description:
{job_description}

Provide feedback as clear bullet points.
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    feedback = response.choices[0].message.content

    return score, matched, unmatched, feedback

# --- Button: Analyze Resume ---
if st.button("✨ Analyze Resume"):
    if not uploaded_resume or not job_description.strip():
        st.warning("Please upload a resume and enter a job description.")
    else:
        with st.spinner("🔍 Extracting resume content..."):
            try:
                with pdfplumber.open(uploaded_resume) as pdf:
                    resume_text = ""
                    for page in pdf.pages:
                        resume_text += page.extract_text() or ""

                if not resume_text.strip():
                    st.error("Couldn't extract any text from the uploaded PDF.")
                else:
                    sections = extract_sections(resume_text)
                    score, matched, unmatched, feedback = analyze_resume(resume_text, job_description)

                    # ✅ Score Progress
                    st.progress(score / 100)
                    st.write(f"✅ **{score}% match** with the job description.")

                    # ✅ Bar Chart
                    if matched or unmatched:
                        chart = go.Figure(data=[
                            go.Bar(x=["Matched", "Unmatched"], y=[len(matched), len(unmatched)],
                                   marker_color=["green", "red"])
                        ])
                        chart.update_layout(
                            xaxis_title="Keyword Type",
                            yaxis_title="Count",
                            title="Keyword Coverage"
                        )
                        st.plotly_chart(chart)

                    # ✅ Section-wise Text
                    st.subheader("📂 Extracted Resume Sections")
                    for title, content in sections.items():
                        with st.expander(f"📌 {title}"):
                            st.text(content)

                    # ✅ AI Suggestions
                    st.subheader("🧠 GPT Suggestions")
                    st.markdown(feedback)

            except Exception as e:
                st.error(f"An error occurred: {e}")

