import streamlit as st
import pdfplumber
import re
from openai import OpenAI
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
	api_key=OPENAI_API_KEY,
	base_url="https://api.groq.com/openai/v1",  # ✅ Required for Groq
)

st.set_page_config(page_title="AI Resume Enhancer", layout="centered")
st.title("📄 AI Resume Enhancer")

uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description", height=200)

def extract_keywords(text):
    return set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9+\-#\.]{1,}\b', text.lower()))

if st.button("Analyze Resume"):
    if uploaded_resume and job_description:
        with pdfplumber.open(uploaded_resume) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text()

        # 🔍 Extract keywords
        jd_keywords = extract_keywords(job_description)
        resume_keywords = extract_keywords(resume_text)

        # ✅ Calculate match score
        matched_keywords = jd_keywords & resume_keywords
        match_score = int(len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords else 0

        # 📊 Show match score
        st.subheader("📈 Match Score:")
        st.progress(match_score / 100)
        st.write(f"**{match_score}% match** between your resume and the job description.")

        # 🤖 GPT Feedback
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

