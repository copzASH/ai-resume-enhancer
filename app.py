import streamlit as st
import pdfplumber
import re
from openai import OpenAI
import plotly.graph_objects as go

# Setup OpenAI (Groq) client
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
	api_key=OPENAI_API_KEY,
	base_url="https://api.groq.com/openai/v1",  # ✅ Required for Groq
)

# Page config
st.set_page_config(page_title="AI Resume Enhancer", layout="centered")
st.title("📄 AI Resume Enhancer")
st.markdown("Enhance your resume to better match job descriptions using AI + keyword matching.")

# Upload Resume
uploaded_resume = st.file_uploader("📎 Upload Your Resume (PDF, <10MB)", type=["pdf"])

# File size check (10MB max)
if uploaded_resume is not None and uploaded_resume.size > 10 * 1024 * 1024:
    st.error("File too large. Please upload a PDF smaller than 10MB.")
    uploaded_resume = None

# Job Description input
job_description = st.text_area("💼 Paste the Job Description", height=200)

# Helper function: Extract keywords
def extract_keywords(text):
    return set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9+\-#\.]{1,}\b', text.lower()))

# Analyze Button
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
                    # Keyword matching
                    jd_keywords = extract_keywords(job_description)
                    resume_keywords = extract_keywords(resume_text)
                    matched = jd_keywords & resume_keywords
                    score = int(len(matched) / len(jd_keywords) * 100) if jd_keywords else 0

		    fig = go.Figure(go.Indicator(
		        mode="gauge+number",
		        value=score,
		        title={'text': "Resume Match Score"},
		        gauge={'axis': {'range': [0, 100]},
		    	   'bar': {'color': "green"},
		    	   'steps': [
		    	       {'range': [0, 50], 'color': "lightcoral"},
		    	       {'range': [50, 75], 'color': "khaki"},
		    	       {'range': [75, 100], 'color': "lightgreen"}]}
		    ))
		    st.plotly_chart(fig)
                    # Display Match Score
                    st.subheader("📈 Resume Match Score")
                    st.progress(score / 100)
                    st.write(f"✅ **{score}% match** with the job description.")
                    st.write(f"🔑 Matched Keywords: {', '.join(sorted(matched))}")
		    
                    # GPT Suggestions
                    prompt = f"""
You are a professional resume reviewer. Analyze the following resume in comparison to the job description. 
Suggest improvements and identify missing skills or keywords.

Resume:
{resume_text}

Job Description:
{job_description}

Provide feedback as clear bullet points.
"""
                    with st.spinner("🤖 AI is reviewing your resume..."):
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.6
                        )
                        feedback = response.choices[0].message.content

                    st.subheader("🧠 GPT Suggestions")
                    st.markdown(feedback)

            except Exception as e:
                st.error("An error occurred while analyzing the resume. Please check your input or try again.")

