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
st.markdown("Enhance your resume to better match job descriptions using AI + ATS scoring and keyword alignment.")

# Upload Resume
uploaded_resume = st.file_uploader("📎 Upload Your Resume (PDF, <10MB)", type=["pdf"])
if uploaded_resume is not None and uploaded_resume.size > 10 * 1024 * 1024:
    st.error("File too large. Please upload a PDF smaller than 10MB.")
    uploaded_resume = None

# Job Description input
job_description = st.text_area("💼 Paste the Job Description", height=200)

# --- AI Scoring Functions ---
def get_ai_score(prompt):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        content = response.choices[0].message.content
        match = re.search(r'\b(\d{1,3})\b', content)
        return int(match.group(1)) if match else 0
    except:
        return 0

def calculate_scores(resume_text, jd_text):
    resume_keywords = set(resume_text.lower().split())
    jd_keywords = set(jd_text.lower().split())
    common_keywords = resume_keywords.intersection(jd_keywords)
    keyword_score = int((len(common_keywords) / len(jd_keywords)) * 100) if jd_keywords else 0

    exp_prompt = f"Rate the relevance of this resume's experience to the following job description on a scale of 0 to 100:\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}"
    experience_score = get_ai_score(exp_prompt)

    skill_prompt = f"Rate the skill alignment of this resume to the job description from 0 to 100:\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}"
    skill_score = get_ai_score(skill_prompt)

    formatting_score = 100
    if len(resume_text.split()) > 1000:
        formatting_score -= 20
    if resume_text.count('.') / len(resume_text.split()) < 0.03:
        formatting_score -= 20

    overall = int(0.3 * keyword_score + 0.3 * experience_score + 0.2 * skill_score + 0.2 * formatting_score)
    return keyword_score, experience_score, skill_score, formatting_score, overall

def show_scorecard(resume_text, jd_text):
    st.subheader("📊 Resume Scorecard")
    kw, exp, skill, fmt, total = calculate_scores(resume_text, jd_text)

    categories = ['Keyword Match', 'Experience Relevance', 'Skill Alignment', 'Formatting Quality', 'Overall']
    values = [kw, exp, skill, fmt, total]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    ))
    fig.update_layout(
        yaxis=dict(title='Score (%)'),
        xaxis=dict(title='Metric'),
        height=400
    )
    st.plotly_chart(fig)

    with st.expander("Detailed Breakdown"):
        st.write(f"**Keyword Match Score:** {kw}%")
        st.progress(kw)
        st.write(f"**Experience Relevance Score:** {exp}%")
        st.progress(exp)
        st.write(f"**Skill Alignment Score:** {skill}%")
        st.progress(skill)
        st.write(f"**Formatting Quality Score:** {fmt}%")
        st.progress(fmt)
        st.markdown(f"### 🏁 Overall Score: `{total}%`")

# --- ATS Scoring ---
def calculate_ats_score(resume_text, job_description):
    score = 0
    details = []

    jd_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    matched_keywords = jd_keywords & resume_words
    keyword_score = min(len(matched_keywords) / len(jd_keywords), 1.0) * 40
    score += keyword_score
    details.append(f"✅ Keyword Match: {int(keyword_score)}/40")

    expected_sections = ['education', 'experience', 'projects', 'skills', 'contact']
    section_hits = sum(1 for s in expected_sections if s in resume_text.lower())
    section_score = (section_hits / len(expected_sections)) * 20
    score += section_score
    details.append(f"✅ Sections Present: {int(section_score)}/20")

    word_count = len(resume_text.split())
    if 300 <= word_count <= 1000:
        length_score = 20
        details.append("✅ Resume Length: 20/20")
    else:
        length_score = 10
        details.append("⚠️ Resume Length: 10/20 (Too short or long)")
    score += length_score

    issues = 0
    if re.search(r'(table|image)', resume_text, re.IGNORECASE):
        issues += 1
    if not re.search(r'[a-zA-Z]{4,}', resume_text):
        issues += 1
    format_score = max(0, 20 - (issues * 10))
    score += format_score
    if issues == 0:
        details.append("✅ Formatting: 20/20")
    else:
        details.append(f"⚠️ Formatting: {int(format_score)}/20 (Detected formatting issues)")

    return int(score), details

# --- ATS Compatibility ---
def check_ats_compatibility(resume_text):
    issues = []

    if re.search(r'(Table|Image)', resume_text, re.IGNORECASE):
        issues.append("❌ Detected words like 'Table' or 'Image'. ATS may not parse these properly.")

    word_count = len(resume_text.split())
    if word_count < 200:
        issues.append("❌ Resume is too short (<200 words). Consider adding more content.")
    elif word_count > 1200:
        issues.append("❌ Resume is too long (>1200 words). Try shortening it.")

    if not re.search(r'[a-zA-Z]{4,}', resume_text):
        issues.append("❌ Font issues or garbled text may exist.")

    expected_sections = ['education', 'experience', 'projects', 'skills']
    if not any(section in resume_text.lower() for section in expected_sections):
        issues.append("❌ Missing standard section headers (e.g., Education, Experience, Skills).")

    if not issues:
        return "✅ ATS Compatible", []
    return "❌ Not Fully ATS Compatible", issues

# --- Section Handlers ---
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

def get_section_feedback(section_name, section_text, job_description):
    prompt = f"""
You are an expert career coach. The following is the **{section_name}** section of a resume.

Section:
{section_text}

Compare it with the job description below and suggest clear, actionable improvements to this section. Focus on keyword inclusion, relevance, and clarity.

Job Description:
{job_description}

List the feedback as bullet points.
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return response.choices[0].message.content

def enhance_section_with_gpt(section_name, section_text, job_description):
    prompt = f"""
You are an expert resume writer. Improve the **{section_name}** section of this resume to better match the job description. 
Focus on improving clarity, relevance, and keyword alignment.

Section:
{section_text}

Job Description:
{job_description}

Return only the rewritten version.
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# --- Analyze Resume Button ---
if st.button("✨ Analyze Resume"):
    if not uploaded_resume or not job_description.strip():
        st.warning("Please upload a resume and enter a job description.")
    else:
        with st.spinner("🔍 Extracting resume content..."):
            try:
                with pdfplumber.open(uploaded_resume) as pdf:
                    resume_text = "".join(page.extract_text() or "" for page in pdf.pages)

                if not resume_text.strip():
                    st.error("Couldn't extract any text from the uploaded PDF.")
                else:
                    sections = extract_sections(resume_text)

                    show_scorecard(resume_text, job_description)

                    ats_status, ats_issues = check_ats_compatibility(resume_text)
                    st.subheader("🤖 ATS Compatibility Check")
                    st.markdown(f"**Status:** {ats_status}")
                    if ats_issues:
                        st.markdown("**Fix Suggestions:**")
                        for issue in ats_issues:
                            st.markdown(f"- {issue}")

                    ats_score, ats_details = calculate_ats_score(resume_text, job_description)
                    st.subheader("📊 ATS Score")
                    st.progress(ats_score)
                    st.markdown(f"**ATS Score: {ats_score}/100**")
                    for detail in ats_details:
                        st.markdown(f"- {detail}")
                        
                    kw_score, exp_score, skill_score, fmt_score, overall_score = calculate_scores(resume_text, job_description)

                    st.progress(score / 100)
                    st.success(f"✅ {score}% match with the job description.")

                    st.subheader("📂 Resume Sections with AI Suggestions")
                    for title, content in sections.items():
                        with st.expander(f"📌 {title}"):
                            st.markdown(f"**Raw Section Content:**\n\n```{content}```")
                            section_feedback = get_section_feedback(title, content, job_description)
                            st.markdown("**🧠 Suggestions:**")
                            st.markdown(section_feedback)

                    st.subheader("📌 Overall Resume Suggestions")
                    st.markdown(feedback)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Auto Enhance Resume ---
if st.button("🚀 Auto Enhance Resume"):
    if not uploaded_resume or not job_description.strip():
        st.warning("Please upload a resume and enter a job description.")
    else:
        with pdfplumber.open(uploaded_resume) as pdf:
            resume_text = "".join(page.extract_text() or "" for page in pdf.pages)
        sections = extract_sections(resume_text)
        st.subheader("✏️ Enhanced Resume (AI Rewritten Sections)")
        enhanced_sections = {}
        markdown_output = []

        for title, content in sections.items():
            enhanced = enhance_section_with_gpt(title, content, job_description)
            enhanced_sections[title] = enhanced

            with st.expander(f"🆚 {title} - Original vs Enhanced"):
                st.markdown(f"**📝 Original:**\n```{content}```")
                st.markdown(f"**✨ Enhanced:**\n```{enhanced}```")

            markdown_output.append(f"### {title}\n{enhanced}\n")

        full_enhanced_md = "\n".join(markdown_output)
        st.download_button("💾 Download Enhanced Resume (Markdown)", full_enhanced_md, file_name="enhanced_resume.md")

