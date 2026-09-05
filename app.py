import os
import json
import io
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & MODERN STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Matcher & ATS Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, Modern Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color); /* Adapts automatically to Light & Dark mode */
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.75; /* Softened secondary color for both themes */
        margin-bottom: 2rem;
    }
    
    .stCard {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    
    .metric-score {
        font-size: 3rem;
        font-weight: 800;
    }
    
    .score-high { color: #10b981; }
    .score-med { color: #f59e0b; }
    .score-low { color: #ef4444; }
    
    div[data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR FILE EXTRACTION
# -----------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes) -> str:
    """Extract text from PDF file bytes."""
    try:
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return ""

def extract_text_from_docx(file_bytes) -> str:
    """Extract text from DOCX file bytes."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        extracted_text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n".join(extracted_text)
    except Exception as e:
        st.error(f"Error parsing DOCX document: {str(e)}")
        return ""

# -----------------------------------------------------------------------------
# GEMINI AI ANALYSIS FUNCTION
# -----------------------------------------------------------------------------
def analyze_resume_with_gemini(api_key: str, resume_content, file_type: str, job_role: str, job_desc: str):
    """
    Sends file or text content along with job details to Gemini 2.0 Flash.
    Returns structured JSON with ATS score, match verdict, and recommendations.
    """
    client = genai.Client(api_key=api_key)
    
    system_instruction = (
        "You are an expert AI Resume Reviewer and Applicant Tracking System (ATS) Specialist. "
        "Analyze the provided candidate resume against the Target Job Role and Job Description. "
        "Be rigorous, realistic, and objective in your evaluation."
    )
    
    prompt = f"""
Target Job Role: {job_role if job_role else 'Not specified'}
Job Description: {job_desc if job_desc else 'Not specified'}

Analyze the resume provided in the context and return your analysis STRICTLY in JSON format with the following keys:
1. "ats_score": An integer from 0 to 100 representing ATS match percentage.
2. "match_verdict": A short summary string (e.g., "Strong Match", "Moderate Match", or "Weak Match").
3. "role_match_summary": A concise paragraph explaining how well the candidate's skills align with the target role.
4. "matching_keywords": List of key skills/keywords present in both the resume and job description.
5. "missing_keywords": List of crucial missing skills or keywords from the job description.
6. "strengths": List of 3-4 key strengths identified in the resume.
7. "actionable_improvements": List of 4-5 specific, bulleted recommendations to optimize the resume for ATS and recruiters.
"""

    contents = []
    
    # Handle image-based inputs (PNG, JPEG) via direct Gemini Vision capability
    if file_type in ["png", "jpg", "jpeg"]:
        contents.append(resume_content)  # PIL Image instance
    else:
        contents.append(f"Resume Text Content:\n{resume_content}")
        
    contents.append(prompt)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.2
        )
    )
    
    return json.loads(response.text)

# -----------------------------------------------------------------------------
# APP UI & CONTROL FLOW
# -----------------------------------------------------------------------------
def main():
    # Header Section
    st.markdown('<div class="main-header">ATS Resume Matcher & Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Optimize your resume for ATS algorithms using Gemini 2.0 Flash AI.</div>', unsafe_allow_html=True)

    # Sidebar - API Key Management
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your API key from Google AI Studio.")
        st.markdown("---")
        st.markdown("**Supported Upload formats:**\n- PDF (`.pdf`)\n- Word Document (`.docx`)\n- Image Files (`.png`, `.jpg`, `.jpeg`)")

    # Main Layout: 2 Columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            help="Upload your existing resume."
        )

        st.subheader("2. Target Job Context")
        job_role = st.text_input("Job Role / Title", placeholder="e.g. Senior Frontend Developer")
        job_desc = st.text_area("Job Description", height=200, placeholder="Paste the full job description here...")

        analyze_btn = st.button("🚀 Analyze Match & Get ATS Score", use_container_width=True, type="primary")

    with col2:
        st.subheader("3. Analysis Results")
        
        if analyze_btn:
            if not api_key:
                st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to continue.")
                return
            if not uploaded_file:
                st.warning("⚠️ Please upload a resume file.")
                return

            with st.spinner("Analyzing resume against job requirements..."):
                file_ext = uploaded_file.name.split(".")[-1].lower()
                file_bytes = uploaded_file.read()
                
                resume_payload = None
                
                # Parse Document according to Extension
                if file_ext == "pdf":
                    resume_payload = extract_text_from_pdf(file_bytes)
                elif file_ext == "docx":
                    resume_payload = extract_text_from_docx(file_bytes)
                elif file_ext in ["png", "jpg", "jpeg"]:
                    resume_payload = Image.open(io.BytesIO(file_bytes))

                if not resume_payload and file_ext not in ["png", "jpg", "jpeg"]:
                    st.error("Failed to read text from the uploaded file. Ensure it is not empty or password protected.")
                    return

                try:
                    # Execute Gemini Analysis
                    result = analyze_resume_with_gemini(
                        api_key=api_key,
                        resume_content=resume_payload,
                        file_type=file_ext,
                        job_role=job_role,
                        job_desc=job_desc
                    )

                    # Display ATS Score Card
                    score = result.get("ats_score", 0)
                    score_class = "score-high" if score >= 80 else ("score-med" if score >= 60 else "score-low")
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <span style="font-weight: 600; color: #475569;">Overall ATS Match Score</span>
                        <span class="metric-score {score_class}">{score}%</span>
                        <span style="font-weight: 500; font-size: 1.1rem;">Verdict: {result.get('match_verdict', 'N/A')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Display Role Alignment
                    st.markdown("#### 🎯 Role Alignment Summary")
                    st.write(result.get("role_match_summary", "N/A"))

                    # Key Metrics Tabs
                    tab1, tab2, tab3 = st.tabs(["💡 Improvements", "🔑 Keywords", "⭐ Strengths"])

                    with tab1:
                        st.markdown("##### Actionable Recommendations to Improve Score:")
                        for item in result.get("actionable_improvements", []):
                            st.write(f"- {item}")

                    with tab2:
                        k_col1, k_col2 = st.columns(2)
                        with k_col1:
                            st.markdown("##### ✅ Matching Keywords")
                            for kw in result.get("matching_keywords", []):
                                st.success(kw)
                        with k_col2:
                            st.markdown("##### ❌ Missing Keywords")
                            for kw in result.get("missing_keywords", []):
                                st.error(kw)

                    with tab3:
                        st.markdown("##### Highlighted Strengths:")
                        for strength in result.get("strengths", []):
                            st.write(f"- {strength}")

                except Exception as e:
                    st.error(f"An error occurred during AI analysis: {str(e)}")
        else:
            st.info("Upload your resume, enter job details, and click **Analyze Match** to get instant feedback.")

if __name__ == "__main__":
    main()
