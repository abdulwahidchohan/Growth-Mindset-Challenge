import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import altair as alt  # type: ignore
from datetime import datetime
import base64
from fpdf import FPDF
import json
import requests  # type: ignore
import sqlite3
import re
import hashlib
from streamlit_lottie import st_lottie  # type: ignore
from PIL import Image  # type: ignore
import os
from dotenv import load_dotenv  # type: ignore
import logging

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(
    page_title="Enterprise Survey Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Database Initialization
def init_db():
    conn = sqlite3.connect('survey.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS responses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  full_name TEXT,
                  email TEXT,
                  age INTEGER,
                  country TEXT,
                  interests TEXT,
                  satisfaction INTEGER,
                  ai_interest TEXT,
                  improvement_feedback TEXT,
                  submission_date TIMESTAMP,
                  consent BOOLEAN)''')
    conn.commit()
    conn.close()

init_db()

# Security Functions
def hash_email(email):
    return hashlib.sha256(email.encode()).hexdigest()

def validate_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    return re.fullmatch(regex, email)

# PDF Generation Class
class PDFReport(FPDF):
    def header(self):
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 25)  # Add company logo
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Survey Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Load Image with Error Handling
def load_image(image_path):
    """Load an image with error handling."""
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return None
    try:
        return Image.open(image_path)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None

# Load Lottie Animation from URL
def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Main App
def main():
    # Session State Initialization
    if 'survey_data' not in st.session_state:
        st.session_state.survey_data = {}
    if 'consent_given' not in st.session_state:
        st.session_state.consent_given = False

    # GDPR Consent Banner
    if not st.session_state.consent_given:
        with st.container():
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
                <h3>Data Collection Consent</h3>
                <p>We use cookies to improve your experience. By continuing you agree to our 
                <a href='#'>Privacy Policy</a> and data collection practices.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Accept"):
                    st.session_state.consent_given = True
            with col2:
                if st.button("Decline"):
                    st.stop()

    # Main Survey Content
    st.title("📈 Enterprise Demographic Survey")
    
    # Progress Bar
    progress = st.session_state.get('progress', 0)
    st.progress(progress)
    
    # Step 1: Basic Information
    if progress < 0.33:
        with st.form("basic_info"):
            st.header("Basic Information")
            email = st.text_input("Work Email*")
            if email and not validate_email(email):
                st.error("Please enter a valid work email address")
            
            st.session_state.survey_data['full_name'] = st.text_input("Full Name*")
            st.session_state.survey_data['age'] = st.slider("Age*", 18, 100)
            st.session_state.survey_data['company'] = st.text_input("Company Name*")
            
            if st.form_submit_button("Next"):
                if all([st.session_state.survey_data.get(k) for k in ['full_name', 'age', 'company']]):
                    st.session_state.survey_data['email'] = email  # Save email input
                    st.session_state.progress = 0.33
                    st.experimental_rerun()
                else:
                    st.error("Please fill all required fields")

    # Step 2: Professional Details
    if 0.33 <= progress < 0.66:
        with st.form("professional_details"):
            st.header("Professional Details")
            st.session_state.survey_data['industry'] = st.selectbox(
                "Industry*",
                ["Technology", "Finance", "Healthcare", "Education", "Other"]
            )
            st.session_state.survey_data['experience'] = st.slider(
                "Years of Experience*", 0, 50
            )
            st.session_state.survey_data['skills'] = st.multiselect(
                "Technical Skills",
                ["Python", "SQL", "Machine Learning", "Data Analysis", "Cloud Computing"]
            )
            
            if st.form_submit_button("Next"):
                st.session_state.progress = 0.66
                st.experimental_rerun()

    # Step 3: Final Review & Submission
    if progress >= 0.66:
        st.header("Review & Submit")
        
        # Data Review Table
        review_df = pd.DataFrame.from_dict(
            st.session_state.survey_data, 
            orient='index', 
            columns=['Responses']
        )
        st.table(review_df)
        
        # Terms of Service
        terms = st.checkbox("I agree to the Terms of Service and Data Processing Agreement")
        
        if terms and st.button("Submit Survey"):
            # Save to database
            conn = sqlite3.connect('survey.db')
            c = conn.cursor()
            c.execute('''INSERT INTO responses 
                      (full_name, email, age, country, interests, submission_date, consent)
                      VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (st.session_state.survey_data['full_name'],
                       hash_email(st.session_state.survey_data['email']),
                       st.session_state.survey_data['age'],
                       st.session_state.survey_data.get('country', ''),
                       ','.join(st.session_state.survey_data.get('interests', [])),
                       datetime.now(),
                       True))
            conn.commit()
            conn.close()
            
            # Generate PDF Report
            pdf = PDFReport()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Professional Survey Report", ln=True, align='C')
            pdf.ln(10)
            
            for key, value in st.session_state.survey_data.items():
                pdf.cell(0, 10, f"{key.title()}: {value}", ln=True)
            
            report_bytes = pdf.output(dest="S").encode("latin1")
            base64_pdf = base64.b64encode(report_bytes).decode()
            
            # Success Message
            st.success("Submission Successful!")
            st_lottie(load_lottie("https://assets9.lottiefiles.com/packages/lf20_sk5h1kfn.json"), height=200)
            
            # Download Button
            st.markdown(f'''
            <a href="data:application/pdf;base64,{base64_pdf}" download="professional_survey_report.pdf">
                <button style="background-color: #4CAF50; color: white; padding: 14px 20px; border: none; border-radius: 4px;">
                    Download Professional Report
                </button>
            </a>
            ''', unsafe_allow_html=True)

# Entry Point
if __name__ == "__main__":
    main()
