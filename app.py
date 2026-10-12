import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Performance & Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Attractive UI, Cards, and Shadows
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.4);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    return joblib.load('knn.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'knn.pkl': {e}")
    st.stop()

# Header Section
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎓 Student Performance Predictor</h1>
        <p style="color: #6c757d; font-size: 1.1rem;">Predict student outcomes using machine learning based on study habits and environment.</p>
    </div>
""", unsafe_allow_html=True)

# Main Form Layout inside a Card Container
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Enter Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course / Stream", ["Engineering", "Science", "Arts", "Commerce", "Other"])
        study_hours = st.slider("Study Hours (Daily)", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=85)
        
    with col2:
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        sleep_hours = st.slider("Sleep Hours (Daily)", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
        study_method = st.selectbox("Study Method", ["Self-Study", "Online Videos", "Group Study", "Coaching"])
        
    with col3:
        facility_rating = st.slider("Facility Rating (1-10)", min_value=1, max_value=10, value=7)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Moderate", "Hard"])
        internet_accesss = st.selectbox("Secondary Internet Access / Backup", ["Yes", "No"])
        
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Section
col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_btn = st.button("🚀 Predict Result")

if predict_btn:
    # Prepare input dataframe matching features in knn.pkl[cite: 1]
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty],
        'internet_accesss': [internet_accesss]
    })
    
    try:
        prediction = model.predict(input_data)
        st.markdown(f"""
            <div class="card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <h2 style="color: white; margin-bottom: 5px;">Prediction Result</h2>
                <p style="font-size: 2rem; font-weight: bold; margin-top: 0;">{prediction[0]:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
