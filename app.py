import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="MSHA Mine Safety Predictor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #ff7f0e;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('xgboost_injury_classifier.pkl')
        regressor = joblib.load('xgboost_days_lost_regressor.pkl')
        encoder = joblib.load('degree_injury_encoder.pkl')
        preprocessor = joblib.load('feature_preprocessor.pkl')
        return classifier, regressor, encoder, preprocessor
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None, None, None, None

classifier, regressor, encoder, preprocessor = load_models()

# Title
st.markdown('<p class="main-header">⛏️ MSHA Mine Safety Risk Predictor</p>', unsafe_allow_html=True)
st.markdown("### 🤖 AI-Powered Accident Severity & Lost Workdays Prediction")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/mining.png", width=100)
    st.markdown("## ℹ️ About")
    st.info(
        "This application uses **XGBoost Machine Learning** models trained on "
        "**200,000+ MSHA accident records** to predict:\n\n"
        "1. **Injury Severity** (Classification)\n"
        "2. **Lost Workdays** (Regression)\n\n"
        "Built with Streamlit & scikit-learn"
    )
    
    st.markdown("## 📊 Model Performance")
    st.success("**Classification Accuracy:** High precision for injury severity")
    st.success("**Regression R² Score:** Optimized for lost workdays prediction")
    
    st.markdown("---")
    st.markdown("### 🔒 Data Privacy")
    st.caption("All predictions are processed locally. No data is stored.")

# Create tabs
tab1, tab2 = st.tabs(["🎯 Make Prediction", "ℹ️ Model Info"])

with tab1:
    st.markdown('<p class="sub-header">Enter Accident Details</p>', unsafe_allow_html=True)
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏭 Mine Information")
        mine_id = st.number_input("Mine ID", min_value=1, value=100003, step=1, 
                                   help="Unique identifier for the mine")
        operator_id = st.number_input("Operator ID", min_value=1, value=13586, step=1,
                                       help="Operator identification number")
        coal_metal_ind = st.selectbox("Mine Type", ["C", "M"], 
                                       help="C = Coal, M = Metal/Non-metal",
                                       index=1)
        subunit = st.selectbox("Subunit", [
            "MILL OPERATION/PREPARATION PLANT",
            "UNDERGROUND",
            "STRIP/OPEN PIT",
            "OFFICE WORKERS AT MINE SITE",
            "INDEPENDENT SHOPS OR YARDS"
        ])
    
    with col2:
        st.markdown("#### 👷 Worker Experience")
        tot_exper = st.slider("Total Mining Experience (years)", 
                               min_value=0.0, max_value=50.0, value=5.0, step=0.5,
                               help="Total years of experience in mining")
        mine_exper = st.slider("Mine-Specific Experience (years)", 
                                min_value=0.0, max_value=50.0, value=3.0, step=0.5,
                                help="Years at this specific mine")
        job_exper = st.slider("Job-Specific Experience (years)", 
                               min_value=0.0, max_value=50.0, value=2.0, step=0.5,
                               help="Years in current job role")
        no_injuries = st.number_input("Previous Injuries", 
                                       min_value=0, max_value=20, value=1, step=1,
                                       help="Number of previous injuries")
    
    with col3:
        st.markdown("#### 🔧 Accident Details")
        occupation_cd = st.number_input("Occupation Code", 
                                         min_value=1, max_value=999, value=304, step=1,
                                         help="MSHA occupation code")
        accident_type = st.selectbox("Accident Type", [
            "SLIP OR FALL OF PERSON",
            "MACHINERY",
            "HANDLING MATERIALS",
            "HAND TOOLS, NOT POWERED",
            "STRIKING OR BUMPING",
            "POWERED HAULAGE",
            "FALLING, ROLLING, SLIDING ROCK OR MATERIAL OF ANY KIND",
            "EXPLODING VESSELS UNDER PRESSURE",
            "EXPLOSIVES AND BREAKING AGENTS"
        ])
        classification = st.selectbox("Classification", [
            "MACHINERY",
            "HANDLING MATERIALS",
            "SLIP OR FALL OF PERSON",
            "HAND TOOLS",
            "ELECTRICAL"
        ])
        days_restrict = st.number_input("Days of Restricted Work", 
                                         min_value=0, max_value=365, value=0, step=1,
                                         help="Days with restricted work activity")
        schedule_charge_text = st.selectbox("Schedule Charge", ["YES", "NO"],
                                             help="Whether schedule charge applies",
                                             index=1)
        schedule_charge = 1 if schedule_charge_text == "YES" else 0
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Risk & Lost Workdays", type="primary"):
        if classifier is not None and regressor is not None:
            # Prepare input data with all required features
            # Use current date/time for temporal features
            from datetime import datetime
            now = datetime.now()
            
            # Calculate experience level (numeric encoding)
            if tot_exper < 1:
                experience_level = 0  # Novice
            elif tot_exper < 5:
                experience_level = 1  # Intermediate
            elif tot_exper < 10:
                experience_level = 2  # Experienced
            else:
                experience_level = 3  # Expert
            
            # Calculate cyclical time features
            hour = now.hour
            day = now.day
            month = now.month
            
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 31)
            day_cos = np.cos(2 * np.pi * day / 31)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            input_data = pd.DataFrame([{
                'MINE_ID': mine_id,
                'OPERATOR_ID': operator_id,
                'OCCUPATION_CD': occupation_cd,
                'TOT_EXPER': tot_exper,
                'MINE_EXPER': mine_exper,
                'JOB_EXPER': job_exper,
                'NO_INJURIES': no_injuries,
                'ACCIDENT_TYPE': accident_type,
                'SUBUNIT': subunit,
                'CLASSIFICATION': classification,
                'COAL_METAL_IND': coal_metal_ind,
                'DAYS_RESTRICT': days_restrict,
                'SCHEDULE_CHARGE': schedule_charge,
                'Experience_Level': experience_level,
                'Hour_Sin': hour_sin,
                'Hour_Cos': hour_cos,
                'Day_Sin': day_sin,
                'Day_Cos': day_cos,
                'Month_Sin': month_sin,
                'Month_Cos': month_cos
            }])
            
            # Make predictions
            try:
                # Classification prediction
                injury_pred = classifier.predict(input_data)[0]
                injury_proba = classifier.predict_proba(input_data)[0]
                injury_label = encoder.inverse_transform([injury_pred])[0]
                
                # Regression prediction
                days_lost_pred = regressor.predict(input_data)[0]
                # Convert from log scale if needed
                days_lost_actual = np.expm1(days_lost_pred)  # inverse of log1p
                
                # Display results
                st.markdown("---")
                st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)
                
                # Main metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 🏥 Predicted Injury Severity")
                    st.markdown(f"## **{injury_label}**")
                    st.markdown(f"**Confidence:** {injury_proba.max():.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 📅 Predicted Lost Workdays")
                    st.markdown(f"## **{max(0, days_lost_actual):.1f} days**")
                    if days_lost_actual < 1:
                        st.markdown("**Category:** Minor incident")
                    elif days_lost_actual < 7:
                        st.markdown("**Category:** Moderate incident")
                    else:
                        st.markdown("**Category:** Serious incident")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("---")
                st.markdown("### 📊 Injury Severity Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Injury Severity': encoder.classes_,
                    'Probability': injury_proba
                }).sort_values('Probability', ascending=False)
                
                # Create horizontal bar chart
                fig = px.bar(prob_df, x='Probability', y='Injury Severity', 
                            orientation='h',
                            color='Probability',
                            color_continuous_scale='RdYlGn_r',
                            text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'))
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="Probability",
                    yaxis_title="",
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment
                st.markdown("---")
                st.markdown("### ⚠️ Risk Assessment")
                
                risk_score = (injury_proba.max() * 50 + min(days_lost_actual, 30))
                
                if risk_score < 20:
                    st.success("✅ **Low Risk**: Implement standard safety protocols")
                elif risk_score < 50:
                    st.warning("⚠️ **Moderate Risk**: Enhanced supervision recommended")
                else:
                    st.error("🚨 **High Risk**: Immediate safety intervention required")
                
                # Recommendations
                st.markdown("### 💡 Safety Recommendations")
                
                recommendations = []
                if tot_exper < 2:
                    recommendations.append("• Provide comprehensive training for inexperienced worker")
                if mine_exper < 1:
                    recommendations.append("• Assign experienced mentor for site-specific guidance")
                if no_injuries > 2:
                    recommendations.append("• Review safety compliance and work practices")
                if accident_type == "MACHINERY":
                    recommendations.append("• Ensure proper machine guarding and lockout/tagout procedures")
                if days_lost_actual > 7:
                    recommendations.append("• Conduct incident investigation and implement corrective actions")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("• Continue current safety practices and monitoring")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check that all model files are compatible with the input features.")

with tab2:
    st.markdown('<p class="sub-header">ℹ️ Model Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Classification Model")
        st.markdown("""
        **Algorithm:** XGBoost Classifier
        
        **Purpose:** Predicts the severity of workplace injuries
        
        **Output Classes:**
        - No Days Away From Work
        - Days Away From Work Only
        - Days of Restricted Work Activity
        - Permanent Total/Partial Disability
        - Fatality
        
        **Training Data:** 200,000+ MSHA accident records
        
        **Features Used:**
        - Worker experience metrics
        - Mine characteristics
        - Accident type and classification
        - Historical injury data
        """)
    
    with col2:
        st.markdown("### 📊 Regression Model")
        st.markdown("""
        **Algorithm:** XGBoost Regressor
        
        **Purpose:** Predicts the number of lost workdays
        
        **Output:** Continuous value (days)
        
        **Transformation:** Log-transformed for better accuracy
        
        **Training Data:** 200,000+ MSHA accident records
        
        **Features Used:**
        - Same features as classification model
        - Optimized for predicting work absence duration
        
        **Use Case:** Resource planning and compensation estimation
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Preprocessing:**
        - Target Encoding for high-cardinality features
        - One-Hot Encoding for categorical features
        - Standard Scaling for numerical features
        - Iterative Imputation for missing values
        """)
    
    with tech_col2:
        st.markdown("""
        **Model Features:**
        - Gradient Boosting with XGBoost
        - Hyperparameter optimization
        - Cross-validation for robustness
        - Feature importance analysis
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Data Source")
    st.info(
        "**MSHA (Mine Safety and Health Administration)** maintains comprehensive "
        "records of all mining accidents in the United States. This model is trained "
        "on historical accident data to help predict and prevent workplace injuries."
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "⛏️ MSHA Mine Safety Predictor | Built with Streamlit & XGBoost | "
    "© 2025 ML Project"
    "</p>",
    unsafe_allow_html=True
)
