#Aqui el codigo de streamlitstreamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import predict_diabetes, FEATURE_EXPLANATIONS

# Configure page
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .result-box { 
        border-radius:10px; 
        padding:20px; 
        margin-top:20px;
        border:1px solid #eee;
        background-color:#f9f9f9;
    }
    .positive { color: #e74c3c; font-weight:bold; }
    .negative { color: #2ecc71; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# App header
st.title("Diabetes Risk Assessment Tool")
st.markdown("""
This tool predicts your risk of developing diabetes based on health indicators from CDC's BRFSS survey.
""")

# User input section
st.header("Your Health Profile")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Vital Signs")
    highbp = st.radio(FEATURE_EXPLANATIONS['HighBP'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    highchol = st.radio(FEATURE_EXPLANATIONS['HighChol'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    bmi = st.slider("Body Mass Index (BMI)", 15.0, 40.0, 25.0, 0.1)
    genhlth = st.selectbox("General Health Rating", [1, 2, 3, 4, 5], 
                         format_func=lambda x: f"{x} - {'Excellent' if x==1 else 'Very Good' if x==2 else 'Good' if x==3 else 'Fair' if x==4 else 'Poor'}")

with col2:
    st.subheader("Lifestyle Factors")
    smoker = st.radio(FEATURE_EXPLANATIONS['Smoker'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    physactivity = st.radio(FEATURE_EXPLANATIONS['PhysActivity'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    fruits = st.radio(FEATURE_EXPLANATIONS['Fruits'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    veggies = st.radio(FEATURE_EXPLANATIONS['Veggies'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    alcohol = st.radio(FEATURE_EXPLANATIONS['HvyAlcoholConsump'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col3:
    st.subheader("Medical History")
    stroke = st.radio(FEATURE_EXPLANATIONS['Stroke'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heartdisease = st.radio(FEATURE_EXPLANATIONS['HeartDiseaseorAttack'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    diffwalk = st.radio(FEATURE_EXPLANATIONS['DiffWalk'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cholcheck = st.radio(FEATURE_EXPLANATIONS['CholCheck'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    nodoc = st.radio(FEATURE_EXPLANATIONS['NoDocbcCost'], [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Additional inputs
st.subheader("Demographic Information")
dcol1, dcol2, dcol3, dcol4 = st.columns(4)
with dcol1:
    age = st.selectbox("Age Group", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                      format_func=lambda x: f"{x*5}-{x*5+4} years")
with dcol2:
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5, 6],
                           format_func=lambda x: ["Never attended", "Elementary", "Some HS", 
                                                 "HS Grad", "Some College", "College Grad"][x-1])
with dcol3:
    income = st.selectbox("Income Level", [1, 2, 3, 4, 5, 6, 7, 8],
                        format_func=lambda x: f"${x*10000}-${(x+1)*10000}" if x<8 else ">$75k")
with dcol4:
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

# Health metrics
st.subheader("Recent Health Status")
mcol1, mcol2 = st.columns(2)
with mcol1:
    menthlth = st.slider("Days of poor mental health (past 30 days)", 0, 30, 0)
with mcol2:
    physhlth = st.slider("Days of poor physical health (past 30 days)", 0, 30, 0)

# Create feature dictionary (with engineered feature)
health_risk_index = genhlth * bmi
input_data = {
    'HighBP': highbp,
    'HighChol': highchol,
    'CholCheck': cholcheck,
    'Smoker': smoker,
    'Stroke': stroke,
    'HeartDiseaseorAttack': heartdisease,
    'PhysActivity': physactivity,
    'Fruits': fruits,
    'Veggies': veggies,
    'HvyAlcoholConsump': alcohol,
    'AnyHealthcare': 1,  # Default to having healthcare
    'NoDocbcCost': nodoc,
    'GenHlth': genhlth,
    'MentHlth': menthlth,
    'PhysHlth': physhlth,
    'DiffWalk': diffwalk,
    'Sex': sex,
    'Age': age,
    'Education': education,
    'Income': income,
    'Health_Risk_Index': health_risk_index
}

# Prediction button
predict_btn = st.button("Assess My Diabetes Risk", type="primary")

# Display results
if predict_btn:
    with st.spinner("Analyzing your health profile..."):
        prediction, probability = predict_diabetes(input_data)
        
        st.divider()
        st.header("Assessment Results")
        
        if prediction == 1:
            st.error(f"## ⚠️ High Diabetes Risk: {probability*100:.1f}% probability")
            st.markdown("""
            **Recommendations:**
            - Consult your doctor immediately
            - Monitor blood sugar levels regularly
            - Increase physical activity
            - Review diet with nutritionist
            """)
        else:
            st.success(f"## ✅ Low Diabetes Risk: {probability*100:.1f}% probability")
            st.markdown("""
            **Maintain your health:**
            - Continue healthy habits
            - Get annual checkups
            - Watch for risk factor changes
            """)
        
        # Show risk factors visualization
        st.subheader("Your Key Risk Factors")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create simplified risk factor display
        risk_factors = {
            'BMI': bmi,
            'Health Risk Index': health_risk_index,
            'High BP': "Yes" if highbp else "No",
            'Physical Inactivity': "Yes" if not physactivity else "No",
            'Poor Diet': "Yes" if not (fruits and veggies) else "No"
        }
        
        pd.Series(risk_factors).plot(kind='barh', ax=ax, color=['#e74c3c' if x in ['Yes', True] or (isinstance(x, float) and x > 25) else '#2ecc71' for x in risk_factors.values()])
        ax.set_title("Your Personal Risk Profile")
        st.pyplot(fig)
        
        # Disclaimer
        st.info("""
        **Disclaimer:** This tool provides risk estimates only. It is not a medical diagnosis. 
        Consult healthcare professionals for personalized medical advice.
        """)

# Add sidebar info
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
- **Model Accuracy**: 77.1%
- **Recall Rate**: 68.1% (detects 7/10 true cases)
- **Data Source**: CDC BRFSS 2015 Diabetes Dataset
""")
st.sidebar.divider()
st.sidebar.markdown("""
**Key Risk Factors Considered:**
- Health Risk Index (General Health × BMI)
- High Blood Pressure
- Physical Inactivity
- Age and Genetics
- Poor Diet Habits
""")