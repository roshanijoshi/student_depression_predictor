import streamlit as st
import joblib
import numpy as np

model = joblib.load('student_depression_dataset.pkl')
st.title("Student Depression Prediction")

work_hours_input = st.text_input("Enter daily study hours", value="5", max_chars=2)
financial_stress_input = st.text_input("Enter financial stress level (1-10)", value="5", max_chars=2)

if st.button('Predict Depression Level'):
    try:
        work_hours = float(work_hours_input)
        financial_stress = int(financial_stress_input)

        if 0 <= work_hours <= 24 and 1 <= financial_stress <= 10:
            predicted_depression = model.predict(np.array([[work_hours, financial_stress]]))[0]
            if predicted_depression == 1:
                  message = "The student have higher level of depression."
            else:
                message = "The student have lower depression."

            st.success(message)
        else:
           st.error("Please enter valid values: work hours between 0-24, financial stress between 1 to 10.")
    except ValueError:
        st.error("Invalid input. Please enter numeric value for work and financial stress ")

st.markdown("**Connect with me on social media for more updates and insights!**")
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    
    <style>

    .social-buttons a{
        font-size: 24px;
        margin-right: 15px; 
        color: #0073e6;
        text-decoration: none;
        }
    </style>       
  
    <div class="social-buttons">
        <a href="https://www.instagram.com/roshanijoshi01?igsh=MXhldm54NWVrZW9xOA==" target="_blank">
            <i class="fab fa-instagram"></i>
        </a> 
        <a href="https://www.facebook.com/roshani.joshi.3348" target="_blank">
            <i class="fab fa-facebook"></i>
        </a>
        <a href="https://www.linkedin.com/in/roshani-joshi-276635232/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
