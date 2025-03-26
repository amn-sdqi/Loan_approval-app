import streamlit as st
import joblib
import re
import pandas as pd
import os
import time

# Get and print the current working directory
cwd = os.getcwd()

# Define correct path to the model file
# Load models and encoders
model_path = os.path.join(cwd, "loan_pred.joblib")
model = joblib.load(model_path)



# emp_title_enc_path = 
emp_title_enc = joblib.load(os.path.join(cwd, "emp_title_enc.joblib"))

title_enc_path = os.path.join(cwd, "title_enc.joblib")
title_enc = joblib.load(title_enc_path)

# Define mappings
grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
sub_grade_mapping = {
    f"{grade}{num}": i + 1
    for i, (grade, num) in enumerate(
        [(g, n) for g in ["A", "B", "C", "D", "E", "F", "G"] for n in range(1, 6)]
    )
}
emp_length_map = {
    "10+ years": 10,
    "2 years": 2,
    "< 1 year": 0.5,
    "3 years": 3,
    "5 years": 5,
    "1 year": 1,
    "4 years": 4,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "Not Provided": 0.1,
}
home_ownership_map = {
    "ANY": 1,
    "MORTGAGE": 13,
    "NONE": 14,
    "OTHER": 15,
    "OWN": 15.1,
    "RENT": 18,
}
verification_status_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 0.5}
purpose_map = {
    "car": 1,
    "credit_card": 1.1,
    "debt_consolidation": 2,
    "educational": 3,
    "home_improvement": 4,
    "house": 4.1,
    "major_purchase": 4.2,
    "medical": 4.3,
    "moving": 4.4,
    "other": 5,
    "renewable_energy": 6,
    "small_business": 7,
    "vacation": 8,
    "wedding": 9,
}
initial_list_status_map = {"f": 1, "w": 0}
application_type_map = {"DIRECT_PAY": 1, "INDIVIDUAL": 2, "JOINT": 3}


def encoding(df):
    df["term"] = df["term"].map({"36 months": 0, "60 months": 1})
    df["grade"] = df["grade"].map(grade_mapping)
    df["sub_grade"] = df["sub_grade"].map(sub_grade_mapping)
    df["emp_length"] = df["emp_length"].map(emp_length_map)
    df["home_ownership"] = df["home_ownership"].map(home_ownership_map)
    df["verification_status"] = df["verification_status"].map(verification_status_map)
    df["purpose"] = df["purpose"].map(purpose_map)
    df["initial_list_status"] = df["initial_list_status"].map(initial_list_status_map)
    df["application_type"] = df["application_type"].map(application_type_map)

    # Handle datetime features
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_d"] = df["issue_d"].apply(lambda x: (x.year * 10 + x.month) / 1000)

    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["earliest_cr_line"] = df["earliest_cr_line"].apply(
        lambda x: (x.year * 10 + x.month) / 1000
    )

    # Apply label encoders
    df["emp_title"] = emp_title_enc.transform(df[["emp_title"]])
    df["title"] = title_enc.transform(df[["title"]])

    # Extract zip code (assuming address has a zip code)
    df["address"] = (
        df["address"]
        .apply(
            lambda x: re.search(r"\b\d{5}\b", x).group()
            if re.search(r"\b\d{5}\b", x)
            else "0"
        )
        .astype(int)
    )

    return df


# Title
st.title("🏦 Loan Approval Prediction")
st.markdown("### 📌 Enter loan details to predict approval status.")

# Divider
st.markdown("---")

# Form with Layout
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        loan_amnt = st.number_input("💰 Loan Amount", value=10000.0, min_value=500.0, step=500.0)
        term = st.selectbox("📅 Loan Term", ["36 months", "60 months"])
        int_rate = st.number_input("📈 Interest Rate (%)", value=11.44, min_value=0.0, max_value=50.0, step=0.1)
        installment = st.number_input("📆 Monthly Installment", value=329.48)
        grade = st.selectbox("🏷️ Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox("🔠 Loan Subgrade", ["1", "2", "3", "4", "5"])
        emp_title = st.text_input("👨‍💼 Employment Title", "Marketing / Tech / Business")
        emp_length = st.selectbox(
            "📅 Employment Length",
            ["Not Provided", "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
        )

    with col2:
        home_ownership = st.selectbox("🏠 Home Ownership", ["NONE", "MORTGAGE", "OTHER", "OWN", "RENT", "ANY"])
        annual_inc = st.number_input("💸 Annual Income", value=117000.0, step=1000.0)
        verification_status = st.selectbox("✅ Verification Status", ["Verified", "Not Verified"])
        issue_d = st.date_input("📅 Loan Issue Date")
        purpose = st.selectbox(
            "🎯 Loan Purpose",
            ["car", "credit_card", "debt_consolidation", "educational", "home_improvement", "house", "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business", "vacation", "wedding"],
        )
        title = st.text_input("📄 Loan Title", "Personal Loan")
        dti = st.number_input("📊 Debt-to-Income Ratio (%)", value=(installment / (annual_inc / 12)) * 100, format="%.2f")
        earliest_cr_line = st.date_input("📅 Earliest Credit Line Date")

    st.markdown("---")  # Divider

    col3, col4 = st.columns(2)
    with col3:
        open_acc = st.number_input("🔓 Open Credit Lines", value=16)
        pub_rec = st.number_input("📌 Public Records", value=0)
        revol_bal = st.number_input("💳 Revolving Balance", value=36369.0)
        revol_util = st.number_input("📈 Revolving Utilization (%)", value=41.8)
    with col4:
        total_acc = st.number_input("📋 Total Credit Accounts", value=25.0)
        initial_list_status = st.selectbox("📝 Initial Listing Status", ["Waiting", "Fulfilled"])
        application_type = st.selectbox("📄 Application Type", ["INDIVIDUAL", "JOINT", "DIRECT_PAY"])
        mort_acc = st.number_input("🏦 Mortgage Accounts", value=0.0)
        pub_rec_bankruptcies = st.selectbox("⚖️ Bankruptcies", ["No", "Yes"])
    
    address = st.text_input("📍 Address", "0174 Michelle Gateway, Mendozaberg, OK 22690")

    submitted = st.form_submit_button("🚀 Predict Loan Status")

if submitted:
    st.info("🔄 Processing your request...")

    # Convert form inputs to JSON
    data = {
        "loan_amnt": loan_amnt,
        "term":0 if term == "36 months" else 1,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": grade + sub_grade,
        "emp_title": emp_title,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "issue_d": issue_d,
        "purpose": purpose,
        "title": title,
        "dti": dti,
        "earliest_cr_line": earliest_cr_line,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "initial_list_status": "w" if initial_list_status == "Waiting" else "f",
        "application_type": application_type,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": 1.0 if pub_rec_bankruptcies == "Yes" else 0.0,
        "address": address,
    }

    # Convert input JSON to DataFrame
    dataframe = pd.DataFrame([data])

    # Simulate processing delay
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Encode categorical features
    encoded_data = dataframe  # Assume encoding function is handled elsewhere

    # Making prediction
    result = model.predict(encoded_data)

    # Show result
    if result == 1:
        st.success("🎉 Your Loan has been Approved!")
    else:
        st.error("❌ Sorry, Our System has suggested you for rejection.")
