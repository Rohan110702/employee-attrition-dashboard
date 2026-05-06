import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop unnecessary columns
df = df.drop(
    ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
    axis=1
)

# Encode categorical columns
for column in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("Dashboard Filters")

selected_attrition = st.sidebar.selectbox(
    "Filter Attrition",
    ["All", "Stayed", "Left"]
)

# -----------------------------------
# MAIN TITLE
# -----------------------------------

st.title("Employee Attrition Analytics Dashboard")

st.markdown(
    """
    Interactive dashboard for analyzing employee attrition trends
    and predicting employee turnover using Machine Learning.
    """
)

# -----------------------------------
# METRICS
# -----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Employees",
        len(df)
    )

with col2:
    st.metric(
        "Employees Left",
        int(df["Attrition"].sum())
    )

with col3:
    attrition_rate = round(
        (df["Attrition"].sum() / len(df)) * 100,
        2
    )

    st.metric(
        "Attrition Rate",
        f"{attrition_rate}%"
    )

# -----------------------------------
# FILTER DATA
# -----------------------------------

filtered_df = df.copy()

if selected_attrition == "Stayed":
    filtered_df = df[df["Attrition"] == 0]

elif selected_attrition == "Left":
    filtered_df = df[df["Attrition"] == 1]

# -----------------------------------
# CHARTS
# -----------------------------------

col4, col5 = st.columns(2)

with col4:

    fig1 = px.histogram(
        filtered_df,
        x="Attrition",
        color="Attrition",
        title="Employee Attrition Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with col5:

    fig2 = px.box(
        filtered_df,
        x="Attrition",
        y="MonthlyIncome",
        color="Attrition",
        title="Monthly Income vs Attrition"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# -----------------------------------
# PREDICTION SECTION
# -----------------------------------

st.header("Predict Employee Attrition")

col6, col7 = st.columns(2)

with col6:

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    monthly_income = st.slider(
        "Monthly Income",
        1000,
        20000,
        5000
    )

with col7:

    years_at_company = st.slider(
        "Years At Company",
        0,
        40,
        5
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

# Sample Input
input_data = X.iloc[[0]].copy()

input_data["Age"] = age
input_data["MonthlyIncome"] = monthly_income
input_data["YearsAtCompany"] = years_at_company
input_data["JobSatisfaction"] = job_satisfaction

# Prediction
prediction = model.predict(input_data)[0]

prediction_probability = model.predict_proba(input_data)[0][1]

# -----------------------------------
# RESULT
# -----------------------------------

st.subheader("Prediction Result")

if prediction == 1:

    st.error(
        f"Employee likely to leave "
        f"(Probability: {prediction_probability:.2f})"
    )

else:

    st.success(
        f"Employee likely to stay "
        f"(Probability: {1 - prediction_probability:.2f})"
    )

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "Built using Python, Streamlit, Scikit-learn, and Plotly"
)