import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="HR Attrition Analytics", layout="wide")
st.title("📊 Employee Attrition Analysis & Prediction Dashboard")
st.markdown("This dashboard tracks key HR metrics and uses Machine Learning to predict employee flight risk.")

# 2. DATA LOADING & PREPROCESSING
@st.cache_data
def load_data():
    df = pd.read_csv("data/employee_data.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Please place 'employee_data.csv' inside the 'data/' folder.")
    st.stop()

# Data Cleaning for ML
df_ml = df.copy()
df_ml['Attrition'] = df_ml['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df_ml = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns])

categorical_cols = df_ml.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

# 3. MACHINE LEARNING MODEL TRAINING
X = df_ml.drop(columns=['Attrition'])
y = df_ml['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Calculate Risk Scores for all current employees
df['Flight_Risk_Score'] = model.predict_proba(X)[:, 1]
df['Risk_Category'] = pd.cut(df['Flight_Risk_Score'], 
                             bins=[0, 0.3, 0.7, 1.0], 
                             labels=['Low Risk', 'Medium Risk', 'High Risk'])

# 4. DASHBOARD KEY PERFORMANCE INDICATORS (KPIs)
total_employees = len(df)
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
avg_age = df['Age'].mean()
high_risk_count = (df['Risk_Category'] == 'High Risk').sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Headcount", f"{total_employees}")
kpi2.metric("Overall Attrition Rate", f"{attrition_rate:.1f}%")
kpi3.metric("Average Employee Age", f"{avg_age:.1f} Yrs")
kpi4.metric("🚨 High Risk Employees", f"{high_risk_count}")

st.markdown("---")

# 5. INTERACTIVE CHARTS & VISUALIZATIONS
col1, col2 = st.columns(2)

with col1:
    st.subheader("Attrition by Department")
    dept_attrition = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
    fig_dept = px.bar(dept_attrition, x='Department', y='Count', color='Attrition',
                      barmode='group', color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'})
    st.plotly_chart(fig_dept, use_container_width=True)

with col2:
    st.subheader("Monthly Income vs. Attrition")
    fig_income = px.box(df, x='Attrition', y='MonthlyIncome', color='Attrition',
                        color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'})
    st.plotly_chart(fig_income, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Impact of Overtime on Turnover")
    ot_attrition = df.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
    fig_ot = px.bar(ot_attrition, x='OverTime', y='Count', color='Attrition', barmode='stack')
    st.plotly_chart(fig_ot, use_container_width=True)

with col4:
    st.subheader("Risk Distribution (ML Model Output)")
    risk_counts = df['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    fig_risk = px.pie(risk_counts, values='Count', names='Risk Category', 
                      color='Risk Category',
                      color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'})
    st.plotly_chart(fig_risk, use_container_width=True)

# 6. RISK EXPLORER TABLE
st.markdown("---")
st.subheader("🔍 High Risk Employee Explorer")
st.markdown("Filter active employees flagged by the AI as potential flight risks to take proactive retention actions.")

selected_dept = st.selectbox("Filter by Department", options=['All'] + list(df['Department'].unique()))

filtered_df = df[(df['Attrition'] == 'No') & (df['Risk_Category'] == 'High Risk')]
if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]

st.dataframe(
    filtered_df[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany', 'Flight_Risk_Score']]
    .sort_values(by='Flight_Risk_Score', ascending=False)
    .style.format({'Flight_Risk_Score': '{:.2%}'})
)