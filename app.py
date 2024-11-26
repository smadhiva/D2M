import openai
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc
)
import os

openai.api_key = " "
def query_openai(prompt, max_tokens=1500):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None

def generate_preprocessing_code(df):
    prompt = f"""
    Analyze the following dataset and generate Python preprocessing code that:
    1. Handles missing values.
    2. Encodes categorical variables.
    3. Scales numerical features.
    Save the preprocessed data as 'preprocessed.csv'.
    Dataset Summary:
    {df.describe(include='all').to_string()}
    """
    return query_openai(prompt)

def generate_model_training_code(df):
    prompt = f"""
    Based on the following dataset summary, suggest suitable ML models, generate the code to train and test them, and evaluate their performance:
    Dataset Summary:
    {df.describe(include='all').to_string()}
    Generate code to:
    - Train at least two models (Random Forest and Logistic Regression)
    - Compare the models' performance (accuracy, AUC, confusion matrix)
    - Print a classification report for each model
    """
    return query_openai(prompt)

def generate_deployment_code():
    prompt = """
    Generate Python code to deploy a trained machine learning model using Streamlit. The model should be able to:
    - Load the pre-trained model.
    - Accept user input (e.g., a CSV file or form) for prediction.
    - Display the prediction result on the app.
    """
    return query_openai(prompt)

def execute_preprocessing():
    try:
        result = subprocess.run(["python", "preprocessing.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Preprocessing completed successfully.")
            return True
        else:
            st.error(f"Error executing preprocessing.py: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def train_and_compare_models(df, target_column, selected_models):
    try:
        st.write("Preprocessed Data Preview:")
        st.dataframe(df)

        X = df.drop(columns=[target_column])  
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_reports = {}
        for model_name in selected_models:
            if model_name == 'Random Forest':
                model = RandomForestClassifier(random_state=42)
            elif model_name == 'Logistic Regression':
                model = LogisticRegression(random_state=42)
            elif model_name == 'SVM':
                model = SVC(probability=True, random_state=42)
            elif model_name == 'XGBoost':
                model = XGBClassifier(random_state=42)
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else model.predict_proba(X_test)

            model_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])

            joblib.dump(model, f"{model_name}_model.joblib")

            model_reports[model_name] = {
                "model": model,
                "auc": model_auc,
                "classification_report": classification_report(y_test, y_pred)
            }

        st.subheader("Model Comparison")
        for model_name, report in model_reports.items():
            st.write(f"{model_name} AUC:", report["auc"])
            st.write(f"{model_name} Classification Report:")
            st.text(report["classification_report"])

        return model_reports

    except Exception as e:
        st.error(f"Error training models: {e}")
        return None

st.title("Automated Machine Learning Workflow with OpenAI")

uploaded_file = st.file_uploader("Upload a dataset (CSV file)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df)

    st.header("Step 1: Generate Preprocessing Code")
    with st.spinner("Generating preprocessing code..."):
        preprocessing_code = generate_preprocessing_code(df)
    if preprocessing_code:
        st.subheader("Generated Preprocessing Code")
        st.code(preprocessing_code)

        with open("preprocessing.py", "w") as f:
            start_index = preprocessing_code.find("```python") + len("```python")
            end_index = preprocessing_code.find("```", start_index)
            python_code = preprocessing_code[start_index:end_index].strip()
            f.write(python_code)

    st.header("Step 2: Execute Preprocessing")
    if st.button("Run Preprocessing"):
        if execute_preprocessing():
            st.success("Preprocessed data saved as 'preprocessed.csv'.")

    df = pd.read_csv("preprocessed.csv")
    st.header("Step 3: Select Target Column")
    target_column = st.selectbox("Select target column", df.columns)

    st.header("Step 4: Select Models to Train")
    available_models = ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']
    selected_models = st.multiselect("Select models to train", available_models, default=available_models)

    st.header("Step 5: Train and Compare Models")
    if os.path.exists("preprocessed.csv"):
        if st.button("Train Models"):
            df = pd.read_csv("preprocessed.csv")
            model_reports = train_and_compare_models(df, target_column, selected_models)
            if model_reports:
                st.subheader("Generated Report")
                for model_name, report in model_reports.items():
                    st.write(f"{model_name} AUC: {report['auc']}")
                    st.text(report['classification_report'])
    else:
        st.warning("Preprocessed data not found. Please run preprocessing first.")

    st.header("Step 6: Generate Deployment Code")
    if st.button("Generate Deployment Code"):
      with st.spinner("Generating model deployment code..."):
        deployment_code = generate_deployment_code()
      if deployment_code:
        st.subheader("Generated Deployment Code")
        st.code(deployment_code)

        with open("model_deployment.py", "w") as f:
            f.write(deployment_code)
