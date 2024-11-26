
# Automated Machine Learning Workflow with OpenAI and Streamlit

This repository contains an automated machine learning workflow that leverages OpenAI's GPT models and Streamlit to help data scientists and developers quickly preprocess datasets, train machine learning models, and generate deployment code. The workflow simplifies the process of preprocessing data, selecting models, evaluating performance, and deploying models.

## Features

- **Dataset Preprocessing:** Automatically generates code for handling missing values, encoding categorical variables, and scaling numerical features.
- **Model Training & Evaluation:** Supports training of multiple machine learning models (Random Forest, Logistic Regression, SVM, XGBoost) and compares their performance using metrics like AUC and classification reports.
- **Deployment Code Generation:** Generates Python code to deploy the trained models using Streamlit, enabling users to input new data and get predictions.
- **Streamlit Interface:** A user-friendly web interface to interact with the workflow, upload datasets, select models, and view the results.

## Requirements

- Python 3.7+
- Streamlit
- OpenAI (API key required)
- scikit-learn
- xgboost
- pandas
- numpy
- joblib


## Setup

1. Clone this repository:
   
   ```bash
   git clone https://github.com/yourusername/automated-ml-workflow.git
   cd automated-ml-workflow
   ```

2. Set up the OpenAI API key by creating an environment variable:
   
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Step 1: Upload Dataset**  
   Upload your CSV file to the app. It will preview the dataset to give you an overview.

2. **Step 2: Generate Preprocessing Code**  
   The app uses OpenAI's GPT model to generate preprocessing code, including handling missing values, encoding categorical features, and scaling numerical features. You can view the generated code and run it.

3. **Step 3: Execute Preprocessing**  
   After reviewing the preprocessing code, you can run it on your dataset. The preprocessed data will be saved as `preprocessed.csv`.

4. **Step 4: Select Target Column**  
   Choose the target column from the dataset for model training.

5. **Step 5: Select Models to Train**  
   Select the models you want to train, including Random Forest, Logistic Regression, SVM, and XGBoost.

6. **Step 6: Train and Compare Models**  
   The selected models will be trained and evaluated on the preprocessed data. A comparison of the models' AUC and classification reports will be displayed.

7. **Step 7: Generate Deployment Code**  
   Generate Python code for deploying the trained models with Streamlit. This code will load the model, accept user inputs, and display predictions.

## Example Workflow

1. Upload your dataset.
2. View the automatically generated preprocessing code and execute it.
3. Choose the target column.
4. Select and train your models.
5. Review the performance reports for each model.
6. Generate deployment code for your trained models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the GPT models.
- Streamlit for creating the interactive app framework.
- scikit-learn and XGBoost for machine learning models.
- pandas and numpy for data manipulation.
```

### Key Sections:
- **Features:** Highlights the core functionalities of the app.
- **Requirements:** Lists all the Python packages required for the app to run.
- **Setup:** Explains how to clone the repository, install dependencies, and run the app.
- **How It Works:** Provides an overview of each step in the app, so users understand the workflow.
- **License:** Includes the licensing information for the project.
- **Acknowledgments:** Mentions the libraries and tools used in the project.

