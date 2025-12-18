# GPU-Accelerated-Telco-Customer-Churn-Prediction

This project implements an end-to-end machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.
The notebook demonstrates GPU-accelerated model training using XGBoost, along with exploratory data analysis, model evaluation, and a Streamlit-based interface.

## Project Contents
```
GPU-Accelerated Telco Customer Churn Prediction
├── churn_gpu.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## Objectives

• Perform exploratory data analysis (EDA) on telecom customer data

• Train a churn prediction model using XGBoost

• Utilize GPU acceleration when available, with CPU fallback

• Evaluate model performance using standard classification metrics

• Demonstrate practical, applied machine learning workflow

## Technologies Used

•Python

• Jupyter Notebook

• XGBoost (GPU / CPU)

• Scikit-learn

• Pandas, NumPy

• Matplotlib, Seaborn

• PyTorch (GPU availability check)

## Dataset

Name: Telco Customer Churn Dataset

Source: IBM Sample Datasets

Target Variable: Churn (Yes / No)

The dataset is included in this repository for learning and academic purposes.

## Workflow Overview

1. GPU availability check

2. Data loading and preprocessing

3. Exploratory Data Analysis (EDA)

    • Count plots

    • KDE plots

    • Correlation heatmap

4. Feature importance

    • Feature encoding and train-test split

5. XGBoost model training (GPU if available)

6. Model evaluation

    • Accuracy
    
    • Classification report
    
    • Confusion matrix

## How to Run the Notebook

### 1. Clone the repository:
``` bash
git clone https://github.com/your-username/your-repo-name.git
```

### 2. Open the notebook:
``` bash
jupyter notebook churn_gpu.ipynb
```

### 3. Install required dependencies (if not already installed):
``` bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn torch
```
No API keys or credentials are required for these steps.


## Optional: Run the Streamlit Web Application

The notebook also demonstrates deploying the trained model using Streamlit.

To run this part:

1. Create a free ngrok account

2. Generate an ngrok authentication token

3. In the notebook, replace the placeholder:
``` bash
ngrok config add-authtoken "YOUR_NGROK_TOKEN"
```

### 5. Run all cells in order.

For GPU execution, ensure CUDA-enabled GPU and compatible XGBoost version are installed.

## Results

• The model achieves strong predictive performance on unseen test data.

• GPU acceleration significantly reduces training time when available.

• Feature importance analysis highlights key drivers of customer churn.
