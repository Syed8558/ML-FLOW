🚀 MLflow – Telco Customer Churn Prediction

This repository contains an MLflow-based Machine Learning experiment tracking system built to predict customer churn for a telecom company.
It demonstrates how to train models, track metrics, and manage experiments using MLflow instead of only notebooks.

📌 What this project does

This project helps answer:

Which telecom customers are likely to leave the service?

Using customer data, we train machine learning models and use MLflow to:

Track experiments

Store models

Compare performance

Save results in a database

⚙️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

MLflow

SQLite (mlflow.db)

📂 Repository Files
File	Purpose
eda2.py	Performs data analysis and model training
mlflow.db	MLflow database storing experiments & metrics
README.md	Project documentation

📊 What is MLflow used for here?

MLflow is used to:

Log model parameters

Log accuracy and other metrics

Save trained models

Track multiple experiments

Compare model performance

This makes the project reproducible and professional.

▶️ How to run

Install required libraries

pip install mlflow pandas scikit-learn


Run the ML experiment

python eda2.py


Start MLflow UI

mlflow ui


Open in browser

http://127.0.0.1:5000

🎯 Output

You will see:

Different ML experiments

Accuracy and metrics

Stored models

Versioned runs

All tracked inside MLflow dashboard.

💡 Why this project is important

Most people only train models in Jupyter notebooks.
This project shows how to use MLflow, which is what companies use to:

Track ML experiments

Manage model versions

Monitor performance

Build production-ready ML systems

👨‍💻 Author

Syed Sadath G
Data Scientist | Machine Learning | MLOps
