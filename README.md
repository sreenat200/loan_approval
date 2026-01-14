# 💰 Loan Approval Prediction App

An **end-to-end machine learning web application** built with **Streamlit** to predict whether a loan application will be **approved or rejected** based on borrower financial and credit information. The system supports **multiple ML models**, rich visual analytics, and generates **professional PDF reports** for decision review.

---

## 📌 Project Overview

The Loan Approval Prediction App is designed to assist banks, financial institutions, and analysts in making **data-driven loan decisions**. By analyzing applicant details such as income, employment experience, credit score, loan amount, and past loan defaults, the application predicts loan approval outcomes with probability scores.

The app provides:

* Real-time predictions
* Model comparison
* Explainable feature importance
* Visual performance analysis
* Downloadable PDF decision reports

---

## 🚀 Key Features

### 1. Multiple Machine Learning Models

* **Random Forest Classifier**
* **Logistic Regression**
* **XGBoost Classifier** (high-performance boosting model)

Users can dynamically select and compare models.

---

### 2. Live Loan Approval Prediction

* Interactive sidebar for borrower input
* Predicts **Approved / Rejected** loan status
* Displays **probability of rejection**
* Risk adjustment logic for previous loan defaults

---

### 3. Model Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score

Metrics are computed on a test dataset and displayed instantly.

---

### 4. Advanced Data Visualizations

* Confusion Matrix
* ROC Curve with AUC score
* Feature Importance plots
* Interactive Scatter Plots

Visualizations are built using **Plotly** for interactivity.

---

### 5. Professional PDF Report Generation

* Generates a detailed loan decision report
* Includes:

  * Customer information (optional)
  * Input borrower features
  * Prediction result and probability
  * Model performance metrics
  * Feature importance table

Reports are styled and generated using **ReportLab**.

---

## 🧠 Machine Learning Workflow

1. Load loan dataset from GitHub
2. Preprocess data (encoding, scaling, missing values)
3. Split dataset into training and testing sets
4. Train selected ML model
5. Evaluate model performance
6. Perform real-time predictions on user input
7. Generate visual insights and PDF reports

---

## 🧪 Dataset

* **Source:** Loan Approval Dataset (GitHub)
* **Target Variable:** `loan_status`

  * 0 → Approved
  * 1 → Rejected

### Features Used

* Person Age
* Person Income
* Employment Experience
* Loan Amount
* Loan Interest Rate
* Loan Percent Income
* Credit History Length
* Credit Score
* Previous Loan Defaults

---

## 🛠️ Technology Stack

### Frontend

* Streamlit
* Custom CSS for UI enhancement

### Backend & Machine Learning

* Python
* Scikit-learn
* XGBoost
* Pandas
* NumPy

### Visualization

* Plotly Express
* Plotly Graph Objects

### Reporting

* ReportLab (PDF generation)

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## 📈 Use Cases

* Loan approval decision support
* Credit risk analysis
* Financial analytics demonstrations
* Machine learning portfolio projects

---

## ✅ Advantages

* User-friendly interface
* Real-time predictions
* Multiple ML algorithms
* Explainable feature importance
* Automated professional reporting

---

## 🔮 Future Enhancements

* Deep learning models
* Cloud deployment (AWS / GCP / Azure)
* Role-based access (analyst, manager)
* Database integration
* Bias and fairness analysis

---

## 📄 License

MIT License

---

This project demonstrates a **production-style machine learning application**, combining predictive modeling, explainability, visualization, and reporti
