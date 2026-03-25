# Customer Churn Prediction ML Pipeline

**Author:** Shashank Mysore

## 📌 Overview

This project builds an end-to-end machine learning pipeline to predict customer churn in an e-commerce setting. The goal is to identify customers likely to leave and help businesses improve retention.

---

## 🚀 Features

* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
* Model training (Logistic Regression, Random Forest, XGBoost, SVM)
* Model evaluation and comparison
* Config-driven pipeline

---

## 📊 Results

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.82     | 0.89    |
| Random Forest       | 0.87     | 0.91    |
| XGBoost             | 0.89     | 0.92    |
| SVM                 | 0.85     | 0.90    |

**Best Model:** XGBoost

### 🔍 Key Insights

* Customers inactive for 30+ days are more likely to churn
* Frequent buyers have lower churn rates
* Recent engagement is the strongest predictor

---

## 🧠 Business Impact

This model can help:

* Identify at-risk customers early
* Improve retention strategies
* Increase customer lifetime value

---

## ⚙️ How to Run

```bash
git clone https://github.com/ShashankMMysore/customer-churn-ml
cd customer-churn-ml
pip install -r requirements.txt
python scripts/train_and_evaluate.py
```

---

## 🧪 Project Structure

```
customer-churn-ml/
├── src/
├── data/
├── notebooks/
├── configs/
├── tests/
```

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* XGBoost
* pandas
* matplotlib

---

## 📈 Future Improvements

* Deploy as API (Flask/FastAPI)
* Add deep learning model
* Improve feature engineering
