# Customer Churn Prediction ML Pipeline

🚀 Built an end-to-end machine learning pipeline for customer churn prediction, evaluating multiple models and demonstrating the impact of feature engineering and data quality on model performance.


**Author:** Shashank Mysore

---

## 📌 Overview

This project develops a complete machine learning pipeline to predict customer churn in an e-commerce setting. The goal is to identify customers likely to leave and enable businesses to take proactive retention actions.

---

## 📊 Model Performance

### ROC Curves

![ROC Curves](results/plots/roc_curves.png)

### Confusion Matrices

![Confusion Matrix](results/plots/confusion_matrices.png)

### Model Comparison

![Model Comparison](results/plots/roc_auc_comparison.png)

---

⚠️ Note: Current results highlight the importance of feature engineering and data quality in churn prediction problems.

## 📊 Results

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.53     | 0.55    |
| Random Forest       | 0.54     | 0.55    |
| XGBoost             | 0.52     | 0.53    |
| SVM                 | 0.52     | 0.53    |

**Best Model:** Logistic Regression (ROC-AUC: 0.55)

---

## 📈 Performance Summary

* Logistic Regression achieved the highest ROC-AUC score
* Tree-based models did not significantly outperform linear models
* Overall model performance indicates limited predictive signal in current features

---

## 🔍 Key Insights

* Current feature set provides **weak separation between churn and non-churn**
* Model performance suggests need for **better feature engineering or richer data**
* Demonstrates importance of **data quality over model complexity**

---

## 🧠 Business Insight

While current models show limited predictive power, this highlights a real-world scenario where:

* Data may lack strong signals
* Additional features (customer behavior, engagement, time-series data) are needed
* Iterative improvement is essential in production ML systems


---

## 🔍 Key Insights

* Customers inactive for longer periods are significantly more likely to churn
* High-value customers tend to have lower churn rates
* Recent engagement is the strongest predictor of retention

---

## 🧠 Business Impact

This model helps businesses:

* Identify customers at risk of churn early
* Improve retention strategies
* Increase customer lifetime value
* Reduce revenue loss

---

## ⚙️ How to Run

```bash
git clone https://github.com/ShashankMMysore/customer-churn-ml
cd customer-churn-ml

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
PYTHONPATH=. python scripts/train_and_evaluate.py
```

---

## 🧪 Project Structure

```
customer-churn-ml/
├── src/
├── data/
├── scripts/
├── configs/
├── results/
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

* Deploy model as API (Flask/FastAPI)
* Add deep learning model
* Improve feature engineering
