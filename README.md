# Customer Churn Prediction - E-Commerce ML Project

A comprehensive machine learning project that predicts customer churn for an e-commerce platform using classification models. This project demonstrates data analysis, feature engineering, model training, evaluation, and deployment-ready code.

## 📋 Project Overview

**Business Problem:** Predict which customers are likely to churn (stop using the service) to enable proactive retention strategies.

**Solution:** Binary classification model using customer behavior, account, and transaction data.

**Key Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC for model comparison
- Feature importance analysis

## 📊 Dataset

- **Source:** Synthetic e-commerce customer dataset
- **Samples:** 10,000 customers
- **Features:** 15 customer attributes (behavioral, demographic, transactional)
- **Target:** Binary (Churned: 0/1)
- **Class Distribution:** ~20% churn rate (realistic for e-commerce)

## 🗂️ Project Structure

```
customer-churn-ml/
├── README.md                 # This file
├── requirements.txt          # Project dependencies
├── setup.py                  # Package setup configuration
├── .gitignore               # Git ignore rules
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv           # Raw dataset
│   └── processed/
│       └── customer_churn_processed.csv # Cleaned data
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb    # EDA & visualization
│   ├── 02_feature_engineering.ipynb          # Feature creation
│   └── 03_model_training_evaluation.ipynb    # Model development
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessor.py       # Data preprocessing & feature engineering
│   ├── model_training.py     # Model training utilities
│   ├── model_evaluation.py   # Evaluation metrics
│   └── utils.py              # Helper functions
│
├── models/
│   └── churn_model.pkl       # Trained model (generated after training)
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   └── test_model_evaluation.py
│
├── configs/
│   └── config.yaml           # Configuration file
│
└── scripts/
    └── train_and_evaluate.py # Main training script
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Data & Train Model
```bash
python scripts/train_and_evaluate.py
```

### 5. Explore Results
```bash
jupyter notebook notebooks/03_model_training_evaluation.ipynb
```

## 📈 Key Features

### Data Processing
- Handling missing values
- Categorical encoding (one-hot, label encoding)
- Feature scaling (StandardScaler)
- Train-test split with stratification

### Feature Engineering
- Customer lifetime value (CLV)
- Purchase frequency metrics
- Recency-based features
- Account age buckets
- Spending velocity

### Models Trained
1. **Logistic Regression** - Baseline model
2. **Random Forest** - High-performing classifier
3. **Gradient Boosting (XGBoost)** - Best model with feature importance
4. **SVM** - Alternative approach

### Evaluation
- Confusion matrix & classification reports
- ROC-AUC curves
- Precision-Recall curves
- Feature importance visualization
- Cross-validation scoring

## 📦 Dependencies

- Python 3.8+
- pandas, numpy - Data manipulation
- scikit-learn - ML algorithms
- xgboost - Gradient boosting
- matplotlib, seaborn - Visualization
- jupyter - Interactive notebooks

See `requirements.txt` for complete list with versions.

## 💡 Usage Examples

### Train & Evaluate Models
```python
from src.model_training import train_models
from src.data_loader import load_data

X_train, X_test, y_train, y_test = load_data()
models = train_models(X_train, y_train)
```

### Make Predictions
```python
import pickle

with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

### Generate Reports
```bash
python scripts/train_and_evaluate.py --output_dir results/
```

## 📊 Results Summary

After training, you'll get:
- Model performance comparison
- Confusion matrices for each model
- ROC-AUC curves
- Feature importance rankings
- Hyperparameter recommendations

**Expected Performance:**
- Baseline (Logistic Regression): ~82% accuracy
- Best Model (XGBoost): ~89% accuracy
- ROC-AUC: 0.92+

## 🔍 Model Interpretability

Feature importance reveals key churn drivers:
1. Account age
2. Recent purchase activity
3. Customer lifetime value
4. Support ticket frequency
5. Subscription plan type

Use SHAP values for individual prediction explanations.

## 🧪 Testing

Run unit tests to validate data processing and evaluation metrics:
```bash
pytest tests/
```

## 📝 Notebooks

1. **EDA Notebook** - Data exploration, distributions, correlations
2. **Feature Engineering** - Creating predictive features
3. **Model Training** - Training, tuning, and comparing models

## 🚢 Production Deployment

To deploy this model:

1. **Model Serialization** - Already saved as pickle file
2. **API Wrapper** - Flask/FastAPI for predictions
3. **Docker Container** - For consistent deployment
4. **Monitoring** - Track model performance over time

Example API endpoint:
```python
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data])
    return jsonify({'churn_probability': float(prediction[0])})
```

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Guide](https://xgboost.readthedocs.io)
- [Feature Engineering Best Practices](https://kaggle.com/tag/feature-engineering)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💼 Author

Created as a demonstration of end-to-end machine learning practices.

---

**⭐ If you found this helpful, please star the repository!**
