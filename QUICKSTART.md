# Quick Start Guide - Customer Churn ML Project

## 📦 What You Have

A complete, production-ready machine learning project for predicting customer churn in e-commerce. All files are ready to push to GitHub!

## 🚀 Getting Started (5 Minutes)

### Step 1: Clone to Your Machine
```bash
# Copy this entire folder to your local machine
# OR if you push to GitHub first:
git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Complete Pipeline
```bash
python scripts/train_and_evaluate.py
```

This will:
- ✅ Generate synthetic data (if not exists)
- ✅ Preprocess and engineer features
- ✅ Train 4 different models
- ✅ Evaluate and compare performance
- ✅ Generate visualizations
- ✅ Save results and best model

**Output**: Check `results/` folder for detailed reports and plots

## 📓 Explore Jupyter Notebooks

```bash
jupyter notebook
```

Then open:
1. **01_exploratory_data_analysis.ipynb** - Understand your data
2. **02_feature_engineering.ipynb** - See feature transformations
3. **03_model_training_evaluation.ipynb** - Train and evaluate models

## 📁 Project Structure

```
customer-churn-ml/
├── README.md                          # Full documentation
├── QUICKSTART.md                      # This file
├── requirements.txt                   # Dependencies
├── setup.py                          # Package setup
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
│
├── data/
│   ├── raw/                         # Raw data goes here
│   └── processed/                   # Processed data
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── src/                            # Source code
│   ├── data_loader.py             # Load/generate data
│   ├── preprocessor.py            # Data preprocessing
│   ├── model_training.py          # Train models
│   ├── model_evaluation.py        # Evaluate models
│   └── utils.py                   # Utilities & plots
│
├── scripts/
│   └── train_and_evaluate.py      # Main pipeline script
│
├── tests/                         # Unit tests
│   ├── test_preprocessor.py
│   └── test_model_evaluation.py
│
├── configs/
│   └── config.yaml               # Configuration
│
└── models/                       # Trained models saved here
```

## 🎯 Models Included

1. **Logistic Regression** - Baseline, interpretable
2. **Random Forest** - Good performance, feature importance
3. **XGBoost** - Best performer, gradient boosting
4. **SVM** - Alternative approach

Expected performance: **89%+ accuracy, 0.92+ ROC-AUC**

## 🔧 Running Tests

```bash
pytest tests/
```

## 📊 Expected Results

After running the pipeline:
- `results/evaluation_results.json` - Model metrics
- `results/summary_report.txt` - Detailed report
- `results/plots/` - Visualizations:
  - confusion_matrices.png
  - roc_curves.png
  - model_comparison charts
  - feature_distributions.png

## 🌐 Push to GitHub

1. Create new repo on GitHub
2. Initialize git:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Customer churn ML project"
   git branch -M main
   git remote add origin https://github.com/yourusername/customer-churn-ml.git
   git push -u origin main
   ```

## 💡 Key Features

✅ **Complete Pipeline** - Data → Features → Models → Evaluation
✅ **Multiple Models** - Compare 4 different algorithms
✅ **Automated** - Run entire pipeline with one command
✅ **Documented** - Comprehensive notebooks and code comments
✅ **Tested** - Unit tests for critical functions
✅ **Production-Ready** - Clean code, best practices
✅ **Visualizations** - Charts and plots included
✅ **Configurable** - YAML configuration file

## 🚀 Next Steps

1. **Customize the data**: Replace `data/raw/customer_churn.csv` with your real data
2. **Adjust features**: Edit feature engineering in `src/preprocessor.py`
3. **Tune hyperparameters**: Modify `configs/config.yaml`
4. **Deploy model**: Save and load trained model for predictions
5. **Monitor performance**: Retrain regularly with new data

## 📝 Making Predictions

```python
import pickle
from src.preprocessor import ChurnPreprocessor

# Load trained model
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
preprocessor = ChurnPreprocessor()

# Make predictions on new data
new_customer = {
    'account_age_months': 12,
    'subscription_plan': 'Premium',
    'total_purchase_amount': 500,
    # ... other features
}

# Preprocess and predict
processed = preprocessor.transform(new_customer)
prediction = model.predict(processed)
probability = model.predict_proba(processed)[0, 1]

print(f"Churn probability: {probability:.2%}")
```

## ❓ Troubleshooting

**ModuleNotFoundError?**
```bash
pip install -r requirements.txt
```

**Jupyter not found?**
```bash
pip install jupyter
jupyter notebook
```

**Data not generating?**
```bash
python src/data_loader.py  # Generates data manually
```

## 📚 Documentation

- **README.md** - Full project documentation
- **Code comments** - All functions have docstrings
- **Notebooks** - Interactive tutorials and examples

## 🎓 Learning Resources

- scikit-learn: https://scikit-learn.org
- XGBoost: https://xgboost.readthedocs.io
- Pandas: https://pandas.pydata.org
- Data Science: https://kaggle.com

## 📞 Support

If you encounter any issues:
1. Check the README.md for detailed info
2. Review the Jupyter notebooks for examples
3. Check function docstrings in source code
4. Review test files for usage examples

## ✨ Tips for Success

1. **Start with notebooks** - They provide interactive exploration
2. **Understand your data** - Run EDA notebook first
3. **Experiment with features** - Feature engineering is crucial
4. **Monitor overfitting** - Check validation scores
5. **Document changes** - Keep track of improvements

---

**You're all set!** 🎉

Run `python scripts/train_and_evaluate.py` to see the magic happen.

Good luck with your project! 🚀
