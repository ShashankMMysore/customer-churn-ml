"""
Data loading and generation utilities for customer churn dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_churn_data(n_samples=10000, random_state=42):
    """
    Generate synthetic customer churn dataset for e-commerce domain.
    
    Parameters
    ----------
    n_samples : int, default=10000
        Number of customer records to generate
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        DataFrame with customer features and churn target
    """
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = np.arange(1, n_samples + 1)
    
    # Account features
    account_age_months = np.random.exponential(scale=12, size=n_samples)
    account_age_months = np.clip(account_age_months, 0, 120)
    
    subscription_plan = np.random.choice(
        ['Basic', 'Premium', 'Enterprise'], 
        size=n_samples, 
        p=[0.5, 0.35, 0.15]
    )
    
    # Purchase behavior
    total_purchase_amount = np.random.exponential(scale=500, size=n_samples)
    total_purchase_amount = np.maximum(total_purchase_amount, 0)
    
    number_of_purchases = np.random.poisson(lam=5, size=n_samples)
    number_of_purchases = np.maximum(number_of_purchases, 0)
    
    average_purchase_value = np.where(
        number_of_purchases > 0,
        total_purchase_amount / number_of_purchases,
        0
    )
    
    # Recency features
    days_since_last_purchase = np.random.exponential(scale=30, size=n_samples)
    days_since_last_purchase = np.clip(days_since_last_purchase, 0, 365)
    
    # Engagement features
    customer_support_tickets = np.random.poisson(lam=2, size=n_samples)
    product_reviews_count = np.random.poisson(lam=1.5, size=n_samples)
    page_views_per_session = np.random.gamma(shape=2, scale=3, size=n_samples)
    page_views_per_session = np.maximum(page_views_per_session, 1)
    
    session_frequency_per_month = np.random.exponential(scale=4, size=n_samples)
    session_frequency_per_month = np.maximum(session_frequency_per_month, 0)
    
    account_active_days = np.random.exponential(scale=200, size=n_samples)
    account_active_days = np.clip(account_active_days, 0, 365)
    
    # Additional features
    payment_method = np.random.choice(
        ['Credit Card', 'PayPal', 'Bank Transfer', 'Other'],
        size=n_samples,
        p=[0.5, 0.3, 0.15, 0.05]
    )
    
    customer_support_contacted = np.random.choice(
        [0, 1],
        size=n_samples,
        p=[0.7, 0.3]
    )
    
    marketing_emails_opted = np.random.choice(
        [0, 1],
        size=n_samples,
        p=[0.4, 0.6]
    )
    
    # Target variable: Churn (influenced by features)
    churn_prob = (
        0.1 +
        (days_since_last_purchase / 365) * 0.3 +
        (1 - np.minimum(account_age_months / 60, 1)) * 0.2 +
        (1 - (number_of_purchases / 20)) * 0.2 +
        (customer_support_tickets / 10) * 0.1 +
        (1 - np.minimum(session_frequency_per_month / 8, 1)) * 0.1
    )
    
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(n=1, p=churn_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'account_age_months': np.round(account_age_months, 2),
        'subscription_plan': subscription_plan,
        'total_purchase_amount': np.round(total_purchase_amount, 2),
        'number_of_purchases': number_of_purchases,
        'average_purchase_value': np.round(average_purchase_value, 2),
        'days_since_last_purchase': np.round(days_since_last_purchase, 2),
        'customer_support_tickets': customer_support_tickets,
        'product_reviews_count': product_reviews_count,
        'page_views_per_session': np.round(page_views_per_session, 2),
        'session_frequency_per_month': np.round(session_frequency_per_month, 2),
        'account_active_days': np.round(account_active_days, 2),
        'payment_method': payment_method,
        'customer_support_contacted': customer_support_contacted,
        'marketing_emails_opted': marketing_emails_opted,
        'churn': churn
    })
    
    return data


def load_data(raw_data_path='data/raw/customer_churn.csv', 
              generate_if_missing=True):
    """
    Load customer churn data from CSV file.
    Generates synthetic data if file doesn't exist.
    
    Parameters
    ----------
    raw_data_path : str
        Path to raw CSV file
    generate_if_missing : bool, default=True
        Whether to generate synthetic data if file is missing
    
    Returns
    -------
    pd.DataFrame
        Customer churn dataset
    """
    if os.path.exists(raw_data_path):
        print(f"Loading data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
    elif generate_if_missing:
        print(f"Data file not found. Generating synthetic data...")
        df = generate_synthetic_churn_data(n_samples=10000, random_state=42)
        
        # Create directory if it doesn't exist
        Path(raw_data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_data_path, index=False)
        print(f"Data saved to {raw_data_path}")
    else:
        raise FileNotFoundError(f"Data file not found at {raw_data_path}")
    
    return df


def load_processed_data(processed_data_path='data/processed/customer_churn_processed.csv'):
    """
    Load processed data for model training.
    
    Parameters
    ----------
    processed_data_path : str
        Path to processed CSV file
    
    Returns
    -------
    pd.DataFrame
        Processed customer churn dataset
    """
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
    
    return pd.read_csv(processed_data_path)


def save_data(df, path):
    """
    Save DataFrame to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    path : str
        Output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")


if __name__ == "__main__":
    # Generate and save sample data
    print("Generating synthetic customer churn dataset...")
    df = generate_synthetic_churn_data(n_samples=10000)
    
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/customer_churn.csv", index=False)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nChurn rate: {df['churn'].mean():.2%}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
