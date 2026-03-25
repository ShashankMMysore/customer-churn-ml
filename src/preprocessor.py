"""
Data preprocessing and feature engineering for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class ChurnPreprocessor:
    """
    Preprocessing pipeline for customer churn dataset.
    Handles missing values, feature engineering, and scaling.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of dataset to include in test split
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def fit_transform(self, df):
        """
        Fit preprocessing on training data and transform.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw customer churn data
        
        Returns
        -------
        tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Separate features and target
        X = df.drop(columns=['churn', 'customer_id'])
        y = df['churn']
        
        # Identify categorical and numerical features
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Engineer new features
        X = self._engineer_features(X)
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X = self._encode_categorical(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale numerical features
        X_train = self._scale_features(X_train, fit=True)
        X_test = self._scale_features(X_test, fit=False)
        
        self.feature_names = X_train.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessing.
        
        Parameters
        ----------
        df : pd.DataFrame
            New customer data
        
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        X = df.drop(columns=['churn', 'customer_id'], errors='ignore')
        
        # Engineer features
        X = self._engineer_features(X)
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical
        X = self._encode_categorical(X, fit=False)
        
        # Scale
        X = self._scale_features(X, fit=False)
        
        return X
    
    def _engineer_features(self, X):
        """
        Create new features from existing ones.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        
        Returns
        -------
        pd.DataFrame
            Features with engineered columns
        """
        X = X.copy()
        
        # Customer lifetime value (CLV) metric
        if 'total_purchase_amount' in X.columns:
            X['clv'] = X['total_purchase_amount']
        
        # Purchase frequency ratio
        if 'number_of_purchases' in X.columns and 'account_age_months' in X.columns:
            X['purchase_frequency'] = X['number_of_purchases'] / (X['account_age_months'] + 1)
        
        # Recent activity indicator
        if 'days_since_last_purchase' in X.columns:
            X['is_inactive'] = (X['days_since_last_purchase'] > 90).astype(int)
        
        # Engagement score
        if 'session_frequency_per_month' in X.columns and 'product_reviews_count' in X.columns:
            X['engagement_score'] = (
                X['session_frequency_per_month'] + 
                X['product_reviews_count']
            )
        
        # Support complexity
        if 'customer_support_tickets' in X.columns:
            X['high_support_user'] = (X['customer_support_tickets'] > 3).astype(int)
        
        return X
    
    def _handle_missing_values(self, X):
        """
        Handle missing values in the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        X = X.copy()
        
        # Fill numerical missing values with mean
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mean(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        return X
    
    def _encode_categorical(self, X, fit=False):
        """
        Encode categorical variables using one-hot encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        fit : bool, default=False
            Whether to fit the encoders
        
        Returns
        -------
        pd.DataFrame
            Data with encoded categorical features
        """
        X = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if fit:
                # Fit on training data
                dummies = pd.get_dummies(X[[col]], prefix=col, drop_first=True)
                self.label_encoders[col] = dummies.columns.tolist()
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
            else:
                # Transform using fitted encoders
                if col in self.label_encoders:
                    dummies = pd.get_dummies(X[[col]], prefix=col, drop_first=True)
                    # Ensure same columns
                    for enc_col in self.label_encoders[col]:
                        if enc_col not in dummies.columns:
                            dummies[enc_col] = 0
                    dummies = dummies[self.label_encoders[col]]
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        return X
    
    def _scale_features(self, X, fit=False):
        """
        Scale numerical features using StandardScaler.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        fit : bool, default=False
            Whether to fit the scaler
        
        Returns
        -------
        pd.DataFrame
            Scaled features
        """
        X = X.copy()
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Convenience function to preprocess data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw customer churn data
    test_size : float, default=0.2
        Test set proportion
    random_state : int, default=42
        Random seed
    
    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test, feature_names, preprocessor)
    """
    preprocessor = ChurnPreprocessor(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    return X_train, X_test, y_train, y_test, preprocessor.feature_names, preprocessor


if __name__ == "__main__":
    from data_loader import load_data
    
    # Example usage
    print("Loading raw data...")
    df = load_data()
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data(df)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nFeatures ({len(feature_names)}): {feature_names[:5]}...")
    print(f"\nClass distribution (train):\n{y_train.value_counts()}")
    print(f"\nClass distribution (test):\n{y_test.value_counts()}")
