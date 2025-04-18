#!/usr/bin/env python3
"""
Script to train an Isolation Forest model for health anomaly detection using generated training data.
This model will be condition-aware, making it sensitive to the specific health profiles of users.
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath):
    """Load the training data from a CSV file"""
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {df['condition'].nunique()} different conditions")
    
    # Show a sample of the data
    print("\nSample data:")
    print(df.head())
    
    # Show data summary
    print("\nData summary:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Preprocess the training data for model training"""
    # Convert categorical variables to one-hot encoding
    # One-hot encode conditions
    condition_dummies = pd.get_dummies(df['condition'], prefix='condition')
    
    # One-hot encode gender
    gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
    
    # One-hot encode activity level
    activity_dummies = pd.get_dummies(df['activity_level'], prefix='activity')
    
    # Create features DataFrame with numeric values and one-hot encoded categories
    features = pd.concat([
        df[['heart_rate', 'blood_oxygen', 'age']],
        condition_dummies,
        gender_dummies,
        activity_dummies
    ], axis=1)
    
    # The target variable is the 'is_anomaly' column
    labels = df['is_anomaly']
    
    return features, labels

def train_model(features, labels, contamination=0.05, n_estimators=100, random_state=42):
    """Train a Random Forest classifier for health anomaly detection"""
    print(f"\nTraining Random Forest classifier with class balancing...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=random_state, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Count normal vs anomaly samples
    n_normal = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    print(f"Training data: {n_normal} normal samples, {n_anomaly} anomaly samples")
    
    # Use class weights to balance the classes, but make them more moderate
    # for more realistic predictions with nuanced data
    class_weight_ratio = min(n_normal/n_anomaly, 5.0) if n_anomaly > 0 else 5.0
    class_weights = {0: 1, 1: class_weight_ratio}
    print(f"Using class weights: {class_weights}")
    
    # Train a Random Forest classifier with more trees for better performance on nuanced data
    model = RandomForestClassifier(
        n_estimators=max(n_estimators, 200),  # Use at least 200 trees for nuanced data
        max_depth=None,                       # Let trees grow fully
        min_samples_split=2,                  # Default split criteria
        min_samples_leaf=1,                   # Allow leaf nodes with just one sample
        max_features='sqrt',                  # Standard practice for random forests
        bootstrap=True,                       # Sample with replacement
        class_weight=class_weights,           # Balance the classes
        random_state=random_state,
        n_jobs=-1                             # Use all available cores
    )
    
    # Train on labeled data
    model.fit(X_train_scaled, y_train)
    
    # Get feature importances to understand what's most useful for classification
    feature_importances = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top 10 most important features
    print("\nTop 10 most important features:")
    print(feature_importances.head(10))
    
    # Evaluate on training set
    train_preds = model.predict(X_train_scaled)
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    
    # Evaluate on test set
    test_preds = model.predict(X_test_scaled)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification reports
    print("\nTraining set evaluation:")
    print(classification_report(y_train, train_preds))
    
    print("\nTest set evaluation:")
    print(classification_report(y_test, test_preds))
    
    # Print additional evaluation metrics for the test set
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_probs)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Calculate Average Precision (area under PR curve)
    avg_precision = average_precision_score(y_test, test_probs)
    print(f"Average Precision: {avg_precision:.4f}")
    
    return model, scaler, features.columns.tolist()

def save_model(model, scaler, features, output_dir):
    """Save the trained model, scaler, and feature names to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'isolation_forest_model.pkl')
    scaler_path = os.path.join(output_dir, 'standard_scaler.pkl')
    features_path = os.path.join(output_dir, 'feature_names.pkl')
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the feature names
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature names saved to {features_path}")

def main():
    parser = argparse.ArgumentParser(description='Train an Isolation Forest model for health anomaly detection')
    parser.add_argument('--input', type=str, default='../data/training_data.csv', help='Input training data CSV file')
    parser.add_argument('--output_dir', type=str, default='../models', help='Output directory for model files')
    parser.add_argument('--contamination', type=float, default=0.05, help='Expected proportion of anomalies in the data')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
    
    args = parser.parse_args()
    
    # Load the training data
    df = load_data(args.input)
    
    # Preprocess the data
    features, labels = preprocess_data(df)
    
    # Train the model
    model, scaler, feature_names = train_model(
        features, 
        labels, 
        contamination=args.contamination,
        n_estimators=args.n_estimators
    )
    
    # Save the model
    save_model(model, scaler, feature_names, args.output_dir)
    
    print("\nDone! The model is now ready for use.")

if __name__ == "__main__":
    main() 