#!/usr/bin/env python3
"""
Script to validate the trained Isolation Forest model on new health data.
This allows testing how well the model performs on unseen data.
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def load_model(model_dir):
    """Load the trained model, scaler, and feature names"""
    model_path = os.path.join(model_dir, 'isolation_forest_model.pkl')
    scaler_path = os.path.join(model_dir, 'standard_scaler.pkl')
    features_path = os.path.join(model_dir, 'feature_names.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature names file not found: {features_path}")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the feature names
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    
    return model, scaler, feature_names

def load_validation_data(filepath):
    """Load validation data from a CSV file"""
    print(f"Loading validation data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Validation data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records for validation")
    
    return df

def preprocess_validation_data(df, required_features):
    """Preprocess the validation data to match the training data format"""
    # Create one-hot encoding for categorical variables
    condition_dummies = pd.get_dummies(df['condition'], prefix='condition')
    gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
    activity_dummies = pd.get_dummies(df['activity_level'], prefix='activity')
    
    # Combine all features
    features = pd.concat([
        df[['heart_rate', 'blood_oxygen', 'age']],
        condition_dummies,
        gender_dummies,
        activity_dummies
    ], axis=1)
    
    # Check for missing features and add them with zeros
    missing_features = set(required_features) - set(features.columns)
    for feature in missing_features:
        features[feature] = 0
    
    # Ensure features are in the same order as in training
    features = features[required_features]
    
    # The target variable is the 'is_anomaly' column
    if 'is_anomaly' in df.columns:
        labels = df['is_anomaly']
        return features, labels
    else:
        return features, None

def validate_model(model, scaler, features, labels=None):
    """Validate the model on new data"""
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # For RandomForestClassifier, we get direct predictions and probabilities
    predictions = model.predict(features_scaled)
    
    # Get probability scores for anomaly class (class 1)
    try:
        # Try to get probability scores if available
        anomaly_scores = model.predict_proba(features_scaled)[:, 1]
    except:
        # Fallback to decision function or binary predictions
        try:
            anomaly_scores = model.decision_function(features_scaled)
        except:
            # Last resort: just use the binary predictions
            anomaly_scores = predictions.astype(float)
    
    results = {
        'predictions': predictions,
        'anomaly_scores': anomaly_scores
    }
    
    if labels is not None:
        # Calculate metrics
        print("\nModel Validation Results:")
        print(classification_report(labels, predictions))
        
        # Create confusion matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        try:
            plt.savefig('roc_curve.png')
            plt.savefig('confusion_matrix.png')
            print("Saved ROC curve and confusion matrix plots")
        except Exception as e:
            print(f"Could not save plots: {e}")
        
        results['accuracy'] = (predictions == labels).mean()
        print(f"Accuracy: {results['accuracy']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate trained health anomaly detection model')
    parser.add_argument('--validation_data', type=str, default='./data/validation_data.csv', 
                       help='Validation data CSV file')
    parser.add_argument('--model_dir', type=str, default='../models', 
                       help='Directory containing the trained model files')
    parser.add_argument('--output', type=str, default='./data/validation_results.csv', 
                       help='Output file for validation results')
    
    args = parser.parse_args()
    
    # Load the model and related components
    model, scaler, feature_names = load_model(args.model_dir)
    
    # Load the validation data
    df = load_validation_data(args.validation_data)
    
    # Preprocess the validation data
    features, labels = preprocess_validation_data(df, feature_names)
    
    # Validate the model
    results = validate_model(model, scaler, features, labels)
    
    # Save the results
    if args.output:
        # Add results to original dataframe
        df['predicted_anomaly'] = results['predictions']
        df['anomaly_score'] = results['anomaly_scores']
        
        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 