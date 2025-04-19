"""
Script to test the classification model with real health data.

This script:
1. Retrieves health data from the database
2. Uses the actual risk_score to determine true class
3. Compares with model predictions
4. Visualizes results

Usage:
python test_with_real_data.py --user_id YOUR_USER_ID --limit 100
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from bson import ObjectId

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('real_data_test.log')
    ]
)

logger = logging.getLogger(__name__)

# Make sure the services modules are in the path
sys.path.append('.')

# Import necessary modules
from services.risk_classification import RiskClassification
from services.classification_model import ClassificationModel
from services.health_service import HealthService
from models.health_data import HealthData
from models.user import User
from database import get_collection

# Directory for saving results
RESULTS_DIR = "real_data_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_user_health_data(user_id, limit=100):
    """
    Get health data for a specific user from the database
    
    Args:
        user_id (str): User ID
        limit (int): Maximum number of records to retrieve
        
    Returns:
        pandas.DataFrame: DataFrame with health data
    """
    try:
        # Convert string ID to ObjectId if needed
        if isinstance(user_id, str):
            user_id = user_id  # Keep as string since your interface uses strings
        
        # Get health data from database
        health_data = HealthData.get_by_user_id(user_id, limit=limit)
        
        if not health_data:
            logger.error(f"No health data found for user {user_id}")
            return None
        
        # Convert to DataFrame
        rows = []
        for record in health_data:
            # Extract basic fields
            row = {
                'heart_rate': float(record.get('heart_rate', 0)),
                'blood_oxygen': float(record.get('blood_oxygen', 0)),
                'timestamp': record.get('created_at', datetime.now())
            }
            
            # Extract legacy risk score if available
            if 'risk_score' in record:
                row['risk_score'] = float(record['risk_score'])
            elif 'legacy_risk_score' in record:
                row['risk_score'] = float(record['legacy_risk_score'])
            elif 'analysis_result' in record and 'risk_score' in record['analysis_result']:
                row['risk_score'] = float(record['analysis_result']['risk_score'])
            elif 'analysis_result' in record and 'legacy_risk_score' in record['analysis_result']:
                row['risk_score'] = float(record['analysis_result']['legacy_risk_score'])
            else:
                # If no risk score, calculate it
                row['risk_score'] = HealthService.calculate_risk_score(
                    row['heart_rate'], row['blood_oxygen'], None
                )
            
            # Convert risk score to risk class
            row['risk_class'] = RiskClassification.score_to_class(row['risk_score'])
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving health data: {e}")
        return None

def test_model_with_real_data(user_id, limit=100):
    """
    Test classification model with real health data
    
    Args:
        user_id (str): User ID
        limit (int): Maximum number of records to use
        
    Returns:
        dict: Test results
    """
    # Get user context
    user = User.get_by_id(user_id)
    user_context = {}
    if user:
        if 'age' in user:
            user_context['age'] = user['age']
        if 'health_conditions' in user:
            user_context['health_conditions'] = user['health_conditions']
    
    # Get health data
    df = get_user_health_data(user_id, limit)
    
    if df is None or len(df) == 0:
        logger.error("No data available for testing")
        return None
    
    logger.info(f"Retrieved {len(df)} records for testing")
    
    # Prepare test data
    X = df[['heart_rate', 'blood_oxygen']].values
    y_true = df['risk_class'].values
    
    # Get model predictions
    y_pred_ml = []
    y_pred_hybrid = []
    y_pred_rule = []
    
    for i, (hr, bo) in enumerate(X):
        # Get rule-based prediction
        risk_score = HealthService.calculate_risk_score(hr, bo, user_context)
        rule_class = RiskClassification.score_to_class(risk_score)
        y_pred_rule.append(rule_class)
        
        # Get ML prediction
        try:
            ml_result = ClassificationModel.predict_risk_class(user_id, [hr, bo], user_context)
            ml_class = ml_result['risk_class']
            hybrid_class = np.argmax(ml_result['probabilities'].values())
            
            y_pred_ml.append(ml_class)
            y_pred_hybrid.append(hybrid_class)
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            y_pred_ml.append(rule_class)
            y_pred_hybrid.append(rule_class)
    
    # Calculate accuracy
    ml_accuracy = np.mean(np.array(y_pred_ml) == y_true)
    rule_accuracy = np.mean(np.array(y_pred_rule) == y_true)
    hybrid_accuracy = np.mean(np.array(y_pred_hybrid) == y_true)
    
    logger.info(f"ML accuracy: {ml_accuracy:.4f}")
    logger.info(f"Rule-based accuracy: {rule_accuracy:.4f}")
    logger.info(f"Hybrid accuracy: {hybrid_accuracy:.4f}")
    
    # Generate confusion matrices
    ml_cm = np.zeros((3, 3), dtype=int)
    rule_cm = np.zeros((3, 3), dtype=int)
    hybrid_cm = np.zeros((3, 3), dtype=int)
    
    for i in range(len(y_true)):
        ml_cm[y_true[i], y_pred_ml[i]] += 1
        rule_cm[y_true[i], y_pred_rule[i]] += 1
        hybrid_cm[y_true[i], y_pred_hybrid[i]] += 1
    
    # Prepare results
    results = {
        'data': df,
        'predictions': {
            'ml': y_pred_ml,
            'rule': y_pred_rule,
            'hybrid': y_pred_hybrid
        },
        'accuracy': {
            'ml': ml_accuracy,
            'rule': rule_accuracy,
            'hybrid': hybrid_accuracy
        },
        'confusion_matrices': {
            'ml': ml_cm,
            'rule': rule_cm,
            'hybrid': hybrid_cm
        },
        'user_context': user_context
    }
    
    return results
def visualize_real_data_results(results):
    """
    Generate visualizations for real data test results
    
    Args:
        results (dict): Test results
    """
    if results is None:
        logger.error("No results to visualize")
        return
    
    # Create output directory
    viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get data
    df = results['data']
    
    # 1. Scatter plot of vital signs colored by risk class
    plt.figure(figsize=(12, 10))
    colors = ['green', 'orange', 'red']
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    for risk_class in [0, 1, 2]:
        subset = df[df['risk_class'] == risk_class]
        plt.scatter(
            subset['heart_rate'], 
            subset['blood_oxygen'], 
            alpha=0.7, 
            label=labels[risk_class],
            color=colors[risk_class]
        )
    
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    plt.title('Distribution of Vital Signs by Risk Class - Real Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'real_data_distribution.png'), dpi=300)
    
    # 2. Confusion matrix heatmaps
    for method, cm in results['confusion_matrices'].items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(f'Confusion Matrix - {method.upper()} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{method}_confusion_matrix_real.png'), dpi=300)
    
    # 3. Model accuracy comparison
    plt.figure(figsize=(10, 6))
    methods = ['Rule-based', 'ML Model', 'Hybrid']
    accuracies = [
        results['accuracy']['rule'],
        results['accuracy']['ml'],
        results['accuracy']['hybrid']
    ]
    plt.bar(methods, accuracies, color=['lightblue', 'lightgreen', 'coral'])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison - Real Data')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'accuracy_comparison_real.png'), dpi=300)
    
    # 4. Timeline view of predictions
    try:
        plt.figure(figsize=(15, 8))
        
        # Timeline of actual risk class
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['risk_class'], 'o-', label='Actual Risk Class')
        plt.yticks([0, 1, 2], labels)
        plt.title('Risk Class Timeline - Actual vs. Predicted')
        plt.ylabel('Actual Risk Class')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Timeline of predicted risk class (hybrid approach)
        plt.subplot(2, 1, 2)
        hybrid_pred = results['predictions']['hybrid']
        plt.plot(df['timestamp'], hybrid_pred, 'o-', label='Hybrid Model Prediction', color='coral')
        plt.yticks([0, 1, 2], labels)
        plt.xlabel('Timestamp')
        plt.ylabel('Predicted Risk Class')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'risk_class_timeline.png'), dpi=300)
    except Exception as e:
        logger.error(f"Error generating timeline plot: {e}")
    
    # 5. Prediction errors timeline
    try:
        plt.figure(figsize=(15, 6))
        hybrid_pred = np.array(results['predictions']['hybrid'])
        y_true = df['risk_class'].values
        
        # Calculate error (0 = correct, 1 = incorrect)
        errors = (hybrid_pred != y_true).astype(int)
        
        # Plot errors over time
        plt.scatter(df['timestamp'], errors, c=['green' if e == 0 else 'red' for e in errors], alpha=0.7)
        plt.yticks([0, 1], ['Correct', 'Incorrect'])
        plt.title('Hybrid Model Prediction Accuracy Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Prediction')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'prediction_errors_timeline.png'), dpi=300)
    except Exception as e:
        logger.error(f"Error generating errors timeline plot: {e}")
    
    logger.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run real data testing"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test classification model with real health data')
    parser.add_argument('--user_id', required=True, help='User ID to test with')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of records to use')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting real data testing for user {args.user_id}...")
    
    # Test model with real data
    results = test_model_with_real_data(args.user_id, args.limit)
    
    if results is not None:
        # Visualize results
        visualize_real_data_results(results)
        
        # Generate summary report
        with open(os.path.join(RESULTS_DIR, 'summary_report.txt'), 'w') as f:
            f.write("CLASSIFICATION MODEL REAL DATA TEST RESULTS\n")
            f.write("===========================================\n\n")
            f.write(f"User ID: {args.user_id}\n")
            f.write(f"Records analyzed: {len(results['data'])}\n\n")
            
            if 'user_context' in results and results['user_context']:
                f.write("User Context:\n")
                for key, value in results['user_context'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("Accuracy Results:\n")
            f.write(f"  ML Model: {results['accuracy']['ml']:.4f}\n")
            f.write(f"  Rule-based: {results['accuracy']['rule']:.4f}\n")
            f.write(f"  Hybrid approach: {results['accuracy']['hybrid']:.4f}\n\n")
            
            best_method = max(results['accuracy'], key=results['accuracy'].get)
            best_accuracy = results['accuracy'][best_method]
            f.write(f"Best performing method: {best_method.upper()} with accuracy of {best_accuracy:.4f}\n\n")
            
            f.write("Distribution of risk classes in data:\n")
            class_counts = results['data']['risk_class'].value_counts().sort_index()
            for idx, count in enumerate(class_counts):
                class_name = ['Low Risk', 'Medium Risk', 'High Risk'][idx]
                percentage = count / len(results['data']) * 100
                f.write(f"  {class_name}: {count} records ({percentage:.1f}%)\n")
        
        logger.info(f"Summary report saved to {os.path.join(RESULTS_DIR, 'summary_report.txt')}")
    
    # Calculate total runtime
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Testing complete in {total_time:.2f} minutes")
    
    if results is not None:
        # Print summary
        print("\n======= TEST SUMMARY =======")
        print(f"User ID: {args.user_id}")
        print(f"Records analyzed: {len(results['data'])}")
        print(f"ML Model Accuracy: {results['accuracy']['ml']:.4f}")
        print(f"Rule-based Accuracy: {results['accuracy']['rule']:.4f}")
        print(f"Hybrid Approach Accuracy: {results['accuracy']['hybrid']:.4f}")
        print(f"Results saved to: {RESULTS_DIR}")
        print("============================\n")

if __name__ == "__main__":
    main()