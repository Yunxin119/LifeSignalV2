"""
Script to generate training data and validate the classification model performance.

This script:
1. Generates synthetic health data for different user profiles
2. Trains classification models with this data
3. Evaluates model performance with test data
4. Creates visualizations of the results

Usage:
python validate_classification_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_validation.log')
    ]
)

logger = logging.getLogger(__name__)

# Make sure the services modules are in the path
sys.path.append('.')

# Import necessary modules
from services.risk_classification import RiskClassification
from services.classification_model import ClassificationModel
from services.health_service import HealthService

# Directory for saving validation results
RESULTS_DIR = "validation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class HealthDataGenerator:
    """Generator for synthetic health data for model training and testing"""
    
    # Health condition definitions
    CONDITIONS = {
        'healthy': {
            'hr_range': (60, 100),
            'bo_range': (95, 100),
            'hr_std': 5,
            'bo_std': 1
        },
        'anxiety': {
            'hr_range': (70, 115),
            'bo_range': (94, 100),
            'hr_std': 10,
            'bo_std': 1.5
        },
        'copd': {
            'hr_range': (65, 105),
            'bo_range': (88, 96),
            'hr_std': 8,
            'bo_std': 2
        },
        'heart_disease': {
            'hr_range': (55, 110),
            'bo_range': (92, 98),
            'hr_std': 12,
            'bo_std': 1.5
        },
        'athlete': {
            'hr_range': (40, 80),
            'bo_range': (95, 100),
            'hr_std': 5,
            'bo_std': 1
        },
        'depression': {
            'hr_range': (60, 100),
            'bo_range': (94, 100),
            'hr_std': 7,
            'bo_std': 1.2
        }
    }
    
    @staticmethod
    def generate_vitals(condition='healthy', n_samples=100, anomaly_rate=0.1):
        """
        Generate synthetic vital sign readings for a given health condition
        
        Args:
            condition (str): Health condition
            n_samples (int): Number of samples to generate
            anomaly_rate (float): Rate of anomalous readings
            
        Returns:
            tuple: (heart_rates, blood_oxygen) arrays
        """
        if condition not in HealthDataGenerator.CONDITIONS:
            condition = 'healthy'
        
        # Get normal ranges for this condition
        config = HealthDataGenerator.CONDITIONS[condition]
        hr_low, hr_high = config['hr_range']
        bo_low, bo_high = config['bo_range']
        hr_std = config['hr_std']
        bo_std = config['bo_std']
        
        # Generate normal distribution for this condition
        normal_samples = int(n_samples * (1 - anomaly_rate))
        anomaly_samples = n_samples - normal_samples
        
        # Generate normal readings
        heart_rates = np.random.normal(
            (hr_low + hr_high) / 2, 
            hr_std, 
            normal_samples
        )
        blood_oxygen = np.random.normal(
            (bo_low + bo_high) / 2, 
            bo_std, 
            normal_samples
        )
        
        # Generate anomalous readings (outside normal range)
        if anomaly_samples > 0:
            # Create anomalous heart rates
            anomaly_hr = np.concatenate([
                np.random.uniform(30, hr_low - 5, anomaly_samples // 2),  # Lower than normal
                np.random.uniform(hr_high + 5, 180, anomaly_samples - anomaly_samples // 2)  # Higher than normal
            ])
            np.random.shuffle(anomaly_hr)
            
            # Create anomalous blood oxygen
            anomaly_bo = np.concatenate([
                np.random.uniform(80, bo_low - 1, anomaly_samples // 2),  # Lower than normal
                np.random.uniform(bo_high + 0.5, 100, anomaly_samples - anomaly_samples // 2)  # Higher than normal (capped at 100)
            ])
            np.random.shuffle(anomaly_bo)
            anomaly_bo = np.clip(anomaly_bo, 80, 100)  # Ensure values are in valid range
            
            # Combine normal and anomalous
            heart_rates = np.append(heart_rates, anomaly_hr)
            blood_oxygen = np.append(blood_oxygen, anomaly_bo)
        
        # Ensure values are in valid range
        heart_rates = np.clip(heart_rates, 30, 180)
        blood_oxygen = np.clip(blood_oxygen, 80, 100)
        
        # Shuffle the data
        indices = np.arange(len(heart_rates))
        np.random.shuffle(indices)
        heart_rates = heart_rates[indices]
        blood_oxygen = blood_oxygen[indices]
        
        return heart_rates, blood_oxygen
    
    @staticmethod
    def generate_dataset(n_per_condition=200, anomaly_rate=0.1):
        """
        Generate a complete dataset with all conditions
        
        Args:
            n_per_condition (int): Number of samples per condition
            anomaly_rate (float): Rate of anomalous readings
            
        Returns:
            pandas.DataFrame: DataFrame with heart_rate, blood_oxygen, condition, and risk_class
        """
        data = []
        
        # Generate data for each condition
        for condition in HealthDataGenerator.CONDITIONS:
            logger.info(f"Generating {n_per_condition} samples for condition: {condition}")
            
            # Generate vitals
            heart_rates, blood_oxygen = HealthDataGenerator.generate_vitals(
                condition=condition,
                n_samples=n_per_condition,
                anomaly_rate=anomaly_rate
            )
            
            # Create user context for this condition
            user_context = {
                'health_conditions': [condition] if condition != 'healthy' else []
            }
            
            # Calculate risk class for each sample
            for hr, bo in zip(heart_rates, blood_oxygen):
                # Use the rule-based risk calculation
                risk_score = HealthService.calculate_risk_score(hr, bo, user_context)
                risk_class = RiskClassification.score_to_class(risk_score)
                
                # Add to dataset
                data.append({
                    'heart_rate': hr,
                    'blood_oxygen': bo,
                    'condition': condition,
                    'has_condition': condition != 'healthy',
                    'risk_score': risk_score,
                    'risk_class': risk_class
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df

def train_and_evaluate_model():
    """Train and evaluate a classification model using synthetic data"""
    # Generate dataset
    logger.info("Generating synthetic health dataset...")
    df = HealthDataGenerator.generate_dataset(n_per_condition=300, anomaly_rate=0.15)
    
    # Save generated dataset
    dataset_path = os.path.join(RESULTS_DIR, "synthetic_health_data.csv")
    df.to_csv(dataset_path, index=False)
    logger.info(f"Dataset saved to {dataset_path}")
    
    # Print dataset summary
    logger.info(f"Dataset shape: {df.shape}")
    logger.info("Class distribution:")
    logger.info(df['risk_class'].value_counts())
    logger.info("Condition distribution:")
    logger.info(df['condition'].value_counts())
    
    # Create features and labels
    X = df[['heart_rate', 'blood_oxygen']].values
    y = df['risk_class'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Testing set: {X_test.shape[0]} samples")
    
    # Initialize condition-specific models
    models = {}
    conditions = df['condition'].unique()
    
    # Train a general model
    logger.info("Training general classification model...")
    from sklearn.ensemble import GradientBoostingClassifier
    general_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    general_model.fit(X_train, y_train)
    
    # Evaluate the general model
    y_pred = general_model.predict(X_test)
    general_accuracy = np.mean(y_pred == y_test)
    logger.info(f"General model accuracy: {general_accuracy:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix:")
    logger.info(cm)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk'])
    logger.info("Classification report:")
    logger.info(report)
    
    # Save the general model
    model_path = os.path.join(RESULTS_DIR, "general_model.pkl")
    joblib.dump(general_model, model_path)
    logger.info(f"General model saved to {model_path}")
    
    # Train condition-specific models
    for condition in conditions:
        logger.info(f"Training model for condition: {condition}")
        
        # Filter data for this condition
        condition_df = df[df['condition'] == condition]
        X_cond = condition_df[['heart_rate', 'blood_oxygen']].values
        y_cond = condition_df['risk_class'].values
        
        # Split into train and test
        X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(
            X_cond, y_cond, test_size=0.3, random_state=42, stratify=y_cond
        )
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train_cond, y_train_cond)
        
        # Evaluate model
        y_pred_cond = model.predict(X_test_cond)
        accuracy = np.mean(y_pred_cond == y_test_cond)
        logger.info(f"Accuracy for {condition}: {accuracy:.4f}")
        
        # Save model
        models[condition] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_test_cond, y_pred_cond)
        }
    
    # Save condition-specific models
    for condition, model_data in models.items():
        model_path = os.path.join(RESULTS_DIR, f"{condition}_model.pkl")
        joblib.dump(model_data['model'], model_path)
    
    # Evaluate the impact of hybrid approach
    logger.info("Evaluating hybrid approach...")
    
    # Calculate rule-based predictions
    rule_predictions = []
    for hr, bo in X_test:
        risk_score = HealthService.calculate_risk_score(hr, bo, None)
        risk_class = RiskClassification.score_to_class(risk_score)
        rule_predictions.append(risk_class)
    
    rule_accuracy = np.mean(np.array(rule_predictions) == y_test)
    logger.info(f"Rule-based accuracy: {rule_accuracy:.4f}")
    
    # Calculate hybrid predictions (simple 50/50 blend for demonstration)
    ml_probas = general_model.predict_proba(X_test)
    
    hybrid_predictions = []
    for i, (hr, bo) in enumerate(X_test):
        # Get rule-based probabilities
        risk_score = HealthService.calculate_risk_score(hr, bo, None)
        rule_probas = RiskClassification.score_to_probabilities(risk_score)
        
        # Blend probabilities (50/50)
        blended_probas = []
        for j in range(3):
            blended_probas.append(0.5 * ml_probas[i, j] + 0.5 * rule_probas[j])
        
        # Get final prediction
        hybrid_pred = np.argmax(blended_probas)
        hybrid_predictions.append(hybrid_pred)
    
    hybrid_accuracy = np.mean(np.array(hybrid_predictions) == y_test)
    logger.info(f"Hybrid approach accuracy: {hybrid_accuracy:.4f}")
    
    # Generate hybrid confusion matrix
    hybrid_cm = confusion_matrix(y_test, hybrid_predictions)
    logger.info("Hybrid confusion matrix:")
    logger.info(hybrid_cm)
    
    # Calculate improvement
    ml_improvement = (general_accuracy - rule_accuracy) * 100
    hybrid_improvement = (hybrid_accuracy - max(rule_accuracy, general_accuracy)) * 100
    
    logger.info(f"ML improvement over rules: {ml_improvement:.2f}%")
    logger.info(f"Hybrid improvement over best individual method: {hybrid_improvement:.2f}%")
    
    # Return all results for visualization
    return {
        'dataset': df,
        'general_model': general_model,
        'condition_models': models,
        'test_data': (X_test, y_test),
        'predictions': {
            'ml': y_pred,
            'rule': rule_predictions,
            'hybrid': hybrid_predictions
        },
        'accuracy': {
            'ml': general_accuracy,
            'rule': rule_accuracy,
            'hybrid': hybrid_accuracy
        },
        'confusion_matrices': {
            'ml': cm,
            'hybrid': hybrid_cm
        }
    }

def generate_visualizations(results):
    """Generate visualizations of model performance"""
    logger.info("Generating visualizations...")
    
    # Create output directory
    viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get data
    df = results['dataset']
    X_test, y_test = results['test_data']
    general_model = results['general_model']
    
    # 1. Scatter plot showing data distribution and risk classes
    plt.figure(figsize=(12, 10))
    for condition in df['condition'].unique():
        subset = df[df['condition'] == condition]
        plt.scatter(
            subset['heart_rate'], 
            subset['blood_oxygen'], 
            alpha=0.5, 
            label=condition
        )
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    plt.title('Distribution of Vital Signs by Health Condition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'vitals_distribution.png'), dpi=300)
    
    # 2. Scatter plot colored by risk class
    plt.figure(figsize=(12, 10))
    colors = ['green', 'orange', 'red']
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    for risk_class in [0, 1, 2]:
        subset = df[df['risk_class'] == risk_class]
        plt.scatter(
            subset['heart_rate'], 
            subset['blood_oxygen'], 
            alpha=0.5, 
            label=labels[risk_class],
            color=colors[risk_class]
        )
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    plt.title('Distribution of Vital Signs by Risk Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'risk_class_distribution.png'), dpi=300)
    
    # 3. Plot decision boundaries
    plt.figure(figsize=(12, 10))
    
    # Create a mesh grid to visualize decision boundaries
    h = 0.5  # step size in the mesh
    x_min, x_max = 30, 180
    y_min, y_max = 80, 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get predictions for each point in the mesh
    Z = general_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn_r)
    
    # Plot the test data points
    for risk_class in [0, 1, 2]:
        idx = y_test == risk_class
        plt.scatter(
            X_test[idx, 0], 
            X_test[idx, 1], 
            alpha=0.8, 
            label=labels[risk_class],
            edgecolors='k'
        )
    
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    plt.title('Model Decision Boundaries')
    plt.legend()
    plt.savefig(os.path.join(viz_dir, 'decision_boundaries.png'), dpi=300)
    
    # 4. Confusion matrix heatmap for general model
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrices']['ml'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix - ML Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'ml_confusion_matrix.png'), dpi=300)
    
    # 5. Confusion matrix heatmap for hybrid approach
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrices']['hybrid'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix - Hybrid Approach')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'hybrid_confusion_matrix.png'), dpi=300)
    
    # 6. Model accuracy comparison
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
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'accuracy_comparison.png'), dpi=300)
    
    # 7. Create visualization for condition-specific accuracy
    condition_accuracy = {}
    for condition, model_data in results['condition_models'].items():
        condition_accuracy[condition] = model_data['accuracy']
    
    plt.figure(figsize=(12, 6))
    conditions = list(condition_accuracy.keys())
    accuracies = list(condition_accuracy.values())
    plt.bar(conditions, accuracies, color=sns.color_palette("husl", len(conditions)))
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.xlabel('Health Condition')
    plt.ylabel('Model Accuracy')
    plt.title('Accuracy by Health Condition')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'condition_accuracy.png'), dpi=300)
    
    logger.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run validation"""
    start_time = datetime.now()
    logger.info("Starting validation process...")
    
    # Train and evaluate models
    results = train_and_evaluate_model()
    
    # Generate visualizations
    generate_visualizations(results)
    
    # Calculate total runtime
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Validation complete in {total_time:.2f} minutes")
    
    # Create summary report
    summary = {
        'ml_accuracy': results['accuracy']['ml'],
        'rule_accuracy': results['accuracy']['rule'],
        'hybrid_accuracy': results['accuracy']['hybrid'],
        'hybrid_improvement': (results['accuracy']['hybrid'] - max(results['accuracy']['ml'], results['accuracy']['rule'])) * 100
    }
    
    # Print summary
    print("\n======= VALIDATION SUMMARY =======")
    print(f"ML Model Accuracy: {summary['ml_accuracy']:.4f}")
    print(f"Rule-based Accuracy: {summary['rule_accuracy']:.4f}")
    print(f"Hybrid Approach Accuracy: {summary['hybrid_accuracy']:.4f}")
    print(f"Hybrid Improvement: {summary['hybrid_improvement']:.2f}%")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=================================\n")

if __name__ == "__main__":
    main()