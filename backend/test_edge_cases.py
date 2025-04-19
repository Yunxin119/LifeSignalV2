"""
Script to thoroughly test edge cases for the health risk classification model.

This script:
1. Tests model performance on extreme vital sign values
2. Evaluates behavior at classification boundaries
3. Analyzes performance across different health conditions
4. Visualizes where misclassifications occur

Usage:
python test_edge_cases.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('edge_case_testing.log')
    ]
)

logger = logging.getLogger(__name__)

# Make sure modules are in the path
sys.path.append('.')

# Import necessary modules
from services.risk_classification import RiskClassification
from services.classification_model import ClassificationModel
from services.health_service import HealthService

# Directory for saving results
RESULTS_DIR = "edge_case_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class EdgeCaseTester:
    """Class for testing edge cases in health risk classification"""
    
    @staticmethod
    def generate_edge_cases():
        """
        Generate a comprehensive set of edge cases for testing
        
        Returns:
            pandas.DataFrame: DataFrame with edge case test data
        """
        data = []
        
        # 1. Boundary value cases
        # These cases are right at or near the boundaries between risk classes
        logger.info("Generating boundary value test cases...")
        
        # Low-medium risk boundary cases (risk score around 30)
        for i in range(10):
            # Generate cases right around the boundary
            hr = np.random.uniform(95, 105)
            bo = np.random.uniform(94, 96)
            
            # Calculate true risk score and class
            risk_score = HealthService.calculate_risk_score(hr, bo, None)
            risk_class = RiskClassification.score_to_class(risk_score)
            
            data.append({
                'heart_rate': hr,
                'blood_oxygen': bo,
                'risk_score': risk_score,
                'risk_class': risk_class,
                'case_type': 'Low-Medium Boundary',
                'condition': 'healthy'
            })
        
        # Medium-high risk boundary cases (risk score around 70)
        for i in range(10):
            # Generate cases right around the boundary
            hr = np.random.uniform(120, 130)
            bo = np.random.uniform(91, 93)
            
            # Calculate true risk score and class
            risk_score = HealthService.calculate_risk_score(hr, bo, None)
            risk_class = RiskClassification.score_to_class(risk_score)
            
            data.append({
                'heart_rate': hr,
                'blood_oxygen': bo,
                'risk_score': risk_score,
                'risk_class': risk_class,
                'case_type': 'Medium-High Boundary',
                'condition': 'healthy'
            })
        
        # 2. Extreme value cases
        logger.info("Generating extreme value test cases...")
        
        # Extreme heart rate cases
        for hr in [25, 35, 180, 200]:
            for bo in [85, 92, 97]:
                # Calculate true risk score and class
                risk_score = HealthService.calculate_risk_score(hr, bo, None)
                risk_class = RiskClassification.score_to_class(risk_score)
                
                data.append({
                    'heart_rate': hr,
                    'blood_oxygen': bo,
                    'risk_score': risk_score,
                    'risk_class': risk_class,
                    'case_type': 'Extreme Heart Rate',
                    'condition': 'healthy'
                })
        
        # Extreme blood oxygen cases
        for bo in [75, 80, 85]:
            for hr in [60, 90, 120]:
                # Calculate true risk score and class
                risk_score = HealthService.calculate_risk_score(hr, bo, None)
                risk_class = RiskClassification.score_to_class(risk_score)
                
                data.append({
                    'heart_rate': hr,
                    'blood_oxygen': bo,
                    'risk_score': risk_score,
                    'risk_class': risk_class,
                    'case_type': 'Extreme Blood Oxygen',
                    'condition': 'healthy'
                })
        
        # 3. Health condition specific cases
        logger.info("Generating condition-specific test cases...")
        
        # Define special cases for various health conditions
        condition_cases = {
            'anxiety': [
                # Anxiety attack cases
                {'heart_rate': 140, 'blood_oxygen': 98, 'case_type': 'Anxiety Attack'},
                {'heart_rate': 155, 'blood_oxygen': 97, 'case_type': 'Anxiety Attack'},
                {'heart_rate': 125, 'blood_oxygen': 96, 'case_type': 'Anxiety Attack'}
            ],
            'copd': [
                # COPD with low blood oxygen
                {'heart_rate': 85, 'blood_oxygen': 88, 'case_type': 'COPD Baseline'},
                {'heart_rate': 95, 'blood_oxygen': 86, 'case_type': 'COPD Exacerbation'},
                {'heart_rate': 110, 'blood_oxygen': 84, 'case_type': 'COPD Exacerbation'}
            ],
            'heart_disease': [
                # Heart disease cases
                {'heart_rate': 45, 'blood_oxygen': 94, 'case_type': 'Bradycardia'},
                {'heart_rate': 160, 'blood_oxygen': 94, 'case_type': 'Tachycardia'},
                {'heart_rate': 50, 'blood_oxygen': 90, 'case_type': 'Heart Disease'}
            ],
            'athlete': [
                # Athletic cases with low heart rate
                {'heart_rate': 42, 'blood_oxygen': 98, 'case_type': 'Athletic Low HR'},
                {'heart_rate': 38, 'blood_oxygen': 97, 'case_type': 'Athletic Very Low HR'},
                {'heart_rate': 48, 'blood_oxygen': 96, 'case_type': 'Athletic Low HR'}
            ],
            'sleep_apnea': [
                # Sleep apnea cases
                {'heart_rate': 65, 'blood_oxygen': 88, 'case_type': 'Sleep Apnea Episode'},
                {'heart_rate': 70, 'blood_oxygen': 85, 'case_type': 'Sleep Apnea Episode'},
                {'heart_rate': 60, 'blood_oxygen': 83, 'case_type': 'Sleep Apnea Episode'}
            ],
            'asthma': [
                # Asthma cases
                {'heart_rate': 100, 'blood_oxygen': 93, 'case_type': 'Asthma Attack'},
                {'heart_rate': 115, 'blood_oxygen': 91, 'case_type': 'Asthma Attack'},
                {'heart_rate': 95, 'blood_oxygen': 92, 'case_type': 'Asthma Attack'}
            ]
        }
        
        # Create test cases for each condition
        for condition, cases in condition_cases.items():
            for case in cases:
                # Create user context with the condition
                user_context = {'health_conditions': [condition]}
                
                # Calculate true risk score and class
                risk_score = HealthService.calculate_risk_score(
                    case['heart_rate'], case['blood_oxygen'], user_context
                )
                risk_class = RiskClassification.score_to_class(risk_score)
                
                # Add to dataset
                data.append({
                    'heart_rate': case['heart_rate'],
                    'blood_oxygen': case['blood_oxygen'],
                    'risk_score': risk_score,
                    'risk_class': risk_class,
                    'case_type': case['case_type'],
                    'condition': condition
                })
        
        # 4. Add paradoxical cases (cases that might be confusing)
        logger.info("Generating paradoxical test cases...")
        
        # Normal heart rate but very low blood oxygen
        data.append({
            'heart_rate': 70,
            'blood_oxygen': 82,
            'risk_score': HealthService.calculate_risk_score(70, 82, None),
            'risk_class': RiskClassification.score_to_class(HealthService.calculate_risk_score(70, 82, None)),
            'case_type': 'Paradoxical',
            'condition': 'healthy',
            'note': 'Normal HR but very low BO'
        })
        
        # Very low heart rate but normal blood oxygen
        data.append({
            'heart_rate': 35,
            'blood_oxygen': 98,
            'risk_score': HealthService.calculate_risk_score(35, 98, None),
            'risk_class': RiskClassification.score_to_class(HealthService.calculate_risk_score(35, 98, None)),
            'case_type': 'Paradoxical',
            'condition': 'healthy',
            'note': 'Very low HR but normal BO'
        })
        
        # Very high heart rate but normal blood oxygen
        data.append({
            'heart_rate': 170,
            'blood_oxygen': 98,
            'risk_score': HealthService.calculate_risk_score(170, 98, None),
            'risk_class': RiskClassification.score_to_class(HealthService.calculate_risk_score(170, 98, None)),
            'case_type': 'Paradoxical',
            'condition': 'healthy',
            'note': 'Very high HR but normal BO'
        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def test_edge_cases(model_path=None):
        """
        Test model performance on edge cases
        
        Args:
            model_path (str, optional): Path to pre-trained model file
            
        Returns:
            dict: Test results
        """
        # Generate edge cases
        df = EdgeCaseTester.generate_edge_cases()
        logger.info(f"Generated {len(df)} edge cases for testing")
        
        # Save dataset
        dataset_path = os.path.join(RESULTS_DIR, "edge_cases.csv")
        df.to_csv(dataset_path, index=False)
        logger.info(f"Edge cases saved to {dataset_path}")
        
        # Load model if provided, otherwise train a new one
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            model = joblib.load(model_path)
        else:
            logger.info("No model provided or model not found. Using random user ID for classification.")
            model = None
        
        # Test each edge case
        results = []
        
        for i, row in df.iterrows():
            # Extract features
            heart_rate = row['heart_rate']
            blood_oxygen = row['blood_oxygen']
            condition = row['condition']
            expected_class = row['risk_class']
            
            # Create user context for condition-specific cases
            user_context = {
                'health_conditions': [condition] if condition != 'healthy' else []
            }
            
            # Get rule-based prediction
            rule_score = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
            rule_class = RiskClassification.score_to_class(rule_score)
            rule_probs = RiskClassification.score_to_probabilities(rule_score)
            
            # Get ML model prediction (if model provided)
            if model:
                ml_class = model.predict([[heart_rate, blood_oxygen]])[0]
                ml_probs = model.predict_proba([[heart_rate, blood_oxygen]])[0].tolist()
            else:
                # Use classification service with a test user ID
                try:
                    ml_result = ClassificationModel.predict_risk_class(
                        "test_user", [heart_rate, blood_oxygen], user_context
                    )
                    ml_class = ml_result['risk_class']
                    ml_probs = [
                        ml_result['probabilities']['low'],
                        ml_result['probabilities']['medium'],
                        ml_result['probabilities']['high']
                    ]
                except Exception as e:
                    logger.warning(f"Error in ML classification: {e}")
                    ml_class = rule_class
                    ml_probs = rule_probs
            
            # Calculate hybrid prediction with 50/50 blend
            hybrid_probs = []
            for j in range(3):
                hybrid_probs.append(0.5 * ml_probs[j] + 0.5 * rule_probs[j])
            hybrid_class = np.argmax(hybrid_probs)
            
            # Store results
            results.append({
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen,
                'condition': condition,
                'case_type': row['case_type'],
                'expected_class': expected_class,
                'rule_class': rule_class,
                'ml_class': ml_class,
                'hybrid_class': hybrid_class,
                'rule_probs': rule_probs,
                'ml_probs': ml_probs,
                'hybrid_probs': hybrid_probs,
                'rule_correct': rule_class == expected_class,
                'ml_correct': ml_class == expected_class,
                'hybrid_correct': hybrid_class == expected_class,
                'note': row.get('note', '')
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy metrics
        rule_accuracy = results_df['rule_correct'].mean()
        ml_accuracy = results_df['ml_correct'].mean()
        hybrid_accuracy = results_df['hybrid_correct'].mean()
        
        logger.info(f"Edge case results:")
        logger.info(f"  Rule accuracy: {rule_accuracy:.4f}")
        logger.info(f"  ML accuracy: {ml_accuracy:.4f}")
        logger.info(f"  Hybrid accuracy: {hybrid_accuracy:.4f}")
        
        # Calculate accuracy by case type
        case_type_accuracy = results_df.groupby('case_type').agg({
            'rule_correct': 'mean',
            'ml_correct': 'mean',
            'hybrid_correct': 'mean'
        })
        
        logger.info("Accuracy by case type:")
        logger.info(case_type_accuracy)
        
        # Calculate accuracy by condition
        condition_accuracy = results_df.groupby('condition').agg({
            'rule_correct': 'mean',
            'ml_correct': 'mean',
            'hybrid_correct': 'mean'
        })
        
        logger.info("Accuracy by condition:")
        logger.info(condition_accuracy)
        
        # Save detailed results
        results_path = os.path.join(RESULTS_DIR, "edge_case_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to {results_path}")
        
        return {
            'data': df,
            'results': results_df,
            'accuracy': {
                'rule': rule_accuracy,
                'ml': ml_accuracy,
                'hybrid': hybrid_accuracy
            },
            'case_type_accuracy': case_type_accuracy,
            'condition_accuracy': condition_accuracy
        }

def visualize_edge_case_results(results):
    """
    Create visualizations for edge case test results
    
    Args:
        results (dict): Test results
    """
    # Create output directory
    viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    results_df = results['results']
    
    # 1. Scatter plot of edge cases colored by case type
    plt.figure(figsize=(14, 10))
    
    for case_type in results_df['case_type'].unique():
        subset = results_df[results_df['case_type'] == case_type]
        plt.scatter(
            subset['heart_rate'],
            subset['blood_oxygen'],
            alpha=0.7,
            label=case_type,
            s=80
        )
    
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    plt.title('Edge Case Distribution by Case Type')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'edge_case_distribution.png'), dpi=300)
    
    # 2. Visualize misclassifications
    plt.figure(figsize=(14, 10))
    
    # Correct classifications
    correct = results_df[results_df['ml_correct']]
    plt.scatter(
        correct['heart_rate'],
        correct['blood_oxygen'],
        color='green',
        alpha=0.7,
        label='Correct Classification',
        marker='o',
        s=80
    )
    
    # Misclassifications
    incorrect = results_df[~results_df['ml_correct']]
    plt.scatter(
        incorrect['heart_rate'],
        incorrect['blood_oxygen'],
        color='red',
        alpha=0.7,
        label='Misclassification',
        marker='x',
        s=100
    )
    
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Blood Oxygen (%)')
    plt.title('ML Model Classification Results on Edge Cases')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'misclassifications.png'), dpi=300)
    
    # 3. Accuracy comparison by method
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
    plt.title('Edge Case Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'edge_case_accuracy.png'), dpi=300)
    
    # 4. Accuracy by case type
    plt.figure(figsize=(14, 8))
    case_type_df = results['case_type_accuracy'].reset_index()
    
    x = np.arange(len(case_type_df))
    width = 0.25
    
    plt.bar(x - width, case_type_df['rule_correct'], width, label='Rule-based', color='lightblue')
    plt.bar(x, case_type_df['ml_correct'], width, label='ML Model', color='lightgreen')
    plt.bar(x + width, case_type_df['hybrid_correct'], width, label='Hybrid', color='coral')
    
    plt.xlabel('Case Type')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Edge Case Type')
    plt.xticks(x, case_type_df['case_type'], rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'accuracy_by_case_type.png'), dpi=300)
    
    # 5. Accuracy by health condition
    plt.figure(figsize=(14, 8))
    condition_df = results['condition_accuracy'].reset_index()
    
    x = np.arange(len(condition_df))
    width = 0.25
    
    plt.bar(x - width, condition_df['rule_correct'], width, label='Rule-based', color='lightblue')
    plt.bar(x, condition_df['ml_correct'], width, label='ML Model', color='lightgreen')
    plt.bar(x + width, condition_df['hybrid_correct'], width, label='Hybrid', color='coral')
    
    plt.xlabel('Health Condition')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Health Condition')
    plt.xticks(x, condition_df['condition'], rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'accuracy_by_condition.png'), dpi=300)
    
    # 6. Confusion matrix for misclassifications
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # ML model confusion matrix
    ml_cm = np.zeros((3, 3), dtype=int)
    for _, row in results_df.iterrows():
        ml_cm[row['expected_class'], row['ml_class']] += 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        ml_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('ML Model Confusion Matrix on Edge Cases')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'ml_edge_confusion_matrix.png'), dpi=300)
    
    # 7. Analyze confidence in predictions
    plt.figure(figsize=(14, 8))
    
    # Calculate confidence as max probability
    results_df['ml_confidence'] = results_df['ml_probs'].apply(lambda x: max(x))
    
    # Group by correct/incorrect prediction
    correct_conf = results_df[results_df['ml_correct']]['ml_confidence']
    incorrect_conf = results_df[~results_df['ml_correct']]['ml_confidence']
    
    # Create histograms
    plt.hist(correct_conf, alpha=0.5, label='Correct Predictions', bins=10, color='green')
    plt.hist(incorrect_conf, alpha=0.5, label='Incorrect Predictions', bins=10, color='red')
    
    plt.xlabel('ML Model Confidence (Max Probability)')
    plt.ylabel('Number of Cases')
    plt.title('Model Confidence Distribution for Correct vs. Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'confidence_analysis.png'), dpi=300)
    
    # 8. Probability distribution across risk classes for all predictions
    for i, label in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
        plt.figure(figsize=(10, 6))
        
        # Extract probabilities for this risk class
        rule_probs = results_df['rule_probs'].apply(lambda x: x[i])
        ml_probs = results_df['ml_probs'].apply(lambda x: x[i])
        hybrid_probs = results_df['hybrid_probs'].apply(lambda x: x[i])
        
        # Plot density curves
        sns.kdeplot(rule_probs, label='Rule-based', color='blue')
        sns.kdeplot(ml_probs, label='ML Model', color='green')
        sns.kdeplot(hybrid_probs, label='Hybrid', color='red')
        
        plt.xlabel(f'Probability of {label}')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution for {label} Classification')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(viz_dir, f'probability_dist_{i}.png'), dpi=300)
    
    logger.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run edge case testing"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test classification model with edge cases')
    parser.add_argument('--model', type=str, default=None, help='Path to pre-trained model file')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info("Starting edge case testing...")
    
    # Test edge cases
    model_path = args.model
    if model_path:
        logger.info(f"Using model from: {model_path}")
    else:
        logger.info("No model specified, using classification service with test user ID")
    
    results = EdgeCaseTester.test_edge_cases(model_path)
    
    # Visualize results
    visualize_edge_case_results(results)
    
    # Calculate total runtime
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Edge case testing complete in {total_time:.2f} minutes")
    
    # Create summary report
    with open(os.path.join(RESULTS_DIR, 'edge_case_summary.txt'), 'w') as f:
        f.write("EDGE CASE TESTING SUMMARY\n")
        f.write("========================\n\n")
        f.write(f"Total edge cases tested: {len(results['results'])}\n\n")
        
        f.write("Overall Accuracy:\n")
        f.write(f"  Rule-based: {results['accuracy']['rule']:.4f}\n")
        f.write(f"  ML Model: {results['accuracy']['ml']:.4f}\n")
        f.write(f"  Hybrid: {results['accuracy']['hybrid']:.4f}\n\n")
        
        f.write("Accuracy by Case Type:\n")
        for idx, row in results['case_type_accuracy'].iterrows():
            f.write(f"  {idx}:\n")
            f.write(f"    Rule-based: {row['rule_correct']:.4f}\n")
            f.write(f"    ML Model: {row['ml_correct']:.4f}\n")
            f.write(f"    Hybrid: {row['hybrid_correct']:.4f}\n")
        f.write("\n")
        
        f.write("Accuracy by Health Condition:\n")
        for idx, row in results['condition_accuracy'].iterrows():
            f.write(f"  {idx}:\n")
            f.write(f"    Rule-based: {row['rule_correct']:.4f}\n")
            f.write(f"    ML Model: {row['ml_correct']:.4f}\n")
            f.write(f"    Hybrid: {row['hybrid_correct']:.4f}\n")
    
    # Print summary
    print("\n======= EDGE CASE TESTING SUMMARY =======")
    print(f"Total edge cases tested: {len(results['results'])}")
    print(f"Rule-based Accuracy: {results['accuracy']['rule']:.4f}")
    print(f"ML Model Accuracy: {results['accuracy']['ml']:.4f}")
    print(f"Hybrid Approach Accuracy: {results['accuracy']['hybrid']:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("========================================\n")

if __name__ == "__main__":
    main()