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
    
    # Enhanced health condition definitions with more conditions
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
        },
        # Added new health conditions
        'asthma': {
            'hr_range': (65, 110),
            'bo_range': (93, 98),
            'hr_std': 8,
            'bo_std': 1.8
        },
        'diabetes': {
            'hr_range': (70, 110),
            'bo_range': (93, 99),
            'hr_std': 7,
            'bo_std': 1.5
        },
        'sleep_apnea': {
            'hr_range': (55, 95),
            'bo_range': (85, 95),
            'hr_std': 10,
            'bo_std': 3
        }
    }
    
    @staticmethod
    def generate_vitals(condition='healthy', n_samples=100, anomaly_rate=0.1, extreme_cases=True):
        """
        Generate synthetic vital sign readings for a given health condition
        
        Args:
            condition (str): Health condition
            n_samples (int): Number of samples to generate
            anomaly_rate (float): Rate of anomalous readings
            extreme_cases (bool): Whether to include extreme edge cases
            
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
        
        # Add extreme edge cases if requested
        if extreme_cases:
            # Add 10 extreme edge cases (about 1% of total if n_samples is 1000)
            n_extreme_cases = max(5, int(n_samples * 0.01))
            
            # Create extreme cases with varying combination levels of severity
            extreme_hr = np.array([
                [25, np.random.uniform(95, 100)],  # Extremely low heart rate with normal oxygen
                [190, np.random.uniform(95, 100)],  # Extremely high heart rate with normal oxygen
                [np.random.uniform(60, 100), 79],  # Normal heart rate with extremely low oxygen
                [25, 85],  # Very low heart rate with low oxygen
                [190, 85],  # Very high heart rate with low oxygen
            ])
            
            # Condition-specific extreme cases
            if condition == 'copd':
                # Even lower blood oxygen levels for COPD patients
                extreme_hr = np.vstack([
                    extreme_hr,
                    [np.random.uniform(60, 80), 82],  # Lower oxygen but not critical
                    [np.random.uniform(90, 120), 80]  # Elevated HR with very low oxygen
                ])
            elif condition == 'anxiety':
                # Higher heart rates during panic attack
                extreme_hr = np.vstack([
                    extreme_hr,
                    [150, np.random.uniform(95, 100)],  # Very high HR during panic
                    [180, np.random.uniform(94, 98)]    # Extreme HR spike
                ])
            elif condition == 'heart_disease':
                # Arrhythmia patterns
                extreme_hr = np.vstack([
                    extreme_hr,
                    [35, np.random.uniform(90, 95)],   # Bradycardia
                    [170, np.random.uniform(90, 95)]   # Tachycardia
                ])
            
            # Add more extreme cases randomly until we have n_extreme_cases
            while len(extreme_hr) < n_extreme_cases:
                # Random extreme case: either HR extreme or BO extreme or both
                case_type = np.random.choice(['hr', 'bo', 'both'])
                if case_type == 'hr':
                    extreme_hr = np.vstack([
                        extreme_hr,
                        [np.random.choice([np.random.uniform(25, 40), np.random.uniform(160, 190)]),
                         np.random.uniform(92, 100)]
                    ])
                elif case_type == 'bo':
                    extreme_hr = np.vstack([
                        extreme_hr,
                        [np.random.uniform(60, 100),
                         np.random.uniform(80, 88)]
                    ])
                else:  # both
                    extreme_hr = np.vstack([
                        extreme_hr,
                        [np.random.choice([np.random.uniform(25, 40), np.random.uniform(160, 190)]),
                         np.random.uniform(80, 88)]
                    ])
            
            # Combine with main dataset
            heart_rates = np.append(heart_rates, extreme_hr[:, 0])
            blood_oxygen = np.append(blood_oxygen, extreme_hr[:, 1])
        
        # Ensure values are in valid range
        heart_rates = np.clip(heart_rates, 25, 200)  # Expanded to allow more extreme values
        blood_oxygen = np.clip(blood_oxygen, 75, 100)  # Expanded to allow more extreme values
        
        # Shuffle the data
        indices = np.arange(len(heart_rates))
        np.random.shuffle(indices)
        heart_rates = heart_rates[indices]
        blood_oxygen = blood_oxygen[indices]
        
        return heart_rates, blood_oxygen
    
    @staticmethod
    def generate_clinical_scenarios():
        """
        Generate specific clinical scenarios for testing
        
        Returns:
            list: List of scenario dictionaries
        """
        scenarios = []
        
        # Scenario 1: Anxiety attack
        scenarios.append({
            'name': 'Anxiety Attack',
            'heart_rate': 135,
            'blood_oxygen': 97,
            'condition': 'anxiety',
            'expected_class': 1  # Medium risk
        })
        
        # Scenario 2: COPD exacerbation
        scenarios.append({
            'name': 'COPD Exacerbation',
            'heart_rate': 95,
            'blood_oxygen': 87,
            'condition': 'copd',
            'expected_class': 2  # High risk
        })
        
        # Scenario 3: Athletic low heart rate
        scenarios.append({
            'name': 'Athletic Low Heart Rate',
            'heart_rate': 45,
            'blood_oxygen': 98,
            'condition': 'athlete',
            'expected_class': 0  # Low risk
        })
        
        # Scenario 4: Heart arrhythmia
        scenarios.append({
            'name': 'Cardiac Arrhythmia',
            'heart_rate': 145,
            'blood_oxygen': 94,
            'condition': 'heart_disease',
            'expected_class': 2  # High risk
        })
        
        # Scenario 5: Diabetic with normal vitals
        scenarios.append({
            'name': 'Diabetic Normal Vitals',
            'heart_rate': 75,
            'blood_oxygen': 96,
            'condition': 'diabetes',
            'expected_class': 0  # Low risk
        })
        
        # Scenario 6: Sleep apnea episode
        scenarios.append({
            'name': 'Sleep Apnea Episode',
            'heart_rate': 65,
            'blood_oxygen': 87,
            'condition': 'sleep_apnea',
            'expected_class': 2  # High risk
        })
        
        # Scenario 7: Asthma attack
        scenarios.append({
            'name': 'Asthma Attack',
            'heart_rate': 110,
            'blood_oxygen': 92,
            'condition': 'asthma',
            'expected_class': 1  # Medium risk
        })
        
        # Scenario 8: Critical condition (very low HR and BO)
        scenarios.append({
            'name': 'Critical Condition',
            'heart_rate': 30,
            'blood_oxygen': 80,
            'condition': 'heart_disease',
            'expected_class': 2  # High risk
        })
        
        # Scenario 9: Borderline case (edge between medium and high risk)
        scenarios.append({
            'name': 'Borderline Medium-High',
            'heart_rate': 125,
            'blood_oxygen': 92,
            'condition': 'healthy',
            'expected_class': 1  # Medium risk, but close to high
        })
        
        # Scenario 10: Borderline case (edge between low and medium risk)
        scenarios.append({
            'name': 'Borderline Low-Medium',
            'heart_rate': 102,
            'blood_oxygen': 96,
            'condition': 'healthy',
            'expected_class': 0  # Low risk, but close to medium
        })
        
        return scenarios
    
    @staticmethod
    def generate_time_series_scenarios():
        """
        Generate time series data for testing progression of conditions
        
        Returns:
            list: List of time series scenario dictionaries
        """
        scenarios = []
        
        # Scenario 1: COPD patient gradually worsening
        for day in range(10):
            # Each day, condition worsens slightly
            bo_decline = day * 0.5  # Blood oxygen declines by 0.5% per day
            hr_increase = day * 2   # Heart rate increases by 2 BPM per day
            
            scenarios.append({
                'series_id': 'copd_decline',
                'day': day,
                'heart_rate': 75 + hr_increase,
                'blood_oxygen': 94 - bo_decline,
                'condition': 'copd',
                'notes': 'Progressive decline'
            })
        
        # Scenario 2: Anxiety attack and recovery
        for hour in range(24):
            if hour < 2:
                # Acute anxiety attack
                scenarios.append({
                    'series_id': 'anxiety_episode',
                    'hour': hour,
                    'heart_rate': 130 - (hour * 5),
                    'blood_oxygen': 97,
                    'condition': 'anxiety',
                    'notes': 'Acute attack'
                })
            elif hour < 6:
                # Recovery phase
                scenarios.append({
                    'series_id': 'anxiety_episode',
                    'hour': hour,
                    'heart_rate': 120 - (hour * 8),
                    'blood_oxygen': 97,
                    'condition': 'anxiety',
                    'notes': 'Recovery phase'
                })
            else:
                # Stabilized
                scenarios.append({
                    'series_id': 'anxiety_episode',
                    'hour': hour,
                    'heart_rate': 75 + np.random.randint(-5, 5),
                    'blood_oxygen': 97 + np.random.randint(-1, 1),
                    'condition': 'anxiety',
                    'notes': 'Stabilized'
                })
        
        # Scenario 3: Sleep apnea during night
        for hour in range(8):
            if hour in [2, 5]:
                # Apnea episodes
                scenarios.append({
                    'series_id': 'sleep_apnea_night',
                    'hour': hour,
                    'heart_rate': 60 + np.random.randint(-5, 5),
                    'blood_oxygen': 85 + np.random.randint(-3, 2),
                    'condition': 'sleep_apnea',
                    'notes': 'Apnea episode'
                })
            else:
                # Normal sleep
                scenarios.append({
                    'series_id': 'sleep_apnea_night',
                    'hour': hour,
                    'heart_rate': 60 + np.random.randint(-5, 5),
                    'blood_oxygen': 94 + np.random.randint(-2, 2),
                    'condition': 'sleep_apnea',
                    'notes': 'Normal sleep'
                })
        
        return scenarios
    
    @staticmethod
    def generate_dataset(n_per_condition=200, anomaly_rate=0.1, extreme_cases=True):
        """
        Generate a complete dataset with all conditions
        
        Args:
            n_per_condition (int): Number of samples per condition
            anomaly_rate (float): Rate of anomalous readings
            extreme_cases (bool): Whether to include extreme edge cases
            
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
                anomaly_rate=anomaly_rate,
                extreme_cases=extreme_cases
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
        
        # Add specific clinical scenarios
        clinical_scenarios = HealthDataGenerator.generate_clinical_scenarios()
        for scenario in clinical_scenarios:
            # Create user context
            user_context = {
                'health_conditions': [scenario['condition']] if scenario['condition'] != 'healthy' else []
            }
            
            # Calculate risk score
            risk_score = HealthService.calculate_risk_score(
                scenario['heart_rate'], scenario['blood_oxygen'], user_context
            )
            
            # Add to dataset
            data.append({
                'heart_rate': scenario['heart_rate'],
                'blood_oxygen': scenario['blood_oxygen'],
                'condition': scenario['condition'],
                'has_condition': scenario['condition'] != 'healthy',
                'risk_score': risk_score,
                'risk_class': scenario['expected_class'],
                'scenario': scenario['name']
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df

def train_and_evaluate_model(n_samples_per_condition=500, anomaly_rate=0.25):
    """
    Train and evaluate a classification model using synthetic data
    
    Args:
        n_samples_per_condition (int): Number of samples per health condition
        anomaly_rate (float): Rate of anomalous readings
        
    Returns:
        dict: Evaluation results
    """
    # Generate dataset
    logger.info("Generating enhanced synthetic health dataset...")
    logger.info(f"Using {n_samples_per_condition} samples per condition with {anomaly_rate*100}% anomalies")
    
    df = HealthDataGenerator.generate_dataset(
        n_per_condition=n_samples_per_condition, 
        anomaly_rate=anomaly_rate,
        extreme_cases=True
    )
    
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
    
    # Test different blend ratios for hybrid approach
    blend_ratios = [0.3, 0.5, 0.7]
    hybrid_results = {}
    
    for ratio in blend_ratios:
        logger.info(f"Testing hybrid approach with ML ratio: {ratio:.1f}")
        
        ml_probas = general_model.predict_proba(X_test)
        hybrid_predictions = []
        
        for i, (hr, bo) in enumerate(X_test):
            # Get rule-based probabilities
            risk_score = HealthService.calculate_risk_score(hr, bo, None)
            rule_probas = RiskClassification.score_to_probabilities(risk_score)
            
            # Blend probabilities
            blended_probas = []
            for j in range(3):
                blended_probas.append(ratio * ml_probas[i, j] + (1 - ratio) * rule_probas[j])
            
            # Get final prediction
            hybrid_pred = np.argmax(blended_probas)
            hybrid_predictions.append(hybrid_pred)
        
        # Calculate accuracy
        accuracy = np.mean(np.array(hybrid_predictions) == y_test)
        logger.info(f"Hybrid accuracy (ratio {ratio:.1f}): {accuracy:.4f}")
        
        hybrid_results[ratio] = {
            'accuracy': accuracy,
            'predictions': hybrid_predictions,
            'confusion_matrix': confusion_matrix(y_test, hybrid_predictions)
        }
    
    # Find best blend ratio
    best_ratio = max(hybrid_results, key=lambda r: hybrid_results[r]['accuracy'])
    best_hybrid_accuracy = hybrid_results[best_ratio]['accuracy']
    logger.info(f"Best hybrid ratio: {best_ratio:.1f} with accuracy: {best_hybrid_accuracy:.4f}")
    
    # Generate hybrid confusion matrix using best ratio
    hybrid_cm = hybrid_results[best_ratio]['confusion_matrix']
    logger.info("Hybrid confusion matrix (best ratio):")
    logger.info(hybrid_cm)
    
    # Evaluate edge cases specifically
    logger.info("Evaluating performance on edge cases...")
    
    # Extract clinical scenarios from dataset
    if 'scenario' in df.columns:
        scenario_df = df[df['scenario'].notna()]
        
        for scenario in scenario_df['scenario'].unique():
            scenario_data = scenario_df[scenario_df['scenario'] == scenario]
            X_scenario = scenario_data[['heart_rate', 'blood_oxygen']].values
            y_scenario = scenario_data['risk_class'].values
            
            # Predict with ML model
            y_pred_ml = general_model.predict(X_scenario)
            ml_correct = np.mean(y_pred_ml == y_scenario)
            
            # Predict with rule-based approach
            y_pred_rule = []
            for i in range(len(X_scenario)):
                hr = X_scenario[i][0]  # First column is heart rate
                bo = X_scenario[i][1]  # Second column is blood oxygen
                risk_score = HealthService.calculate_risk_score(hr, bo, None)
                risk_class = RiskClassification.score_to_class(risk_score)
                y_pred_rule.append(risk_class)
            rule_correct = np.mean(np.array(y_pred_rule) == y_scenario)
            
            # Predict with hybrid approach (best ratio)
            ml_probas = general_model.predict_proba(X_scenario)
            y_pred_hybrid = []
            
            for i, (hr, bo) in enumerate(X_scenario):
                risk_score = HealthService.calculate_risk_score(hr, bo, None)
                rule_probas = RiskClassification.score_to_probabilities(risk_score)
                
                blended_probas = []
                for j in range(3):
                    blended_probas.append(best_ratio * ml_probas[i, j] + (1 - best_ratio) * rule_probas[j])
                
                hybrid_pred = np.argmax(blended_probas)
                y_pred_hybrid.append(hybrid_pred)
            
            hybrid_correct = np.mean(np.array(y_pred_hybrid) == y_scenario)
            
            logger.info(f"Scenario: {scenario}")
            logger.info(f"  ML accuracy: {ml_correct:.4f}")
            logger.info(f"  Rule accuracy: {rule_correct:.4f}")
            logger.info(f"  Hybrid accuracy: {hybrid_correct:.4f}")
    
    # Calculate improvement
    ml_improvement = (general_accuracy - rule_accuracy) * 100
    hybrid_improvement = (best_hybrid_accuracy - max(rule_accuracy, general_accuracy)) * 100
    
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
            'hybrid': hybrid_results[best_ratio]['predictions']
        },
        'accuracy': {
            'ml': general_accuracy,
            'rule': rule_accuracy,
            'hybrid': best_hybrid_accuracy
        },
        'confusion_matrices': {
            'ml': cm,
            'hybrid': hybrid_cm
        },
        'hybrid_results': hybrid_results,
        'best_hybrid_ratio': best_ratio
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
    plt.figure(figsize=(14, 10))
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
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'vitals_distribution.png'), dpi=300)
    
    # 2. Scatter plot colored by risk class
    plt.figure(figsize=(14, 10))
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
    plt.figure(figsize=(14, 10))
    
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
    
    plt.figure(figsize=(14, 6))
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
    
    # 8. Hybrid blend ratio comparison
    if 'hybrid_results' in results:
        plt.figure(figsize=(10, 6))
        ratios = list(results['hybrid_results'].keys())
        accuracies = [results['hybrid_results'][r]['accuracy'] for r in ratios]
        
        plt.plot(ratios, accuracies, 'o-', linewidth=2, markersize=8)
        for i, (r, a) in enumerate(zip(ratios, accuracies)):
            plt.text(r, a + 0.01, f"{a:.3f}", ha='center')
        
        # Add horizontal lines for pure ML and rule accuracies
        plt.axhline(y=results['accuracy']['ml'], color='green', linestyle='--', 
                    label=f"ML: {results['accuracy']['ml']:.3f}")
        plt.axhline(y=results['accuracy']['rule'], color='blue', linestyle='--',
                    label=f"Rule: {results['accuracy']['rule']:.3f}")
        
        plt.xlabel('ML Weight in Hybrid Blend')
        plt.ylabel('Accuracy')
        plt.title('Hybrid Approach Accuracy by Blend Ratio')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(viz_dir, 'hybrid_ratio_comparison.png'), dpi=300)
    
    # 9. Plot distribution of edge cases and their predictions
    if 'scenario' in df.columns:
        scenario_df = df[df['scenario'].notna()]
        
        plt.figure(figsize=(14, 10))
        for scenario in scenario_df['scenario'].unique():
            subset = scenario_df[scenario_df['scenario'] == scenario]
            plt.scatter(
                subset['heart_rate'], 
                subset['blood_oxygen'], 
                alpha=0.8, 
                label=scenario,
                marker='*',
                s=150,  # Larger marker size for scenarios
                edgecolors='black'
            )
        
        # Add decision boundary background
        plt.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.RdYlGn_r)
        
        plt.xlabel('Heart Rate (BPM)')
        plt.ylabel('Blood Oxygen (%)')
        plt.title('Clinical Scenarios Distribution over Decision Boundaries')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'clinical_scenarios.png'), dpi=300)
    
    logger.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run validation"""
    start_time = datetime.now()
    logger.info("Starting enhanced validation process...")
    
    # Train and evaluate models with increased samples and anomalies
    results = train_and_evaluate_model(n_samples_per_condition=500, anomaly_rate=0.25)
    
    # Generate visualizations
    generate_visualizations(results)
    
    # Calculate total runtime
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Enhanced validation complete in {total_time:.2f} minutes")
    
    # Create summary report
    summary = {
        'ml_accuracy': results['accuracy']['ml'],
        'rule_accuracy': results['accuracy']['rule'],
        'hybrid_accuracy': results['accuracy']['hybrid'],
        'best_hybrid_ratio': results['best_hybrid_ratio'],
        'hybrid_improvement': (results['accuracy']['hybrid'] - max(results['accuracy']['ml'], results['accuracy']['rule'])) * 100
    }
    
    # Save summary to file
    with open(os.path.join(RESULTS_DIR, 'validation_summary.txt'), 'w') as f:
        f.write("ENHANCED VALIDATION SUMMARY\n")
        f.write("==========================\n\n")
        f.write(f"ML Model Accuracy: {summary['ml_accuracy']:.4f}\n")
        f.write(f"Rule-based Accuracy: {summary['rule_accuracy']:.4f}\n")
        f.write(f"Hybrid Approach Accuracy: {summary['hybrid_accuracy']:.4f}\n")
        f.write(f"Best Hybrid Ratio: {summary['best_hybrid_ratio']:.2f}\n")
        f.write(f"Hybrid Improvement: {summary['hybrid_improvement']:.2f}%\n\n")
        
        # Add condition-specific accuracies
        f.write("Accuracy by Health Condition:\n")
        for condition, model_data in results['condition_models'].items():
            f.write(f"  {condition}: {model_data['accuracy']:.4f}\n")
    
    # Print summary
    print("\n======= ENHANCED VALIDATION SUMMARY =======")
    print(f"ML Model Accuracy: {summary['ml_accuracy']:.4f}")
    print(f"Rule-based Accuracy: {summary['rule_accuracy']:.4f}")
    print(f"Hybrid Approach Accuracy: {summary['hybrid_accuracy']:.4f}")
    print(f"Best Hybrid Ratio: {summary['best_hybrid_ratio']:.2f}")
    print(f"Hybrid Improvement: {summary['hybrid_improvement']:.2f}%")
    print(f"Results saved to: {RESULTS_DIR}")
    print("===========================================\n")

if __name__ == "__main__":
    main()