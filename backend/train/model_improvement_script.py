"""
Health ML Model Improvement Tool

This script improves the health risk prediction model based on test results
and implements condition-specific adjustments.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import joblib
from datetime import datetime
import json

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required project modules
from services.health_service import HealthService
from services.health_ml_service import HealthMLService
from services.feature_engineering import FeatureEngineering
from train.data_simulator import HealthDataSimulator

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Move HealthRiskEnsemble outside of the ModelImprover class
class HealthRiskEnsemble:
    """Ensemble model for health risk prediction"""
    
    def __init__(self, models, feature_engineering=None, rule_weight=0.3):
        self.models = models
        self.feature_engineering = feature_engineering
        self.rule_weight = rule_weight
    
    def predict(self, heart_rate, blood_oxygen, user_context=None):
        """Make ensemble prediction"""
        # Get rule-based prediction
        rule_risk = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
        
        # Extract features
        basic_features = [heart_rate, blood_oxygen]
        
        if self.feature_engineering:
            enhanced_features = self.feature_engineering.extract_features(
                heart_rate, blood_oxygen, None, user_context
            )
        else:
            enhanced_features = None
        
        # Get ML predictions
        ml_predictions = []
        
        for model_name, model in self.models.items():
            # Determine which features to use
            if "enhanced" in model_name and enhanced_features is not None:
                features = np.array([enhanced_features])
            else:
                features = np.array([basic_features])
            
            # Make prediction
            try:
                prediction = model.predict(features)[0]
                ml_predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
        
        # Calculate average ML prediction
        if ml_predictions:
            ml_risk = np.mean(ml_predictions)
        else:
            ml_risk = rule_risk
        
        # Calculate ensemble prediction
        ensemble_risk = (ml_risk * (1 - self.rule_weight)) + (rule_risk * self.rule_weight)
        
        return ensemble_risk

class ModelImprover:
    """Class to improve health ML models"""
    
    def __init__(self, output_dir="improved_models"):
        """Initialize the model improver with an output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory if it doesn't exist
        os.makedirs(HealthMLService.MODEL_DIR, exist_ok=True)
    
    def generate_training_data(self, samples_per_condition=200, samples_per_age_group=100):
        """Generate comprehensive training data across conditions and age groups"""
        logger.info("Generating comprehensive training dataset")
        
        all_data = []
        
        # Common health conditions
        conditions = [
            "Hypertension", "Type 2 Diabetes", "Asthma", "COPD", 
            "Heart Disease", "Arrhythmia", "Anxiety", "Depression",
            "Obesity", "Sleep Apnea", "Hypothyroidism", "Anemia"
        ]
        
        # Generate data for each condition
        for condition in conditions:
            logger.info(f"Generating data for condition: {condition}")
            
            # Create profiles with this condition at different ages
            for age_group in [(20, 40), (41, 65), (66, 85)]:
                for _ in range(3):  # 3 users per age group per condition
                    age = np.random.randint(age_group[0], age_group[1])
                    user_profile = {
                        'age': age,
                        'health_conditions': [condition]
                    }
                    
                    # Generate data
                    data = self._generate_condition_specific_data(
                        user_profile, 
                        samples=samples_per_condition // 9,  # Divide by 3 age groups * 3 users
                        abnormal_prob=0.3
                    )
                    
                    # Add user context to each record
                    for record in data:
                        record['user_context'] = user_profile.copy()
                    
                    all_data.extend(data)
        
        # Generate data for combined conditions
        combined_conditions = [
            ["Hypertension", "Type 2 Diabetes"],
            ["Anxiety", "Depression"],
            ["COPD", "Sleep Apnea"],
            ["Heart Disease", "Hypertension"],
            ["Obesity", "Type 2 Diabetes"]
        ]
        
        for conditions_list in combined_conditions:
            logger.info(f"Generating data for combined conditions: {', '.join(conditions_list)}")
            
            for age_group in [(20, 40), (41, 65), (66, 85)]:
                age = np.random.randint(age_group[0], age_group[1])
                user_profile = {
                    'age': age,
                    'health_conditions': conditions_list
                }
                
                # Generate data
                data = self._generate_condition_specific_data(
                    user_profile, 
                    samples=samples_per_condition // 3,  # Divide by 3 age groups
                    abnormal_prob=0.3
                )
                
                # Add user context to each record
                for record in data:
                    record['user_context'] = user_profile.copy()
                
                all_data.extend(data)
        
        # Generate data for different age groups without conditions
        for age_group in [(13, 17), (18, 30), (31, 45), (46, 60), (61, 75), (76, 90)]:
            logger.info(f"Generating data for age group: {age_group[0]}-{age_group[1]}")
            
            for _ in range(3):  # 3 users per age group
                age = np.random.randint(age_group[0], age_group[1])
                user_profile = {
                    'age': age,
                    'health_conditions': []
                }
                
                # Generate data
                data = self._generate_condition_specific_data(
                    user_profile, 
                    samples=samples_per_age_group // 3,  # Divide by 3 users
                    abnormal_prob=0.2
                )
                
                # Add user context to each record
                for record in data:
                    record['user_context'] = user_profile.copy()
                
                all_data.extend(data)
        
        logger.info(f"Generated {len(all_data)} training samples")
        
        # Calculate risk scores
        logger.info("Calculating risk scores for training data")
        for record in all_data:
            # Extract vitals
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            
            # Calculate risk score
            risk_score = HealthService.calculate_risk_score(
                heart_rate, 
                blood_oxygen, 
                record.get('user_context')
            )
            
            # Add risk score to record
            record['risk_score'] = risk_score
        
        return all_data
    
    def _generate_condition_specific_data(self, user_profile, samples=100, abnormal_prob=0.3):
        """Generate condition-specific test data"""
        # Calculate days needed based on samples and avg readings per day
        days_needed = max(10, samples // 3 + 1)  # Assume ~3 readings per day
        
        # Generate timeline with specific user profile
        simulation_params = self._get_simulation_params_for_condition(user_profile)
        
        # Check if enhanced simulation is supported
        if hasattr(HealthDataSimulator, 'generate_enhanced_health_timeline') and simulation_params:
            timeline = HealthDataSimulator.generate_enhanced_health_timeline(
                user_profile,
                days=days_needed,
                abnormal_prob=abnormal_prob,
                simulation_params=simulation_params
            )
        else:
            # Fallback to standard method
            timeline = HealthDataSimulator.generate_health_timeline(
                user_profile,
                days=days_needed,
                abnormal_prob=abnormal_prob
            )
        
        # Ensure we have enough samples
        if len(timeline) > samples:
            # Randomly select the required number of samples
            indices = np.random.choice(len(timeline), samples, replace=False)
            timeline = [timeline[i] for i in indices]
        
        return timeline
    
    def _get_simulation_params_for_condition(self, user_profile):
        """Get condition-specific simulation parameters"""
        simulation_params = {}
        
        if 'health_conditions' not in user_profile or not user_profile['health_conditions']:
            return simulation_params
        
        health_conditions = [c.lower() for c in user_profile['health_conditions']]
        health_conditions_text = " ".join(health_conditions)
        
        # Anxiety adjustments
        if any(term in health_conditions_text for term in ['anxiety', 'panic disorder', 'stress disorder']):
            simulation_params['hr_variability_factor'] = 1.5  # More heart rate variability
            simulation_params['hr_baseline_shift'] = 10  # Higher baseline heart rate
            simulation_params['anxiety_episodes'] = True  # Generate occasional episodes of high HR
        
        # COPD adjustments
        if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis']):
            simulation_params['bo_variability_factor'] = 1.5  # More blood oxygen variability
            simulation_params['bo_baseline_shift'] = -3  # Lower baseline blood oxygen
            simulation_params['altitude_sensitive'] = True  # More affected by environmental factors
            
        # Athlete adjustments
        if any(term in health_conditions_text for term in ['athlete', 'athletic']):
            simulation_params['hr_baseline_shift'] = -10  # Lower resting heart rate
            simulation_params['recovery_factor'] = 1.5  # Better recovery from exertion
            
        # Diabetes adjustments
        if any(term in health_conditions_text for term in ['diabetes', 'diabetic']):
            simulation_params['glucose_related_fluctuations'] = True  # Heart rate affected by glucose
            simulation_params['hr_variability_factor'] = 1.3  # More heart rate variability
            
        # Heart condition adjustments
        if any(term in health_conditions_text for term in ['heart disease', 'hypertension', 'arrhythmia']):
            simulation_params['arrhythmia_episodes'] = True  # Occasional irregular patterns
            simulation_params['stress_sensitivity'] = 1.5  # More sensitive to stress factors
        
        return simulation_params
    
    def train_improved_models(self, training_data=None):
        """Train improved models for health risk prediction"""
        if training_data is None:
            logger.info("No training data provided, generating new data")
            training_data = self.generate_training_data()
        
        logger.info(f"Training improved models with {len(training_data)} samples")
        
        # Create features and targets
        X_basic = []  # Just heart rate and blood oxygen
        X_enhanced = []  # All features
        y = []
        
        for record in training_data:
            # Extract basic vitals
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            
            # Extract user context
            user_context = record.get('user_context')
            
            # Extract features
            all_features = FeatureEngineering.extract_features(
                heart_rate,
                blood_oxygen,
                None,  # No additional metrics
                user_context
            )
            
            # Add to training data
            X_basic.append([heart_rate, blood_oxygen])
            X_enhanced.append(all_features)
            y.append(record['risk_score'])
        
        # Convert to numpy arrays
        X_basic = np.array(X_basic)
        X_enhanced = np.array(X_enhanced)
        y = np.array(y)
        
        # Train models
        models = self._train_model_variants(X_basic, X_enhanced, y)
        
        # Save trained models
        for model_name, model in models.items():
            model_path = os.path.join(self.output_dir, f"{model_name}_{self.timestamp}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
            
            # Save default model if it's the basic model
            if model_name == "basic_model":
                default_path = os.path.join(HealthMLService.MODEL_DIR, "default_model.pkl")
                joblib.dump(model, default_path)
                logger.info(f"Saved default model to {default_path}")
        
        return models
    
    def _train_model_variants(self, X_basic, X_enhanced, y):
        """Train different model variants"""
        models = {}
        
        # Basic GradientBoostingRegressor with default parameters
        logger.info("Training basic gradient boosting model")
        basic_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        basic_model.fit(X_basic, y)
        models["basic_model"] = basic_model
        
        # Enhanced GradientBoostingRegressor with all features
        if X_enhanced.shape[1] > X_basic.shape[1]:
            logger.info("Training enhanced gradient boosting model")
            enhanced_model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                random_state=42
            )
            enhanced_model.fit(X_enhanced, y)
            models["enhanced_model"] = enhanced_model
        
        # RandomForest model
        logger.info("Training random forest model")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X_basic, y)
        models["random_forest_model"] = rf_model
        
        # Enhanced RandomForest with all features
        if X_enhanced.shape[1] > X_basic.shape[1]:
            logger.info("Training enhanced random forest model")
            rf_enhanced_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=6,
                random_state=42
            )
            rf_enhanced_model.fit(X_enhanced, y)
            models["random_forest_enhanced_model"] = rf_enhanced_model
        
        return models
    
    def evaluate_models(self, models, test_data=None, visualize=True):
        """Evaluate the trained models on test data"""
        if test_data is None:
            logger.info("No test data provided, generating test data")
            # Generate smaller test set with different conditions
            conditions = [
                "Hypertension", "Asthma", "COPD", "Heart Disease", 
                "Arrhythmia", "Anxiety", "Depression"
            ]
            
            test_data = []
            for condition in conditions:
                # Create profiles with this condition
                user_profile = {
                    'age': np.random.randint(30, 70),
                    'health_conditions': [condition]
                }
                
                # Generate data
                condition_data = self._generate_condition_specific_data(
                    user_profile, 
                    samples=50,
                    abnormal_prob=0.3
                )
                
                # Add user context
                for record in condition_data:
                    record['user_context'] = user_profile.copy()
                
                test_data.extend(condition_data)
            
            # Add data without conditions
            for age in [20, 40, 60, 80]:
                user_profile = {
                    'age': age,
                    'health_conditions': []
                }
                
                # Generate data
                age_data = self._generate_condition_specific_data(
                    user_profile, 
                    samples=25,
                    abnormal_prob=0.2
                )
                
                # Add user context
                for record in age_data:
                    record['user_context'] = user_profile.copy()
                
                test_data.extend(age_data)
            
            # Calculate risk scores
            for record in test_data:
                # Extract vitals
                heart_rate = record['heart_rate']
                blood_oxygen = record['blood_oxygen']
                
                # Calculate risk score
                risk_score = HealthService.calculate_risk_score(
                    heart_rate, 
                    blood_oxygen, 
                    record.get('user_context')
                )
                
                # Add risk score to record
                record['risk_score'] = risk_score
        
        logger.info(f"Evaluating models on {len(test_data)} test samples")
        
        # Prepare test data
        X_basic_test = []
        X_enhanced_test = []
        y_test = []
        
        for record in test_data:
            # Extract basic vitals
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            
            # Extract user context
            user_context = record.get('user_context')
            
            # Extract features
            all_features = FeatureEngineering.extract_features(
                heart_rate,
                blood_oxygen,
                None,  # No additional metrics
                user_context
            )
            
            # Add to test data
            X_basic_test.append([heart_rate, blood_oxygen])
            X_enhanced_test.append(all_features)
            y_test.append(record['risk_score'])
        
        # Convert to numpy arrays
        X_basic_test = np.array(X_basic_test)
        X_enhanced_test = np.array(X_enhanced_test)
        y_test = np.array(y_test)
        
        # Evaluate each model
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            # Determine which features to use
            if "enhanced" in model_name:
                X_test = X_enhanced_test
            else:
                X_test = X_basic_test
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store results
            results[model_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'predictions': y_pred
            }
            
            logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Visualize results
        if visualize:
            self._visualize_model_comparison(results, y_test, test_data, X_basic_test)
        
        return results
    
    def _visualize_model_comparison(self, results, y_true, test_data, X_test):
        """Visualize model comparison results"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Comparison", fontsize=16)
        
        # Bar chart of model performance metrics
        model_names = list(results.keys())
        mae_values = [results[model]['mae'] for model in model_names]
        rmse_values = [results[model]['rmse'] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, mae_values, width, label='MAE')
        axes[0, 0].bar(x + width/2, rmse_values, width, label='RMSE')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].set_title('Model Error Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Scatter plot of predictions vs true values
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, model_name in enumerate(model_names):
            if i >= len(colors):
                break
                
            y_pred = results[model_name]['predictions']
            axes[0, 1].scatter(y_true, y_pred, alpha=0.3, color=colors[i], label=model_name)
        
        # Add identity line
        min_val = min(min(y_true), min(results[model_names[0]]['predictions']))
        max_val = max(max(y_true), max(results[model_names[0]]['predictions']))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--')
        axes[0, 1].set_xlabel('True Risk Score')
        axes[0, 1].set_ylabel('Predicted Risk Score')
        axes[0, 1].set_title('Predictions vs True Values')
        axes[0, 1].legend()
        
        # Error heatmap by vitals for best model
        best_model = min(results.items(), key=lambda x: x[1]['mae'])[0]
        logger.info(f"Creating error heatmap for best model: {best_model}")
        
        # Extract predictions and calculate errors
        y_pred_best = results[best_model]['predictions']
        errors = y_pred_best - y_true
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame({
            'heart_rate': X_test[:, 0],
            'blood_oxygen': X_test[:, 1],
            'error': errors
        })
        
        # Create pivot table for heatmap
        hr_bins = np.linspace(min(heatmap_data['heart_rate']), max(heatmap_data['heart_rate']), 15)
        bo_bins = np.linspace(min(heatmap_data['blood_oxygen']), max(heatmap_data['blood_oxygen']), 15)
        
        heatmap_data['hr_bin'] = pd.cut(heatmap_data['heart_rate'], bins=hr_bins, labels=hr_bins[:-1])
        heatmap_data['bo_bin'] = pd.cut(heatmap_data['blood_oxygen'], bins=bo_bins, labels=bo_bins[:-1])
        
        pivot = heatmap_data.pivot_table(
            values='error', 
            index='bo_bin', 
            columns='hr_bin', 
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(pivot, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title(f'Error Heatmap for {best_model}')
        axes[1, 0].set_xlabel('Heart Rate')
        axes[1, 0].set_ylabel('Blood Oxygen')
        
        # Error distribution by condition
        condition_errors = {}
        for record, error in zip(test_data, errors):
            if 'user_context' in record and 'health_conditions' in record['user_context']:
                conditions = record['user_context']['health_conditions']
                condition_key = ', '.join(conditions) if conditions else 'No conditions'
                
                if condition_key not in condition_errors:
                    condition_errors[condition_key] = []
                
                condition_errors[condition_key].append(error)
        
        # Calculate average error by condition
        condition_names = []
        condition_avg_errors = []
        condition_std_errors = []
        
        for condition, errors_list in condition_errors.items():
            if len(errors_list) > 5:  # Only include conditions with enough samples
                condition_names.append(condition)
                condition_avg_errors.append(np.mean(errors_list))
                condition_std_errors.append(np.std(errors_list))
        
        # Sort by average error
        sort_idx = np.argsort(condition_avg_errors)
        condition_names = [condition_names[i] for i in sort_idx]
        condition_avg_errors = [condition_avg_errors[i] for i in sort_idx]
        condition_std_errors = [condition_std_errors[i] for i in sort_idx]
        
        # Plot error by condition
        axes[1, 1].barh(condition_names, condition_avg_errors, xerr=condition_std_errors)
        axes[1, 1].set_title('Average Error by Condition')
        axes[1, 1].set_xlabel('Error (Predicted - True)')
        axes[1, 1].axvline(x=0, color='k', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison_{self.timestamp}.png")
        plt.close()

    def implement_ensemble_approach(self, test_on_sample=True):
        """Implement and test an ensemble approach for risk prediction"""
        logger.info("Implementing ensemble approach for risk prediction")
        
        # Train individual models
        training_data = self.generate_training_data()
        models = self.train_improved_models(training_data)
        
        # Create ensemble (using the now global class)
        ensemble = HealthRiskEnsemble(
            models=models,
            feature_engineering=FeatureEngineering,
            rule_weight=0.3
        )
        
        # Save ensemble
        ensemble_path = os.path.join(self.output_dir, f"ensemble_model_{self.timestamp}.pkl")
        joblib.dump(ensemble, ensemble_path)
        logger.info(f"Saved ensemble model to {ensemble_path}")
        
        # Copy to default location
        default_ensemble_path = os.path.join(HealthMLService.MODEL_DIR, "ensemble_model.pkl")
        joblib.dump(ensemble, default_ensemble_path)
        logger.info(f"Saved default ensemble model to {default_ensemble_path}")
        
        # Test ensemble if requested
        if test_on_sample:
            logger.info("Testing ensemble model on sample data")
            
            # Generate test data
            conditions = ["Hypertension", "COPD", "Anxiety", "Heart Disease"]
            test_data = []
            
            for condition in conditions:
                user_profile = {
                    'age': 50,
                    'health_conditions': [condition]
                }
                
                # Generate 25 samples for each condition
                condition_data = self._generate_condition_specific_data(
                    user_profile, samples=25, abnormal_prob=0.3
                )
                
                # Add user context and condition label
                for record in condition_data:
                    record['user_context'] = user_profile.copy()
                    record['condition'] = condition
                
                test_data.extend(condition_data)
            
            # Add no-condition data
            user_profile = {
                'age': 50,
                'health_conditions': []
            }
            
            no_condition_data = self._generate_condition_specific_data(
                user_profile, samples=25, abnormal_prob=0.2
            )
            
            for record in no_condition_data:
                record['user_context'] = user_profile.copy()
                record['condition'] = "No conditions"
            
            test_data.extend(no_condition_data)
            
            # Calculate risk scores and make predictions
            results = []
            
            for record in test_data:
                heart_rate = record['heart_rate']
                blood_oxygen = record['blood_oxygen']
                user_context = record['user_context']
                
                # Get rule-based risk
                rule_risk = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
                
                # Get ensemble prediction
                ensemble_risk = ensemble.predict(heart_rate, blood_oxygen, user_context)
                
                # Create result record
                result = {
                    'heart_rate': heart_rate,
                    'blood_oxygen': blood_oxygen,
                    'condition': record['condition'],
                    'rule_risk': rule_risk,
                    'ensemble_risk': ensemble_risk,
                    'difference': ensemble_risk - rule_risk
                }
                
                results.append(result)
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate metrics by condition
            metrics_by_condition = df.groupby('condition').agg({
                'rule_risk': 'mean',
                'ensemble_risk': 'mean',
                'difference': ['mean', 'std']
            })
            
            # Save results
            df.to_csv(f"{self.output_dir}/ensemble_test_{self.timestamp}.csv", index=False)
            
            # Visualize results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot risk distribution by condition
            sns.boxplot(x='condition', y='rule_risk', data=df, ax=ax1, color='skyblue')
            sns.boxplot(x='condition', y='ensemble_risk', data=df, ax=ax1, color='salmon')
            ax1.set_title('Risk Score Distribution by Condition')
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('Risk Score')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Plot difference distribution
            sns.boxplot(x='condition', y='difference', data=df, ax=ax2)
            ax2.set_title('Ensemble - Rule Difference by Condition')
            ax2.set_xlabel('Condition')
            ax2.set_ylabel('Difference')
            ax2.axhline(y=0, color='k', linestyle='--')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/ensemble_results_{self.timestamp}.png")
            plt.close()
            
            logger.info(f"Ensemble test results saved to {self.output_dir}")
            
            return ensemble, df
        
        return ensemble

def main():
    """Main function to run the model improvement"""
    import argparse
    parser = argparse.ArgumentParser(description='Improve and test health ML models')
    parser.add_argument('--output-dir', default='improved_models', 
                        help='Directory to save improved models and results')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train models, do not implement ensemble')
    parser.add_argument('--ensemble-only', action='store_true',
                        help='Only implement and test ensemble approach')
    args = parser.parse_args()
    
    # Create model improver
    improver = ModelImprover(output_dir=args.output_dir)
    
    if args.ensemble_only:
        # Only implement ensemble
        logger.info("Implementing ensemble approach")
        ensemble, results = improver.implement_ensemble_approach()
    elif args.train_only:
        # Only train models
        logger.info("Training improved models")
        training_data = improver.generate_training_data()
        models = improver.train_improved_models(training_data)
        improver.evaluate_models(models)
    else:
        # Full improvement process
        logger.info("Running full model improvement process")
        training_data = improver.generate_training_data()
        models = improver.train_improved_models(training_data)
        improver.evaluate_models(models)
        ensemble, results = improver.implement_ensemble_approach(test_on_sample=True)
    
    logger.info("Model improvement complete")

if __name__ == "__main__":
    main()