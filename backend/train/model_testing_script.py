"""
Condition-Aware Health ML Model Testing Script

This standalone script tests the improved condition-aware ML model by:
1. Creating a condition-aware model
2. Testing it with various health conditions
3. Generating visualizations to show model performance
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from datetime import datetime
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directories
MODEL_DIR = "improved_models"
TEST_RESULTS_DIR = "model_test_results"

class HealthModelTester:
    """Tests condition-aware health risk prediction models"""
    
    def __init__(self, output_dir=TEST_RESULTS_DIR):
        """Initialize the model tester"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Store test results
        self.condition_results = {}
        
        # Store models
        self.rule_based_model = None
        self.hybrid_model = None
        self.pure_ml_model = None
    
    def calculate_rule_based_risk(self, heart_rate, blood_oxygen, user_context=None):
        """Calculate rule-based risk score using similar logic as HealthService"""
        # Normal ranges
        hr_normal_low, hr_normal_high = 60, 100
        bo_normal_low = 95
        
        # Apply condition-specific adjustments
        if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
            health_conditions = [c.lower() for c in user_context['health_conditions']]
            health_conditions_text = " ".join(health_conditions)
            
            # Adjust heart rate range for anxiety
            if any(term in health_conditions_text for term in ['anxiety', 'panic', 'stress']):
                hr_normal_high += 15  # Allow higher heart rate for anxiety patients
            
            # Adjust blood oxygen threshold for COPD
            if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis']):
                bo_normal_low = 92  # Lower threshold for COPD patients
                
            # Athletes might have lower resting heart rates
            if 'athlete' in health_conditions_text:
                hr_normal_low = 50  # Lower threshold for athletes
        
        # Calculate heart rate risk
        if heart_rate < hr_normal_low:
            # Low heart rate risk
            hr_risk = 40 + (hr_normal_low - heart_rate) * 3
        elif heart_rate > hr_normal_high:
            # High heart rate risk
            hr_risk = 40 + (heart_rate - hr_normal_high) * 1.5
        else:
            # Normal heart rate
            hr_risk = 20 + abs(heart_rate - 75) * 0.4  # Slight risk away from ideal
        
        # Calculate blood oxygen risk
        if blood_oxygen < bo_normal_low:
            # Low blood oxygen risk
            bo_risk = 50 + (bo_normal_low - blood_oxygen) * 10
        else:
            # Normal blood oxygen
            bo_risk = max(0, 20 - (blood_oxygen - bo_normal_low) * 2)
        
        # Combine risks (blood oxygen issues are more serious)
        combined_risk = (hr_risk * 0.4) + (bo_risk * 0.6)
        
        # Cap at 100
        return min(100, combined_risk)
    
    def create_condition_aware_model(self, health_conditions=None):
        """Create condition-aware ML model that incorporates health conditions as features"""
        logger.info(f"Creating condition-aware model for conditions: {health_conditions or 'None'}")
        
        # Create base model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Generate training data with condition-specific examples
        X_train, y_train = self._generate_training_data(health_conditions or [])
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info(f"Model trained with {len(X_train)} samples")
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"condition_aware_model_{self.timestamp}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        self.hybrid_model = model
        return model
    
    def _generate_training_data(self, health_conditions):
        """Generate synthetic training data with condition-specific adjustments"""
        # Combine all conditions into a single string for easier checking
        conditions_text = " ".join(health_conditions).lower()
        
        # Generate base samples across the range of heart rates and blood oxygen
        X_samples = []
        y_samples = []
        
        # Generate samples across normal ranges
        for hr in range(40, 180, 5):  # Heart rate from 40 to 180
            for bo in range(80, 101):  # Blood oxygen from 80 to 100
                # Create feature vector
                features = [hr, bo]
                
                # Add condition markers as features
                has_copd = 1.0 if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']) else 0.0
                has_anxiety = 1.0 if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']) else 0.0
                has_heart_issue = 1.0 if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']) else 0.0
                is_athlete = 1.0 if 'athlete' in conditions_text else 0.0
                
                # Complete feature vector with condition indicators
                features.extend([has_copd, has_anxiety, has_heart_issue, is_athlete])
                
                X_samples.append(features)
                
                # Create user context for rule-based calculation
                user_context = {'health_conditions': health_conditions} if health_conditions else None
                
                # Get rule-based risk score
                risk_score = self.calculate_rule_based_risk(hr, bo, user_context)
                y_samples.append(risk_score)
        
        # Add extra condition-specific samples to better learn special cases
        if health_conditions:
            # COPD-specific samples: need more examples with lowered blood oxygen
            if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']):
                for _ in range(200):
                    hr = np.random.uniform(60, 100)
                    bo = np.random.uniform(88, 94)  # COPD patients have lower baseline
                    
                    features = [hr, bo, 1.0, 0.0, 0.0, 0.0]  # COPD=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(self.calculate_rule_based_risk(hr, bo, {'health_conditions': health_conditions}))
            
            # Anxiety-specific samples: need more examples with elevated heart rate
            if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']):
                for _ in range(200):
                    hr = np.random.uniform(70, 115)  # Higher baseline heart rate
                    bo = np.random.uniform(95, 100)
                    
                    features = [hr, bo, 0.0, 1.0, 0.0, 0.0]  # Anxiety=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(self.calculate_rule_based_risk(hr, bo, {'health_conditions': health_conditions}))
            
            # Heart condition samples
            if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']):
                for _ in range(200):
                    # Higher risk for unusual heart rates
                    hr = np.random.choice([
                        np.random.uniform(40, 55),   # Low heart rate
                        np.random.uniform(100, 130)  # High heart rate
                    ])
                    bo = np.random.uniform(93, 100)
                    
                    features = [hr, bo, 0.0, 0.0, 1.0, 0.0]  # Heart issue=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(self.calculate_rule_based_risk(hr, bo, {'health_conditions': health_conditions}))
            
            # Athlete samples: lower resting heart rate
            if any(c in conditions_text for c in ['athlete']):
                for _ in range(200):
                    hr = np.random.uniform(40, 70)  # Athletes often have lower heart rates
                    bo = np.random.uniform(95, 100)
                    
                    features = [hr, bo, 0.0, 0.0, 0.0, 1.0]  # Athlete=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(self.calculate_rule_based_risk(hr, bo, {'health_conditions': health_conditions}))
        
        return np.array(X_samples), np.array(y_samples)
    
    def predict_with_model(self, model, heart_rate, blood_oxygen, health_conditions=None):
        """Make prediction using condition-aware model"""
        # Process health conditions for expanded features
        has_copd = 0.0
        has_anxiety = 0.0
        has_heart_issue = 0.0
        is_athlete = 0.0
        
        if health_conditions:
            conditions_text = " ".join([c.lower() for c in health_conditions])
            
            # Set condition indicators
            has_copd = 1.0 if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']) else 0.0
            has_anxiety = 1.0 if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']) else 0.0
            has_heart_issue = 1.0 if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']) else 0.0
            is_athlete = 1.0 if 'athlete' in conditions_text else 0.0
        
        # Prepare expanded features
        features = [heart_rate, blood_oxygen, has_copd, has_anxiety, has_heart_issue, is_athlete]
        X = np.array([features])
        
        # Make prediction
        prediction = model.predict(X)[0]
        return float(prediction)
    
    def create_pure_ml_model(self):
        """Create a pure ML model without condition awareness"""
        logger.info("Creating pure ML model with no condition awareness")
        
        # Create base model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Generate training data without condition awareness
        X_train, y_train = self._generate_pure_ml_training_data()
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info(f"Pure ML model trained with {len(X_train)} samples")
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"pure_ml_model_{self.timestamp}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Pure ML model saved to {model_path}")
        
        self.pure_ml_model = model
        return model
    
    def _generate_pure_ml_training_data(self):
        """Generate synthetic training data for pure ML model (no condition awareness)"""
        X_samples = []
        y_samples = []
        
        # Generate samples across normal ranges
        for hr in range(40, 180, 5):  # Heart rate from 40 to 180
            for bo in range(80, 101):  # Blood oxygen from 80 to 100
                # Create feature vector (only vital signs, no condition features)
                features = [hr, bo]
                X_samples.append(features)
                
                # Get risk score (without any condition awareness)
                risk_score = self.calculate_rule_based_risk(hr, bo)
                y_samples.append(risk_score)
                
                # Add some noise to make the model learn patterns
                if np.random.random() < 0.3:  # 30% of samples have noise
                    noise = np.random.normal(0, 5)  # mean 0, std 5
                    noisy_risk = max(0, min(100, risk_score + noise))
                    X_samples.append(features)
                    y_samples.append(noisy_risk)
        
        return np.array(X_samples), np.array(y_samples)
    
    def predict_with_pure_ml_model(self, model, heart_rate, blood_oxygen):
        """Make prediction using pure ML model"""
        # Prepare features (only vital signs)
        features = [heart_rate, blood_oxygen]
        X = np.array([features])
        
        # Make prediction
        prediction = model.predict(X)[0]
        return float(prediction)
    
    def test_all_models_across_conditions(self, conditions_to_test):
        """Test all models (rule-based, hybrid, pure ML) across different health conditions"""
        logger.info(f"Testing all models across {len(conditions_to_test)} different conditions")
        
        # Ensure we have all three models
        if not self.hybrid_model:
            self.hybrid_model = self.create_condition_aware_model()
            
        if not self.pure_ml_model:
            self.pure_ml_model = self.create_pure_ml_model()
        
        all_results = []
        
        for condition in conditions_to_test:
            logger.info(f"Testing condition: {condition}")
            
            # Generate test data for this condition
            test_data = self._generate_test_data_for_condition(condition)
            
            # Test models on this data
            results = []
            
            for hr, bo in test_data:
                # Calculate rule-based risk score
                user_context = {'health_conditions': [condition]} if condition != "None" else None
                rule_risk = self.calculate_rule_based_risk(hr, bo, user_context)
                
                # Get hybrid model prediction (condition-aware)
                if condition == "None":
                    hybrid_risk = self.predict_with_model(self.hybrid_model, hr, bo)
                else:
                    hybrid_risk = self.predict_with_model(self.hybrid_model, hr, bo, [condition])
                
                # Get pure ML model prediction (not condition-aware)
                pure_ml_risk = self.predict_with_pure_ml_model(self.pure_ml_model, hr, bo)
            
            # Store results
                results.append({
                    'condition': condition,
                    'heart_rate': hr,
                    'blood_oxygen': bo,
                    'rule_risk': rule_risk,
                'hybrid_risk': hybrid_risk,
                    'pure_ml_risk': pure_ml_risk,
                    'hybrid_diff': hybrid_risk - rule_risk,
                    'pure_ml_diff': pure_ml_risk - rule_risk
                })
            
            # Calculate metrics for both models
            df = pd.DataFrame(results)
            hybrid_mae = mean_absolute_error(df['rule_risk'], df['hybrid_risk'])
            pure_ml_mae = mean_absolute_error(df['rule_risk'], df['pure_ml_risk'])
            
            # Store results for this condition
            self.condition_results[condition] = {
                'detailed_results': results,
                'metrics': {
                    'hybrid_mae': hybrid_mae,
                    'pure_ml_mae': pure_ml_mae,
                    'samples': len(results),
                    'avg_rule_risk': df['rule_risk'].mean(),
                    'avg_hybrid_risk': df['hybrid_risk'].mean(),
                    'avg_pure_ml_risk': df['pure_ml_risk'].mean(),
                    'avg_hybrid_diff': df['hybrid_diff'].mean(),
                    'avg_pure_ml_diff': df['pure_ml_diff'].mean(),
                    'max_hybrid_diff': abs(df['hybrid_diff']).max(),
                    'max_pure_ml_diff': abs(df['pure_ml_diff']).max()
                }
            }
            
            all_results.extend(results)
        
        # Convert all results to DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        # Save combined results
        self.results_df.to_csv(f"{self.output_dir}/all_models_comparison_{self.timestamp}.csv", index=False)
        
        return self.condition_results
    
    def _generate_test_data_for_condition(self, condition, samples=100):
        """Generate test data specific to a health condition"""
        test_points = []
        
        if condition.lower() == "copd" or condition.lower() == "emphysema":
            # For COPD, include more data points with lower blood oxygen
            for _ in range(samples // 2):
                hr = np.random.uniform(60, 100)
                bo = np.random.uniform(88, 94)  # Lower blood oxygen range
                test_points.append((hr, bo))
            
            for _ in range(samples // 2):
                hr = np.random.uniform(50, 120)
                bo = np.random.uniform(85, 100)
                test_points.append((hr, bo))
                
        elif condition.lower() == "anxiety":
            # For anxiety, include more data points with higher heart rates
            for _ in range(samples // 2):
                hr = np.random.uniform(80, 120)  # Higher heart rate range
                bo = np.random.uniform(95, 100)
                test_points.append((hr, bo))
            
            for _ in range(samples // 2):
                hr = np.random.uniform(60, 140)
                bo = np.random.uniform(90, 100)
                test_points.append((hr, bo))
                
        elif "heart" in condition.lower():
            # For heart conditions, include more variety in heart rates
            for _ in range(samples // 3):
                hr = np.random.uniform(40, 60)  # Low heart rate
                bo = np.random.uniform(93, 100)
                test_points.append((hr, bo))
                
            for _ in range(samples // 3):
                hr = np.random.uniform(100, 130)  # High heart rate
                bo = np.random.uniform(93, 100)
                test_points.append((hr, bo))
                
            for _ in range(samples // 3):
                hr = np.random.uniform(60, 100)
                bo = np.random.uniform(90, 100)
                test_points.append((hr, bo))
                
        elif "athlete" in condition.lower():
            # For athletes, include more data with lower baseline heart rates
            for _ in range(samples // 2):
                hr = np.random.uniform(40, 70)  # Lower heart rate range
                bo = np.random.uniform(95, 100)
                test_points.append((hr, bo))
                
            for _ in range(samples // 2):
                hr = np.random.uniform(60, 110)
                bo = np.random.uniform(93, 100)
                test_points.append((hr, bo))
                
            else:
            # For other conditions or no condition, use a general distribution
            for _ in range(samples):
                hr = np.random.uniform(50, 120)
                bo = np.random.uniform(88, 100)
                test_points.append((hr, bo))
        
        return test_points
    
    def visualize_condition_performance(self):
        """Visualize model performance across different conditions"""
        if not self.condition_results:
            logger.error("No condition results to visualize. Run test_all_models_across_conditions first.")
            return
        
        # Extract metrics for visualization
        conditions = []
        hybrid_maes = []
        pure_ml_maes = []
        hybrid_diffs = []
        pure_ml_diffs = []
        
        for condition, results in self.condition_results.items():
            conditions.append(condition)
            metrics = results['metrics']
            hybrid_maes.append(metrics['hybrid_mae'])
            pure_ml_maes.append(metrics['pure_ml_mae'])
            hybrid_diffs.append(metrics['avg_hybrid_diff'])
            pure_ml_diffs.append(metrics['avg_pure_ml_diff'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot MAE by condition
        plt.subplot(1, 2, 1)
        bars = plt.bar(conditions, hybrid_maes, color='skyblue', label='Hybrid Model')
        bars = plt.bar(conditions, pure_ml_maes, color='lightgreen', label='Pure ML Model', bottom=hybrid_maes)
        plt.title('Mean Absolute Error by Condition')
        plt.xlabel('Condition')
        plt.ylabel('MAE')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Plot average differences
        plt.subplot(1, 2, 2)
        colors = ['red' if d < 0 else 'green' for d in hybrid_diffs]
        bars = plt.bar(conditions, hybrid_diffs, color=colors, label='Hybrid Model')
        colors = ['red' if d < 0 else 'green' for d in pure_ml_diffs]
        bars = plt.bar(conditions, pure_ml_diffs, color=colors, label='Pure ML Model', bottom=hybrid_diffs)
        plt.title('Average Difference (Hybrid - Rule)')
        plt.xlabel('Condition')
        plt.ylabel('Difference')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height if height >= 0 else height - 1,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/condition_performance_{self.timestamp}.png")
        plt.close()
        
        # Create additional visualizations
        self._visualize_error_distribution()
        self._visualize_decision_boundaries()
    
    def _visualize_error_distribution(self):
        """Visualize distribution of errors"""
        if not hasattr(self, 'results_df'):
            logger.error("No results dataframe found. Run test_all_models_across_conditions first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot error histogram by condition
        sns.histplot(data=self.results_df, x='hybrid_diff', hue='condition', bins=20, alpha=0.5, label='Hybrid Model')
        sns.histplot(data=self.results_df, x='pure_ml_diff', hue='condition', bins=20, alpha=0.5, label='Pure ML Model')
        plt.title('Distribution of Prediction Differences by Condition')
        plt.xlabel('ML Risk - Rule Risk')
        plt.axvline(x=0, color='black', linestyle='-')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/error_distribution_{self.timestamp}.png")
        plt.close()
    
    def _visualize_decision_boundaries(self):
        """Visualize decision boundaries for different conditions"""
        conditions_to_visualize = ['COPD', 'Anxiety', 'Heart Disease', 'None']
        
        for condition in conditions_to_visualize:
            if condition not in self.condition_results:
                    continue
                
            # Generate grid data
            hr_range = np.linspace(40, 140, 40)
            bo_range = np.linspace(85, 100, 30)
            
            grid_data = []
            
            for hr in hr_range:
                for bo in bo_range:
                    # Calculate rule-based risk
                    user_context = {'health_conditions': [condition]} if condition != "None" else None
                    rule_risk = self.calculate_rule_based_risk(hr, bo, user_context)
                    
                    # Get hybrid model prediction
                    hybrid_risk = self.predict_with_model(self.hybrid_model, hr, bo)
                    
                    # Get pure ML model prediction
                    pure_ml_risk = self.predict_with_pure_ml_model(self.pure_ml_model, hr, bo)
                    
                    grid_data.append({
                        'heart_rate': hr,
                        'blood_oxygen': bo,
                        'rule_risk': rule_risk,
                        'hybrid_risk': hybrid_risk,
                        'pure_ml_risk': pure_ml_risk,
                        'hybrid_diff': hybrid_risk - rule_risk,
                        'pure_ml_diff': pure_ml_risk - rule_risk
                    })
            
            # Convert to DataFrame
            grid_df = pd.DataFrame(grid_data)
            
            # Create visualizations
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Decision Boundaries for {condition}', fontsize=16)
            
            # Rule-based risk
            pivot_rule = grid_df.pivot_table(values='rule_risk', 
                                         index='blood_oxygen', 
                                         columns='heart_rate')
            sns.heatmap(pivot_rule, ax=axes[0], cmap='YlOrRd', vmin=0, vmax=100)
            axes[0].set_title(f'Rule-based Risk Score ({condition})')
            axes[0].set_xlabel('Heart Rate')
            axes[0].set_ylabel('Blood Oxygen')
            
            # Hybrid model
            pivot_hybrid = grid_df.pivot_table(values='hybrid_risk', 
                                       index='blood_oxygen', 
                                       columns='heart_rate')
            sns.heatmap(pivot_hybrid, ax=axes[1], cmap='YlOrRd', vmin=0, vmax=100)
            axes[1].set_title(f'Hybrid Model Risk Score ({condition})')
            axes[1].set_xlabel('Heart Rate')
            axes[1].set_ylabel('Blood Oxygen')
            
            # Pure ML model
            pivot_pure_ml = grid_df.pivot_table(values='pure_ml_risk', 
                                       index='blood_oxygen', 
                                       columns='heart_rate')
            sns.heatmap(pivot_pure_ml, ax=axes[2], cmap='YlOrRd', vmin=0, vmax=100)
            axes[2].set_title(f'Pure ML Model Risk Score ({condition})')
            axes[2].set_xlabel('Heart Rate')
            axes[2].set_ylabel('Blood Oxygen')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/decision_boundary_{condition}_{self.timestamp}.png")
        plt.close()

def main():
    """Run model testing and generate visualizations"""
    import argparse
    parser = argparse.ArgumentParser(description="Test condition-aware health ML model")
    parser.add_argument("--output-dir", default=TEST_RESULTS_DIR, help="Directory to save test results")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before running")
    args = parser.parse_args()
    
    # Set up directories
    if args.clean and os.path.exists(args.output_dir):
        logger.info(f"Cleaning output directory: {args.output_dir}")
        
        # Create backup folder with timestamp
        if os.listdir(args.output_dir):  # Only backup if not empty
            backup_dir = f"{args.output_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(args.output_dir, backup_dir)
            logger.info(f"Backed up existing results to {backup_dir}")
        
        # Clean directory
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = HealthModelTester(output_dir=args.output_dir)
    
    # Define conditions to test
    conditions_to_test = [
        "COPD", 
        "Anxiety",
        "Heart Disease", 
        "Athlete",
        "Diabetes",
        "Hypertension",
        "None"  # No condition
    ]
    
    # Test all models across conditions
    logger.info("Testing all models across different health conditions")
    condition_results = tester.test_all_models_across_conditions(conditions_to_test)
    
    # Visualize results
    logger.info("Generating visualizations")
    tester.visualize_condition_performance()
    
    # Print summary
    logger.info("\n===== MODEL TEST SUMMARY =====")
    logger.info(f"Tested all models with {len(conditions_to_test)} conditions")
    
    for condition, results in condition_results.items():
        metrics = results['metrics']
        logger.info(f"  {condition}: Hybrid MAE={metrics['hybrid_mae']:.2f}, Pure ML MAE={metrics['pure_ml_mae']:.2f}")
    
    logger.info(f"\nTest results saved to {args.output_dir}")
    logger.info("Run dashboard_script.py to generate comprehensive dashboard")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())