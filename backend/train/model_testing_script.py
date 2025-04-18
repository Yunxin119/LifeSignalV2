"""
Health ML Model Testing and Visualization Tool

This script tests the health risk prediction model across different conditions
and visualizes its performance compared to the rule-based approach.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime, timedelta
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

class ModelTester:
    def __init__(self, output_dir="model_test_results"):
        """Initialize the model tester with an output directory for results"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data storage
        self.test_results = {}
        self.condition_results = {}
        
    def test_model_across_conditions(self, conditions_to_test, samples_per_condition=100):
        """Test model performance across different health conditions"""
        logger.info(f"Testing model across {len(conditions_to_test)} different health conditions")
        
        all_results = []
        
        for condition in conditions_to_test:
            logger.info(f"Testing condition: {condition}")
            
            # Create user profile with this condition
            user_profile = {
                'age': 50,  # Default age
                'health_conditions': [condition]
            }
            
            # Generate test data for this condition
            test_data = self._generate_condition_specific_data(user_profile, 
                                                              samples=samples_per_condition,
                                                              abnormal_prob=0.3)
            
            # Test model on this data
            results = self._evaluate_model_on_data(test_data, user_profile)
            
            # Store results for this condition
            self.condition_results[condition] = results
            
            # Add condition identifier to each result
            for record in results['detailed_results']:
                record['condition'] = condition
                all_results.append(record)
        
        # Convert all detailed results to DataFrame for easier analysis
        self.results_df = pd.DataFrame(all_results)
        
        # Save combined results
        self.results_df.to_csv(f"{self.output_dir}/all_conditions_{self.timestamp}.csv", index=False)
        
        return self.condition_results
    
    def test_model_across_age_groups(self, age_groups=[(18, 30), (31, 50), (51, 70), (71, 90)]):
        """Test model performance across different age groups"""
        logger.info(f"Testing model across {len(age_groups)} different age groups")
        
        age_results = {}
        all_results = []
        
        for age_min, age_max in age_groups:
            age_group = f"{age_min}-{age_max}"
            logger.info(f"Testing age group: {age_group}")
            
            # Create user profiles with different ages in this range
            results_for_group = []
            
            for _ in range(5):  # Test 5 different users in each age group
                age = np.random.randint(age_min, age_max+1)
                user_profile = {
                    'age': age,
                    'health_conditions': []  # No conditions to isolate age effect
                }
                
                # Generate test data for this age
                test_data = self._generate_condition_specific_data(user_profile, 
                                                                samples=40,  # 40 samples per user
                                                                abnormal_prob=0.3)
                
                # Test model on this data
                results = self._evaluate_model_on_data(test_data, user_profile)
                results_for_group.extend(results['detailed_results'])
            
            # Add age group identifier
            for record in results_for_group:
                record['age_group'] = age_group
                all_results.append(record)
                
            # Calculate aggregated metrics for this age group
            df = pd.DataFrame(results_for_group)
            age_results[age_group] = {
                'rule_ml_mae': mean_absolute_error(df['true_risk'], df['ml_risk']),
                'samples': len(df),
                'avg_true_risk': df['true_risk'].mean(),
                'avg_ml_risk': df['ml_risk'].mean()
            }
        
        # Save combined results
        age_df = pd.DataFrame(all_results)
        age_df.to_csv(f"{self.output_dir}/age_groups_{self.timestamp}.csv", index=False)
        
        return age_results
    
    def test_decision_boundaries(self, range_pairs=(
            {'hr': (40, 180), 'bo': (85, 100)},  # Full range
            {'hr': (85, 105), 'bo': (90, 98)}     # Normal-ish range
        )):
        """Test how model performs around decision boundaries"""
        logger.info("Testing model performance around decision boundaries")
        
        boundary_results = {}
        
        for i, ranges in enumerate(range_pairs):
            hr_range = ranges['hr']
            bo_range = ranges['bo']
            
            range_name = f"range_{i+1}_hr{hr_range[0]}-{hr_range[1]}_bo{bo_range[0]}-{bo_range[1]}"
            logger.info(f"Testing {range_name}")
            
            # Generate grid of test points
            hr_points = np.linspace(hr_range[0], hr_range[1], 20)
            bo_points = np.linspace(bo_range[0], bo_range[1], 20)
            
            grid_results = []
            
            # Define user profiles to test
            profiles = [
                {'age': 40, 'health_conditions': []},
                {'age': 70, 'health_conditions': []},
                {'age': 50, 'health_conditions': ['Anxiety']},
                {'age': 50, 'health_conditions': ['COPD']},
            ]
            
            for profile in profiles:
                profile_str = f"age{profile['age']}_conditions{'_'.join(profile['health_conditions']) if profile['health_conditions'] else 'none'}"
                logger.info(f"  Testing profile: {profile_str}")
                
                for hr in hr_points:
                    for bo in bo_points:
                        # Calculate rule-based risk
                        rule_risk = HealthService.calculate_risk_score(hr, bo, profile)
                        
                        # Calculate ML-based risk
                        features = FeatureEngineering.extract_features(hr, bo, None, profile)
                        ml_risk = self._predict_with_default_model(features[:2], profile)
                        
                        grid_results.append({
                            'heart_rate': hr,
                            'blood_oxygen': bo,
                            'true_risk': rule_risk,
                            'ml_risk': ml_risk,
                            'difference': ml_risk - rule_risk,
                            'profile': profile_str
                        })
            
            # Convert to DataFrame
            grid_df = pd.DataFrame(grid_results)
            
            # Save results
            grid_df.to_csv(f"{self.output_dir}/boundary_{range_name}_{self.timestamp}.csv", index=False)
            
            # Store in results
            boundary_results[range_name] = grid_df
        
        return boundary_results
    
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
    
    def _evaluate_model_on_data(self, test_data, user_profile):
        """Evaluate model performance on test data"""
        results = {
            'detailed_results': [],
            'metrics': {}
        }
        
        true_risks = []
        ml_risks = []
        hybrid_risks = []
        
        for record in test_data:
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            
            # Calculate rule-based risk score
            true_risk = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_profile)
            
            # Extract features and get ML prediction
            features = FeatureEngineering.extract_features(heart_rate, blood_oxygen, None, user_profile)
            ml_risk = self._predict_with_default_model(features[:2], user_profile)
            
            # Calculate hybrid risk score
            hybrid_risk = (ml_risk * 0.7) + (true_risk * 0.3)
            
            # Store results
            result_record = {
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen,
                'true_risk': true_risk,
                'ml_risk': ml_risk,
                'hybrid_risk': hybrid_risk,
                'diff': ml_risk - true_risk
            }
            
            results['detailed_results'].append(result_record)
            true_risks.append(true_risk)
            ml_risks.append(ml_risk)
            hybrid_risks.append(hybrid_risk)
        
        # Calculate metrics
        true_risks = np.array(true_risks)
        ml_risks = np.array(ml_risks)
        hybrid_risks = np.array(hybrid_risks)
        
        results['metrics'] = {
            'samples': len(test_data),
            'rule_ml_mae': mean_absolute_error(true_risks, ml_risks),
            'rule_ml_mse': mean_squared_error(true_risks, ml_risks),
            'rule_ml_rmse': np.sqrt(mean_squared_error(true_risks, ml_risks)),
            'rule_hybrid_mae': mean_absolute_error(true_risks, hybrid_risks),
            'avg_true_risk': np.mean(true_risks),
            'avg_ml_risk': np.mean(ml_risks),
            'avg_diff': np.mean(ml_risks - true_risks),
            'max_diff': np.max(np.abs(ml_risks - true_risks))
        }
        
        return results
    
    def _predict_with_default_model(self, features, user_profile=None):
        """Make prediction using the default model"""
        try:
            # Try to use the default model from the ML service
            model_path = os.path.join(HealthMLService.MODEL_DIR, "default_model.pkl")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                X = np.array([features])
                prediction = model.predict(X)[0]
                return float(prediction)
            else:
                # If no model exists, create a temporary one
                model = HealthMLService._create_base_model(user_profile)
                X = np.array([features])
                prediction = model.predict(X)[0]
                return float(prediction)
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Fallback to rule-based approach
            if len(features) >= 2:
                heart_rate, blood_oxygen = features[:2]
                return HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_profile)
            else:
                return 50.0  # Default mid-range risk
    
    def visualize_condition_comparisons(self):
        """Visualize performance comparison across different conditions"""
        if not self.condition_results:
            logger.error("No condition results to visualize. Run test_model_across_conditions first.")
            return
        
        # Extract metrics for visualization
        conditions = []
        maes = []
        avg_diffs = []
        max_diffs = []
        
        for condition, results in self.condition_results.items():
            conditions.append(condition)
            metrics = results['metrics']
            maes.append(metrics['rule_ml_mae'])
            avg_diffs.append(metrics['avg_diff'])
            max_diffs.append(metrics['max_diff'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot MAE by condition
        ax1.bar(conditions, maes, color='skyblue')
        ax1.set_title('Mean Absolute Error by Condition')
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('MAE')
        ax1.set_ylim(bottom=0)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot avg and max differences
        x = np.arange(len(conditions))
        width = 0.35
        ax2.bar(x - width/2, avg_diffs, width, label='Avg Difference', color='lightgreen')
        ax2.bar(x + width/2, max_diffs, width, label='Max Difference', color='salmon')
        ax2.set_title('Risk Score Differences by Condition')
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('Difference')
        ax2.set_xticks(x)
        ax2.set_xticklabels(conditions)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/condition_comparison_{self.timestamp}.png")
        plt.close()
    
    def visualize_decision_boundaries(self, boundary_data, profile_filter=None):
        """Visualize decision boundaries between rule-based and ML approaches"""
        if not boundary_data:
            logger.error("No boundary data to visualize.")
            return
        
        for range_name, data_df in boundary_data.items():
            # Filter by profile if specified
            if profile_filter:
                filtered_df = data_df[data_df['profile'] == profile_filter]
                if len(filtered_df) == 0:
                    logger.warning(f"No data for profile {profile_filter} in range {range_name}")
                    continue
                suffix = f"_{profile_filter}"
            else:
                # Use first profile as default
                profile = data_df['profile'].iloc[0]
                filtered_df = data_df[data_df['profile'] == profile]
                suffix = f"_{profile}"
            
            # Create figure with multiple plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Decision Boundary Analysis - {range_name}{suffix}", fontsize=16)
            
            # Get unique heart rates and blood oxygen values
            hr_values = sorted(filtered_df['heart_rate'].unique())
            bo_values = sorted(filtered_df['blood_oxygen'].unique())
            
            # Reshape data for heatmaps
            pivot_true = filtered_df.pivot_table(
                values='true_risk', index='blood_oxygen', columns='heart_rate')
            pivot_ml = filtered_df.pivot_table(
                values='ml_risk', index='blood_oxygen', columns='heart_rate')
            pivot_diff = filtered_df.pivot_table(
                values='difference', index='blood_oxygen', columns='heart_rate')
            
            # Plot rule-based risk
            sns.heatmap(pivot_true, ax=axes[0, 0], cmap='YlOrRd', vmin=0, vmax=100)
            axes[0, 0].set_title('Rule-based Risk Score')
            axes[0, 0].set_xlabel('Heart Rate')
            axes[0, 0].set_ylabel('Blood Oxygen')
            
            # Plot ML-based risk
            sns.heatmap(pivot_ml, ax=axes[0, 1], cmap='YlOrRd', vmin=0, vmax=100)
            axes[0, 1].set_title('ML-based Risk Score')
            axes[0, 1].set_xlabel('Heart Rate')
            axes[0, 1].set_ylabel('Blood Oxygen')
            
            # Plot difference
            diff_max = max(abs(pivot_diff.min().min()), abs(pivot_diff.max().max()))
            sns.heatmap(pivot_diff, ax=axes[1, 0], cmap='coolwarm', vmin=-diff_max, vmax=diff_max)
            axes[1, 0].set_title('Risk Score Difference (ML - Rule)')
            axes[1, 0].set_xlabel('Heart Rate')
            axes[1, 0].set_ylabel('Blood Oxygen')
            
            # Scatter plot of differences
            scatter = axes[1, 1].scatter(
                filtered_df['heart_rate'], 
                filtered_df['blood_oxygen'],
                c=filtered_df['difference'], 
                cmap='coolwarm',
                vmin=-diff_max, vmax=diff_max,
                s=100, alpha=0.7
            )
            axes[1, 1].set_title('Risk Score Difference (Scatter)')
            axes[1, 1].set_xlabel('Heart Rate')
            axes[1, 1].set_ylabel('Blood Oxygen')
            plt.colorbar(scatter, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/boundary_{range_name}{suffix}_{self.timestamp}.png")
            plt.close()
    
    def visualize_error_distribution(self):
        """Visualize error distribution across all data"""
        if hasattr(self, 'results_df'):
            df = self.results_df
        else:
            logger.error("No results dataframe available. Run tests first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Error Distribution Analysis", fontsize=16)
        
        # Plot error histogram
        sns.histplot(df['diff'], bins=20, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of ML-Rule Differences')
        axes[0, 0].set_xlabel('Difference')
        axes[0, 0].set_ylabel('Count')
        
        # Plot error by heart rate
        axes[0, 1].scatter(df['heart_rate'], df['diff'], alpha=0.5)
        axes[0, 1].set_title('Error by Heart Rate')
        axes[0, 1].set_xlabel('Heart Rate')
        axes[0, 1].set_ylabel('ML-Rule Difference')
        
        # Plot error by blood oxygen
        axes[1, 0].scatter(df['blood_oxygen'], df['diff'], alpha=0.5)
        axes[1, 0].set_title('Error by Blood Oxygen')
        axes[1, 0].set_xlabel('Blood Oxygen')
        axes[1, 0].set_ylabel('ML-Rule Difference')
        
        # Box plot by condition
        if 'condition' in df.columns:
            sns.boxplot(x='condition', y='diff', data=df, ax=axes[1, 1])
            axes[1, 1].set_title('Error Distribution by Condition')
            axes[1, 1].set_xlabel('Condition')
            axes[1, 1].set_ylabel('ML-Rule Difference')
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/error_distribution_{self.timestamp}.png")
        plt.close()

def main():
    """Main function to run the model testing"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test and visualize health ML model')
    parser.add_argument('--output-dir', default='model_test_results', 
                        help='Directory to save test results and visualizations')
    parser.add_argument('--test-conditions', action='store_true',
                        help='Run tests across different conditions')
    parser.add_argument('--test-age-groups', action='store_true',
                        help='Run tests across different age groups')
    parser.add_argument('--test-boundaries', action='store_true',
                        help='Run tests around decision boundaries')
    args = parser.parse_args()
    
    # Create model tester
    tester = ModelTester(output_dir=args.output_dir)
    
    # Conditions to test
    conditions_to_test = [
        "Hypertension", 
        "Type 2 Diabetes", 
        "Asthma", 
        "COPD", 
        "Heart Disease", 
        "Arrhythmia", 
        "Anxiety", 
        "Depression",
        "Obesity"
    ]
    
    # Run tests
    if args.test_conditions or not (args.test_conditions or args.test_age_groups or args.test_boundaries):
        logger.info("Testing model across different health conditions")
        condition_results = tester.test_model_across_conditions(conditions_to_test)
        tester.visualize_condition_comparisons()
        tester.visualize_error_distribution()
    
    if args.test_age_groups:
        logger.info("Testing model across different age groups")
        age_results = tester.test_model_across_age_groups()
        
    if args.test_boundaries:
        logger.info("Testing model around decision boundaries")
        boundary_results = tester.test_decision_boundaries()
        
        # Visualize for different profiles
        profiles = [
            "age40_conditionsnone",
            "age70_conditionsnone",
            "age50_conditionsAnxiety",
            "age50_conditionsCOPD"
        ]
        
        for profile in profiles:
            tester.visualize_decision_boundaries(boundary_results, profile)
    
    logger.info(f"All tests complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()