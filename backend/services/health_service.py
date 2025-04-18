import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import logging
from datetime import datetime, timedelta
import pandas as pd
import os
import pickle
from config import DEBUG
from models.health_data import HealthData
from models.user import User
from gemini_client import gemini

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for conditions and activity levels matching the training data
CONDITION_EFFECTS = {
    'none': {'heart_rate': 0, 'blood_oxygen': 0},
    'hypertension': {'heart_rate': 5, 'blood_oxygen': -0.5},
    'anxiety': {'heart_rate': 3, 'blood_oxygen': -2},
    'asthma': {'heart_rate': 3, 'blood_oxygen': -2},
    'COPD': {'heart_rate': 6, 'blood_oxygen': -3},
    'heart_disease': {'heart_rate': 8, 'blood_oxygen': -1},
    'sleep_apnea': {'heart_rate': 2, 'blood_oxygen': -2.5},
    'anemia': {'heart_rate': 7, 'blood_oxygen': -2},
    'diabetes': {'heart_rate': 3, 'blood_oxygen': -0.5}
}

ACTIVITY_EFFECTS = {
    'sedentary': {'heart_rate': -5, 'blood_oxygen': -0.5},
    'light': {'heart_rate': 0, 'blood_oxygen': 0},
    'moderate': {'heart_rate': 10, 'blood_oxygen': 0.5},
    'high': {'heart_rate': 20, 'blood_oxygen': 0.2}
}

class HealthService:
    """Service for handling health data analysis"""
    
    # Initialize anomaly detection model
    _anomaly_detector = None
    # Store user-specific baseline values
    _user_baselines = {}
    # Cached training data
    _training_data = None
    # Store scaler for feature normalization
    _scaler = None
    # Store feature names for model input
    _feature_names = None
    # Track if we're using the new model
    _using_new_model = False
    
    @classmethod
    def load_training_data(cls, filepath=None):
        """
        Load training data from CSV file
        
        Args:
            filepath (str, optional): Path to CSV file with training data.
                If not provided, uses default path.
            
        Returns:
            pandas.DataFrame: Loaded training data
        """
        if cls._training_data is not None:
            return cls._training_data
            
        try:
            # Use default path if not provided
            if filepath is None:
                filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'data', 'training_data.csv')
                
                # Check for new training data first
                if not os.path.exists(filepath):
                    # Fall back to simulated data
                    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'data', 'simulated_health_data.csv')
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Training data file not found: {filepath}")
                return None
                
            # Load CSV data
            data = pd.read_csv(filepath, comment='#')
            logger.info(f"Loaded {len(data)} training data points from {filepath}")
            
            # Cache the data
            cls._training_data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    @classmethod
    def get_condition_specific_training_data(cls, condition=None):
        """
        Get condition-specific training data
        
        Args:
            condition (str, optional): Medical condition to filter data by.
                If None, returns all data.
                
        Returns:
            numpy.ndarray: Array of [heart_rate, blood_oxygen] pairs
        """
        # Load training data
        data = cls.load_training_data()
        if data is None:
            return None
            
        # Filter by condition if provided
        if condition:
            # Normalize condition name for matching
            condition_norm = condition.lower()
            
            # Map condition to standardized values
            if 'hypertension' in condition_norm or 'high blood pressure' in condition_norm:
                condition = 'hypertension'
            elif 'asthma' in condition_norm:
                condition = 'asthma'
            elif 'copd' in condition_norm or 'emphysema' in condition_norm or 'bronchitis' in condition_norm:
                condition = 'COPD'
            elif 'heart disease' in condition_norm or 'heart failure' in condition_norm or 'chf' in condition_norm:
                condition = 'heart_disease'
            elif 'sleep apnea' in condition_norm:
                condition = 'sleep_apnea'
            elif 'anemia' in condition_norm:
                condition = 'anemia'
            elif 'diabetes' in condition_norm:
                condition = 'diabetes'
            else:
                condition = 'none'
            
            # Check if condition exists in data
            if condition not in data['condition'].unique():
                logger.warning(f"Condition '{condition}' not found in training data")
                # Fall back to all conditions
                filtered_data = data
            else:
                filtered_data = data[data['condition'] == condition]
        else:
            filtered_data = data
            
        # Extract heart rate and blood oxygen
        return filtered_data[['heart_rate', 'blood_oxygen']].values
    
    @classmethod
    def get_anomaly_detector(cls, condition=None):
        """
        Get or initialize the anomaly detection model
        
        Args:
            condition (str, optional): Medical condition to consider.
                If provided, adjusts the model for the condition.
                
        Returns:
            model: Trained anomaly detection model (either RandomForestClassifier or IsolationForest)
        """
        # If we already loaded the new model and no specific condition is requested, return it
        if cls._using_new_model and cls._anomaly_detector is not None and condition is None:
            return cls._anomaly_detector
            
        # Try to load pre-trained model files
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'models', 'isolation_forest_model.pkl')
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'models', 'standard_scaler.pkl')
        features_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'models', 'feature_names.pkl')
                              
        # Try to load the pre-trained model if we don't have a specific condition
        if condition is None and os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    cls._scaler = pickle.load(f)
                with open(features_path, 'rb') as f:
                    cls._feature_names = pickle.load(f)
                    
                cls._anomaly_detector = model
                cls._using_new_model = True
                logger.info("Loaded pre-trained RandomForest model with scaler and feature names")
                return model
            except Exception as e:
                logger.error(f"Error loading pre-trained model: {e}")
                # Fall back to the old approach
                cls._using_new_model = False
                logger.info("Falling back to on-the-fly IsolationForest model")
        
        # If we have a specific condition or couldn't load the pre-trained model,
        # fall back to the original approach
        
        # Try to load from CSV first
        csv_data = cls.load_training_data()
        
        if csv_data is not None:
            # Get condition-specific training data if available
            condition_data = cls.get_condition_specific_training_data(condition)
            
            # Initialize arrays for synthetic data
            heart_rates = []
            blood_oxygen = []
            
            # If we're using condition-specific data, we need less synthetic data
            n_samples_per_group = 100 if condition else 200
            
            # Add condition-specific data if available
            if condition_data is not None and len(condition_data) > 0:
                logger.info(f"Using {len(condition_data)} condition-specific data points for {condition}")
                
                # Extract heart rates and blood oxygen from condition data
                condition_hr = condition_data[:, 0]
                condition_bo = condition_data[:, 1]
                
                # Add to arrays
                heart_rates = list(condition_hr)
                blood_oxygen = list(condition_bo)
                
                # Since we have real data, reduce the synthetic data amount
                n_samples_per_group = 50
        else:
            # If CSV loading failed, initialize empty arrays
            heart_rates = []
            blood_oxygen = []
            n_samples_per_group = 200
                
        # Add synthetic data for different groups
        
        # Children and adolescents (higher normal heart rates)
        # Age 10-18: HR 60-120 BPM, SpO2 95-100%
        children_hr = np.random.normal(loc=85, scale=12, size=n_samples_per_group)
        children_hr = np.clip(children_hr, 60, 120)
        children_bo = np.random.normal(loc=98, scale=1, size=n_samples_per_group)
        children_bo = np.clip(children_bo, 95, 100)
        
        # Young adults (age 19-35): HR 55-100 BPM, SpO2 95-100%
        young_adult_hr = np.random.normal(loc=75, scale=10, size=n_samples_per_group)
        young_adult_hr = np.clip(young_adult_hr, 55, 100)
        young_adult_bo = np.random.normal(loc=97.5, scale=1, size=n_samples_per_group)
        young_adult_bo = np.clip(young_adult_bo, 95, 100)
        
        # Middle-aged adults (age 36-60): HR 55-95 BPM, SpO2 94-100%
        middle_adult_hr = np.random.normal(loc=72, scale=9, size=n_samples_per_group)
        middle_adult_hr = np.clip(middle_adult_hr, 55, 95)
        middle_adult_bo = np.random.normal(loc=97, scale=1, size=n_samples_per_group)
        middle_adult_bo = np.clip(middle_adult_bo, 94, 100)
        
        # Older adults (age 60+): HR 50-90 BPM, SpO2 93-100%
        older_adult_hr = np.random.normal(loc=70, scale=8, size=n_samples_per_group)
        older_adult_hr = np.clip(older_adult_hr, 50, 90)
        older_adult_bo = np.random.normal(loc=96.5, scale=1.2, size=n_samples_per_group)
        older_adult_bo = np.clip(older_adult_bo, 93, 100)
        
        # Athletes (lower normal heart rates): HR 40-80 BPM, SpO2 95-100%
        athlete_hr = np.random.normal(loc=60, scale=8, size=n_samples_per_group)
        athlete_hr = np.clip(athlete_hr, 40, 80)
        athlete_bo = np.random.normal(loc=98, scale=0.8, size=n_samples_per_group)
        athlete_bo = np.clip(athlete_bo, 95, 100)
        
        # Include some data for people with mild conditions
        # Higher heart rates (e.g., anxiety, mild hypertension): HR 70-120 BPM
        higher_hr_condition = np.random.normal(loc=95, scale=10, size=n_samples_per_group)
        higher_hr_condition = np.clip(higher_hr_condition, 70, 120)
        higher_hr_bo = np.random.normal(loc=97, scale=1, size=n_samples_per_group)
        higher_hr_bo = np.clip(higher_hr_bo, 94, 100)
        
        # Add synthetic data to arrays
        heart_rates.extend(list(children_hr))
        heart_rates.extend(list(young_adult_hr))
        heart_rates.extend(list(middle_adult_hr))
        heart_rates.extend(list(older_adult_hr))
        heart_rates.extend(list(athlete_hr))
        heart_rates.extend(list(higher_hr_condition))
        
        blood_oxygen.extend(list(children_bo))
        blood_oxygen.extend(list(young_adult_bo))
        blood_oxygen.extend(list(middle_adult_bo))
        blood_oxygen.extend(list(older_adult_bo))
        blood_oxygen.extend(list(athlete_bo))
        blood_oxygen.extend(list(higher_hr_bo))
        
        # Create training dataset
        training_data = np.column_stack((heart_rates, blood_oxygen))
        
        # Initialize and train the model
        # Use smaller contamination for condition-specific models to be more permissive
        contamination = 0.03 if condition else 0.05
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(training_data)
        
        # Store in class variable if it's the generic model
        if condition is None:
            cls._anomaly_detector = model
            logger.info("Generic anomaly detection model initialized with combined real and synthetic data")
        else:
            logger.info(f"Condition-specific anomaly detection model initialized for '{condition}'")
        
        return model
    
    @classmethod
    def get_user_baseline(cls, user_id):
        """
        Get or calculate personalized baseline metrics for a user
        
        Args:
            user_id (str): User ID
            
        Returns:
            dict: User's baseline metrics
        """
        # Return cached baseline if available
        if user_id in cls._user_baselines:
            # Check if baseline is still valid (recalculate every 24 hours)
            baseline = cls._user_baselines[user_id]
            if datetime.now() - baseline['calculated_at'] < timedelta(hours=24):
                return baseline
        
        # Get user's health history
        health_history = HealthData.get_by_user_id(user_id, limit=30)  # Use up to 30 recent readings
        
        # Default baseline values
        baseline = {
            'heart_rate': {'mean': 80, 'std': 10, 'min': 60, 'max': 100},
            'blood_oxygen': {'mean': 97, 'std': 1, 'min': 95, 'max': 100},
            'calculated_at': datetime.now(),
            'data_points': 0
        }
        
        # Return default baseline if no history available
        if not health_history or len(health_history) < 5:  # Need at least 5 data points
            logger.info(f"Using default baseline for user {user_id} - insufficient history")
            return baseline
        
        # Extract health metrics
        heart_rates = [entry['heart_rate'] for entry in health_history if 'heart_rate' in entry]
        blood_oxygen = [entry['blood_oxygen'] for entry in health_history if 'blood_oxygen' in entry]
        
        # Calculate personalized baseline if enough data available
        if len(heart_rates) >= 5:
            # For users with more data, we can be more confident in their true baseline
            # Adjust the outlier threshold based on the amount of data
            outlier_threshold = 3.0
            if len(heart_rates) >= 15:
                # More data = more permissive outlier detection
                outlier_threshold = 3.5
            if len(heart_rates) >= 25:
                # For users with lots of data, be even more permissive
                outlier_threshold = 4.0
            
            # Filter out extreme outliers (based on adaptive threshold)
            hr_array = np.array(heart_rates)
            hr_mean = np.mean(hr_array)
            hr_std = np.std(hr_array)
            filtered_hr = hr_array[np.abs(hr_array - hr_mean) <= outlier_threshold * hr_std]
            
            # If too many points were filtered out, use more points to avoid bias
            if len(filtered_hr) < len(hr_array) * 0.7:
                # If we're filtering out more than 30% of points, be more permissive
                filtered_hr = hr_array[np.abs(hr_array - hr_mean) <= (outlier_threshold + 1.0) * hr_std]
            
            baseline['heart_rate'] = {
                'mean': float(np.mean(filtered_hr)),
                'std': float(np.std(filtered_hr)),
                'min': float(np.percentile(filtered_hr, 5)),  # 5th percentile
                'max': float(np.percentile(filtered_hr, 95))  # 95th percentile
            }
            baseline['data_points'] = len(heart_rates)
            
            # Log the baseline calculation for debugging
            logger.info(f"User {user_id} heart rate baseline: {baseline['heart_rate']}")
            
        if len(blood_oxygen) >= 5:
            # Apply same adaptive approach to blood oxygen
            outlier_threshold = 3.0
            if len(blood_oxygen) >= 15:
                outlier_threshold = 3.5
            if len(blood_oxygen) >= 25:
                outlier_threshold = 4.0
                
            # Filter out extreme outliers
            bo_array = np.array(blood_oxygen)
            bo_mean = np.mean(bo_array)
            bo_std = np.std(bo_array)
            filtered_bo = bo_array[np.abs(bo_array - bo_mean) <= outlier_threshold * bo_std]
            
            # If too many points were filtered out, use more points
            if len(filtered_bo) < len(bo_array) * 0.7:
                filtered_bo = bo_array[np.abs(bo_array - bo_mean) <= (outlier_threshold + 1.0) * bo_std]
            
            baseline['blood_oxygen'] = {
                'mean': float(np.mean(filtered_bo)),
                'std': float(np.std(filtered_bo)),
                'min': float(np.percentile(filtered_bo, 5)),
                'max': float(np.percentile(filtered_bo, 95))
            }
            
            # Log the baseline calculation for debugging
            logger.info(f"User {user_id} blood oxygen baseline: {baseline['blood_oxygen']}")
        
        # Cache and return baseline
        cls._user_baselines[user_id] = baseline
        logger.info(f"Calculated personalized baseline for user {user_id} based on {baseline['data_points']} readings")
        
        return baseline
    
    @classmethod
    def simplified_risk_calculation(cls, heart_rate, blood_oxygen, user_condition=None, user_age=None):
        """
        A simplified approach to risk calculation using clear risk bands
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            user_condition (str, optional): User's medical condition
            user_age (int, optional): User's age
            
        Returns:
            dict: Contains risk_score (0-100) and risk factors explanation
        """
        # Initialize risk and explanations
        risk_score = 0
        risk_factors = []
        
        # Define normal ranges that will be adjusted by conditions
        hr_low = 60
        hr_high = 100
        bo_low = 95
        
        # Condition-based adjustments
        has_copd = False
        has_anxiety = False
        has_cardiac = False
        is_athlete = False
        
        # Apply condition adjustments
        if user_condition:
            conditions = user_condition.lower() if isinstance(user_condition, str) else ""
            
            # Check for specific conditions
            if 'copd' in conditions or 'emphysema' in conditions or 'bronchitis' in conditions:
                has_copd = True
                bo_low = 88  # COPD patients often have lower blood oxygen
                risk_factors.append("COPD condition factored in blood oxygen assessment")
                
            if 'anxiety' in conditions or 'stress' in conditions or 'panic' in conditions:
                has_anxiety = True
                hr_high = 110  # Anxiety can cause higher heart rates
                risk_factors.append("Anxiety condition factored in heart rate assessment")
                
            if 'heart disease' in conditions or 'hypertension' in conditions or 'cardiac' in conditions:
                has_cardiac = True
                # Heart conditions require more careful monitoring but same ranges
                risk_factors.append("Heart condition noted in assessment")
                
            if 'athlete' in conditions or 'very active' in conditions or 'fit' in conditions:
                is_athlete = True
                hr_low = 45  # Athletes often have lower resting heart rates
                risk_factors.append("Athletic condition factored in heart rate assessment")
        
        # Age-based adjustments
        if user_age:
            if user_age > 60:
                # Elderly people may have different normal ranges
                hr_high += 5  # Slightly higher upper limit for heart rate
                if not has_copd:
                    bo_low = 94  # Slightly lower acceptable blood oxygen
                risk_factors.append(f"Age {user_age} factored in vital signs assessment")
            elif user_age < 18:
                # Younger people tend to have higher heart rates
                hr_high += 10
                hr_low += 5
                risk_factors.append(f"Age {user_age} factored in heart rate assessment")
        
        # Calculate heart rate risk (0-50 scale)
        hr_risk = 0
        if heart_rate < hr_low:
            # Low heart rate risk
            deviation = hr_low - heart_rate
            if deviation <= 10:
                hr_risk = 10  # Minor
            elif deviation <= 20:
                hr_risk = 25  # Moderate
            else:
                hr_risk = 40  # Significant
            risk_factors.append(f"Heart rate {heart_rate} is below adjusted normal range ({hr_low}-{hr_high})")
        elif heart_rate > hr_high:
            # High heart rate risk
            deviation = heart_rate - hr_high
            if deviation <= 10:
                hr_risk = 10  # Minor
            elif deviation <= 20:
                hr_risk = 25  # Moderate
            else:
                hr_risk = 40  # Significant
            risk_factors.append(f"Heart rate {heart_rate} is above adjusted normal range ({hr_low}-{hr_high})")
        
        # Calculate blood oxygen risk (0-50 scale)
        bo_risk = 0
        if blood_oxygen < bo_low:
            # Low blood oxygen risk
            deviation = bo_low - blood_oxygen
            if deviation <= 2:
                bo_risk = 20  # Minor
            elif deviation <= 5:
                bo_risk = 35  # Moderate
            else:
                bo_risk = 50  # Significant
            risk_factors.append(f"Blood oxygen {blood_oxygen}% is below adjusted normal level ({bo_low}%)")
        
        # Calculate final risk score (0-100)
        risk_score = hr_risk + bo_risk
        
        # Add condition-specific context
        if has_copd and blood_oxygen >= bo_low:
            risk_factors.append(f"Blood oxygen {blood_oxygen}% is normal for someone with COPD")
        if has_anxiety and heart_rate <= hr_high:
            risk_factors.append(f"Heart rate {heart_rate} is normal for someone with anxiety")
        
        # Final risk assessment
        severity = "normal"
        if risk_score >= 70:
            severity = "severe"
        elif risk_score >= 40:
            severity = "moderate"
        elif risk_score >= 20:
            severity = "mild"
        else:
            severity = "normal"
            if len(risk_factors) == 0:
                risk_factors.append("All vital signs within normal ranges")
        
        return {
            "risk_score": risk_score,
            "severity": severity,
            "risk_factors": risk_factors
        }
        
    @classmethod
    def hybrid_risk_calculation(cls, heart_rate, blood_oxygen, ml_anomaly_score=None, is_anomaly=False, user_condition=None, user_age=None):
        """
        A hybrid approach combining rule-based and machine learning for risk assessment
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            ml_anomaly_score (float, optional): Anomaly score from machine learning (0-1 range)
            is_anomaly (bool): Whether the ML model detected an anomaly
            user_condition (str, optional): User's medical condition
            user_age (int, optional): User's age
            
        Returns:
            dict: Contains risk_score, severity, and risk factors
        """
        # Get rule-based risk calculation
        rule_result = cls.simplified_risk_calculation(
            heart_rate, 
            blood_oxygen, 
            user_condition=user_condition,
            user_age=user_age
        )
        
        # Extract values
        rule_risk_score = rule_result["risk_score"]
        rule_risk_factors = rule_result["risk_factors"]
        
        # Calculate final risk score combining rules and ML
        final_risk_score = rule_risk_score
        
        # If ML model detected an anomaly, increase risk score
        if is_anomaly:
            # Add ML anomaly explanation
            rule_risk_factors.append("Machine learning model detected unusual pattern in vital signs")
            
            # Apply ML influence based on current risk level
            if rule_risk_score < 20:
                # If risk is considered normal but ML detected anomaly, increase to at least mild
                final_risk_score = max(final_risk_score, 30)
            elif rule_risk_score < 40:
                # For mild risk, increase to at least moderate
                final_risk_score = max(final_risk_score, 45)
            else:
                # For already significant risk, add small increase
                final_risk_score = min(100, final_risk_score + 10)
        
        # If we have a specific anomaly score, use it to further influence the risk
        if ml_anomaly_score is not None:
            # Scale the anomaly score to have meaningful impact (0-30 points)
            ml_score_contribution = ml_anomaly_score * 30
            
            # Add weighted ML score to rule-based score
            # Weight of ML score increases with anomaly confidence
            ml_weight = 0.7 if ml_anomaly_score > 0.7 else (0.5 if ml_anomaly_score > 0.5 else 0.3)
            rule_weight = 1.0 - ml_weight
            
            # Calculate weighted score
            weighted_score = (rule_risk_score * rule_weight) + (ml_score_contribution * ml_weight)
            final_risk_score = min(100, weighted_score)
            
            # Add explanation if ML significantly influenced the score
            if ml_score_contribution > 15 and ml_weight >= 0.5:
                rule_risk_factors.append(f"Machine learning analysis strongly influenced risk score (confidence: {ml_anomaly_score:.2f})")
        
        # Determine severity level based on final risk score
        severity = "normal"
        if final_risk_score >= 70:
            severity = "severe"
        elif final_risk_score >= 40:
            severity = "moderate"
        elif final_risk_score >= 20:
            severity = "mild"
            
        return {
            "risk_score": final_risk_score,
            "severity": severity,
            "risk_factors": rule_risk_factors
        }

    @classmethod
    def analyze_health_data(cls, user_id, heart_rate, blood_oxygen, additional_metrics=None):
        """
        Analyze health data and save to database
        
        Args:
            user_id (str): User ID
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            additional_metrics (dict, optional): Additional health metrics
            
        Returns:
            dict: Analysis result
        """
        try:
            # Get user for condition information
            user = User.get_by_id(user_id)
            user_condition = None
            user_age = None
            user_gender = None
            user_activity_level = None
            
            # Extract user information
            if user:
                # Get medical conditions
                if 'medical_conditions' in user and user['medical_conditions']:
                    if isinstance(user['medical_conditions'], list) and user['medical_conditions']:
                        user_condition = user['medical_conditions'][0]
                    elif isinstance(user['medical_conditions'], str) and user['medical_conditions']:
                        user_condition = user['medical_conditions']
                
                # Get age
                if 'age' in user and user['age']:
                    user_age = int(user['age'])
                
                # Get gender
                if 'gender' in user and user['gender']:
                    user_gender = user['gender']
                
                # Get activity level
                if 'activity_level' in user and user['activity_level']:
                    user_activity_level = user['activity_level']
                
                # If no specific condition found, try medical_history
                if not user_condition and 'medical_history' in user:
                    # If medical_history is a dictionary with a 'conditions' key
                    if isinstance(user['medical_history'], dict) and 'conditions' in user['medical_history']:
                        conditions = user['medical_history']['conditions']
                        if isinstance(conditions, list) and conditions:
                            user_condition = conditions[0]
                        elif isinstance(conditions, str) and conditions:
                            user_condition = conditions
                    # If medical_history is a string, try to extract a condition
                    elif isinstance(user['medical_history'], str) and user['medical_history']:
                        # Simple condition extraction from text
                        medical_history = user['medical_history'].lower()
                        for condition in ['asthma', 'diabetes', 'hypertension', 'heart disease', 
                                         'copd', 'sleep apnea', 'anemia']:
                            if condition in medical_history:
                                user_condition = condition
                                break
            
            # Get anomaly detector model
            anomaly_detector = cls.get_anomaly_detector()
            
            # Check if we're using the new model
            is_anomaly = False
            anomaly_confidence = 0.0
            
            # ML models for anomaly detection
            if cls._using_new_model and cls._scaler is not None and cls._feature_names is not None:
                # Prepare features for the new model
                features_dict = {
                    'heart_rate': heart_rate,
                    'blood_oxygen': blood_oxygen,
                    'age': user_age if user_age else 35,  # Default age if missing
                    'gender': 1 if user_gender and user_gender.lower() == 'female' else 0
                }
                
                # Add one-hot encoded condition features
                for condition in CONDITION_EFFECTS.keys():
                    condition_key = f"condition_{condition}"
                    features_dict[condition_key] = 1 if condition in str(user_condition).lower() else 0
                
                # Add one-hot encoded activity features
                for activity in ACTIVITY_EFFECTS.keys():
                    activity_key = f"activity_{activity}"
                    features_dict[activity_key] = 1 if activity == user_activity_level else 0
                
                # Create DataFrame and ensure it has all the expected features
                feature_df = pd.DataFrame([features_dict])
                
                # Make sure all expected features are present
                for feature in cls._feature_names:
                    if feature not in feature_df.columns:
                        feature_df[feature] = 0
                
                # Reorder columns to match the expected order
                feature_df = feature_df[cls._feature_names]
                
                # Scale features
                features_scaled = cls._scaler.transform(feature_df)
                
                # Use the model for prediction
                if hasattr(anomaly_detector, 'predict_proba'):
                    # For RandomForestClassifier
                    anomaly_probs = anomaly_detector.predict_proba(features_scaled)
                    # Assuming the second column is the probability of being an anomaly
                    anomaly_confidence = anomaly_probs[0][1] if anomaly_probs.shape[1] > 1 else 0
                    is_anomaly = anomaly_confidence >= 0.6  # Threshold can be adjusted
                else:
                    # Fallback for other model types
                    prediction = anomaly_detector.predict(features_scaled)
                    is_anomaly = prediction[0] == 1  # For RandomForest, 1 is anomaly
                    anomaly_confidence = 0.7 if is_anomaly else 0.0  # Default confidence when no probs available
                
                logger.info(f"Using new ML model with confidence: {anomaly_confidence:.2f}")
            else:
                # Use Isolation Forest for anomaly detection - more significant role now
                features = np.array([[heart_rate, blood_oxygen]])
                
                # Get anomaly score and prediction from Isolation Forest
                anomaly_score = anomaly_detector.score_samples(features)
                prediction = anomaly_detector.predict(features)
                is_anomaly = prediction[0] == -1  # For IsolationForest, -1 is anomaly
                
                # Convert anomaly score to a 0-1 range (IsolationForest scores are negative, with more negative being more anomalous)
                # Typical range is from -0.1 (normal) to -0.5 or lower (very anomalous)
                normalized_score = min(1.0, max(0.0, -anomaly_score[0] * 2))
                anomaly_confidence = normalized_score
                
                logger.info(f"Using Isolation Forest with calculated anomaly score: {normalized_score:.2f}")
            
            # Use hybrid approach combining rules and ML for risk calculation
            risk_result = cls.hybrid_risk_calculation(
                heart_rate,
                blood_oxygen,
                ml_anomaly_score=anomaly_confidence,
                is_anomaly=is_anomaly,
                user_condition=user_condition,
                user_age=user_age
            )
            
            risk_score = risk_result["risk_score"]
            severity = risk_result["severity"]
            risk_factors = risk_result["risk_factors"]
            
            # Get user context for AI analysis
            user_context = {
                'age': user_age,
                'gender': user_gender,
                'activity_level': user_activity_level,
                'medical_condition': user_condition
            }
            
            # Prepare health data for AI analysis
            health_data = {
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen
            }
            if additional_metrics and isinstance(additional_metrics, dict):
                for key, value in additional_metrics.items():
                    health_data[key] = value
            
            # Get AI-generated recommendations
            ai_analysis = gemini.generate_health_advice(health_data, user_context)
            
            # Get recommendations based on risk score and metrics
            recommendations = cls.generate_recommendations(
                heart_rate, 
                blood_oxygen, 
                risk_score, 
                user_condition=user_condition
            )
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': bool(is_anomaly),
                'anomaly_confidence': float(anomaly_confidence),
                'ml_contribution': "significant" if anomaly_confidence > 0.6 else "moderate" if anomaly_confidence > 0.3 else "minor",
                'risk_score': risk_score,
                'severity': severity,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'ai_analysis': ai_analysis
            }
            
            # Prepare additional metrics for saving to database
            metrics_to_save = {}
            if additional_metrics:
                metrics_to_save.update(additional_metrics)
            
            # Store ML anomaly information
            metrics_to_save['ml_analysis'] = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_confidence': float(anomaly_confidence),
                'model_type': 'RandomForestClassifier' if cls._using_new_model else 'IsolationForest'
            }
            
            # Store analysis results and recommendations at top level and in nested object
            metrics_to_save['analysis_result'] = {
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score,
                'severity': severity,
                'risk_factors': risk_factors,
                'recommendations': recommendations,  # Include recommendations in analysis_result
                'using_new_model': cls._using_new_model
            }
            
            # Save data to database with recommendations at top level 
            health_data_id = HealthData.create(
                user_id=user_id,
                heart_rate=heart_rate,
                blood_oxygen=blood_oxygen,
                additional_metrics=metrics_to_save
            )
            
            # Update the document to add recommendations at top level
            HealthData.update(health_data_id, {
                'recommendations': recommendations,
                'is_anomaly': bool(is_anomaly),
                'anomaly_confidence': float(anomaly_confidence),
                'risk_score': risk_score,
                'severity': severity,
                'risk_factors': risk_factors,
                'ai_analysis': ai_analysis,
                'using_new_model': cls._using_new_model
            })
            
            result['health_data_id'] = health_data_id
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing health data: {e}")
            return {
                'error': str(e),
                'recommendations': ["Unable to analyze health data. Please try again later."]
            }
    
    @staticmethod
    def generate_recommendations(heart_rate, blood_oxygen, risk_score, user_condition=None):
        """
        Generate health recommendations based on measured metrics and risk score
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            risk_score (float): Calculated risk score
            user_condition (str, optional): User's medical condition
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        # Determine severity level based on risk score
        severity = "normal"
        if risk_score >= 70:
            severity = "severe"
        elif risk_score >= 40:
            severity = "moderate"
        elif risk_score >= 20:
            severity = "mild"
        
        # General recommendations based on risk
        if risk_score < 20:
            recommendations.append("Your vitals are within normal range. Continue with your regular health routine.")
        elif risk_score < 40:
            recommendations.append(f"Minor {severity} deviations in your vitals detected. Monitor for any changes.")
        else:
            recommendations.append(f"{severity.capitalize()} deviations detected. Consider consulting a healthcare professional.")
        
        # Heart rate specific recommendations
        hr_high = 100
        hr_low = 60
        bo_low = 95
        
        # Adjust thresholds based on condition
        if user_condition:
            condition = user_condition.lower() if isinstance(user_condition, str) else ""
            
            if 'anxiety' in condition or 'stress' in condition:
                hr_high = 110
            if 'athlete' in condition or 'very active' in condition:
                hr_low = 50
            if 'copd' in condition or 'emphysema' in condition:
                bo_low = 88
            if 'sleep apnea' in condition:
                bo_low = 90
        
        # Heart rate recommendations
        if heart_rate > hr_high:
            # Check for specific conditions that affect heart rate interpretation
            if user_condition and ('tachycardia' in str(user_condition).lower() or 'anxiety' in str(user_condition).lower()):
                recommendations.append("Your heart rate is elevated, which is expected given your condition. If you feel any discomfort, consider using relaxation techniques.")
            else:
                recommendations.append("Your heart rate is elevated. Try relaxation techniques or reduce physical activity.")
                if heart_rate > hr_high + 20:
                    recommendations.append("Significantly elevated heart rate. If this persists while resting, consider medical attention.")
        elif heart_rate < hr_low:
            # Check for conditions where low heart rate is expected
            if user_condition and ('bradycardia' in str(user_condition).lower() or 'athlete' in str(user_condition).lower()):
                recommendations.append("Your heart rate is below average, which is consistent with your condition/fitness level. No immediate action needed if you feel normal.")
            else:
                recommendations.append("Your heart rate is lower than normal. Ensure you're staying hydrated and consider rest.")
                if heart_rate < hr_low - 10:
                    recommendations.append("Significantly low heart rate. If you feel dizzy or weak, consider medical attention.")
        
        # Blood oxygen specific recommendations
        if blood_oxygen < bo_low:
            # For COPD patients
            if user_condition and ('copd' in str(user_condition).lower() or 'emphysema' in str(user_condition).lower()):
                if blood_oxygen < bo_low - 3:
                    recommendations.append("Your blood oxygen is below your condition's baseline. Consider using prescribed treatments or seek medical attention.")
                else:
                    recommendations.append("Your blood oxygen is lower than optimal. Try breathing exercises and ensure proper medication use.")
            else:
                if blood_oxygen < 92:
                    recommendations.append("Low blood oxygen levels. If you're experiencing shortness of breath, consider medical attention.")
                else:
                    recommendations.append("Your blood oxygen is slightly low. Try deep breathing exercises or improve ventilation.")
        
        # Condition-specific general recommendations
        if user_condition:
            condition = user_condition.lower() if isinstance(user_condition, str) else ""
            
            if 'heart disease' in condition or 'hypertension' in condition:
                recommendations.append("Remember to take your prescribed medications and follow your heart-healthy diet plan.")
                
            elif 'diabetes' in condition:
                recommendations.append("Remember to monitor your blood glucose levels regularly alongside your heart metrics.")
                
            elif 'asthma' in condition:
                recommendations.append("Keep your rescue inhaler accessible, especially if you notice changes in your breathing patterns.")
                
            elif 'copd' in condition or 'emphysema' in condition:
                recommendations.append("Practice your prescribed breathing exercises regularly and monitor for increased mucus production or coughing.")
                
            elif 'sleep apnea' in condition:
                recommendations.append("Ensure consistent use of your CPAP device and maintain a regular sleep schedule.")
                
            elif 'anxiety' in condition or 'panic' in condition:
                recommendations.append("Practice mindfulness or your prescribed anxiety management techniques, especially when noticing elevated heart rate.")
        
        # Add general wellness recommendations
        if len(recommendations) < 4:  # Add extra general recommendations if we don't have many specific ones
            general_recommendations = [
                "Stay hydrated by drinking at least 8 glasses of water daily.",
                "Aim for 7-9 hours of quality sleep each night.",
                "Regular physical activity can improve your overall cardiovascular health.",
                "Practice stress-reduction techniques like deep breathing or meditation.",
                "Maintain a balanced diet rich in fruits, vegetables, and whole grains."
            ]
            
            # Add general recommendations until we have enough
            for rec in general_recommendations:
                if rec not in recommendations and len(recommendations) < 4:
                    recommendations.append(rec)
        
        return recommendations
    
    @classmethod
    def get_user_health_history(cls, user_id, limit=10):
        """
        Get health history for a user
        
        Args:
            user_id (str): User ID
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of health data records
        """
        try:
            return HealthData.get_by_user_id(user_id, limit=limit)
        except Exception as e:
            logger.error(f"Error getting user health history: {e}")
            return []
    
    @classmethod
    def get_health_trends(cls, user_id, days=30):
        """
        Analyze health data trends for a user over a specified period
        
        Args:
            user_id (str): User ID
            days (int): Number of days to analyze
            
        Returns:
            dict: Health trends analysis
        """
        try:
            # Get health data for the specified period
            collection = HealthData.get_collection()
            start_date = datetime.now() - timedelta(days=days)
            
            # Query database for data points in the period
            cursor = collection.find({
                'user_id': user_id,
                'created_at': {'$gte': start_date}
            }).sort('created_at', 1)  # Sort by timestamp ascending
            
            # Convert to list
            data_points = list(cursor)
            
            if not data_points:
                return {
                    'error': 'Not enough data for analysis',
                    'message': f'No health data available for the past {days} days'
                }
            
            # Extract heart rate and blood oxygen values
            heart_rates = [float(dp.get('heart_rate', 0)) for dp in data_points if 'heart_rate' in dp]
            blood_oxygen = [float(dp.get('blood_oxygen', 0)) for dp in data_points if 'blood_oxygen' in dp]
            
            if not heart_rates or not blood_oxygen:
                return {
                    'error': 'Incomplete data for analysis',
                    'message': 'Missing heart rate or blood oxygen data'
                }
                
            # Calculate statistics
            hr_stats = {
                'mean': np.mean(heart_rates),
                'std': np.std(heart_rates),
                'min': np.min(heart_rates),
                'max': np.max(heart_rates)
            }
            
            bo_stats = {
                'mean': np.mean(blood_oxygen),
                'std': np.std(blood_oxygen),
                'min': np.min(blood_oxygen),
                'max': np.max(blood_oxygen)
            }
            
            # Determine trends
            if len(heart_rates) >= 3:
                # Calculate linear regression slope for heart rate trend
                x = list(range(len(heart_rates)))
                hr_coef = np.polyfit(x, heart_rates, 1)[0]
                hr_trend = "increasing" if hr_coef > 0.5 else "decreasing" if hr_coef < -0.5 else "stable"
                
                # Calculate linear regression slope for blood oxygen trend
                bo_coef = np.polyfit(x, blood_oxygen, 1)[0]
                bo_trend = "increasing" if bo_coef > 0.05 else "decreasing" if bo_coef < -0.05 else "stable"
            else:
                hr_trend = "stable"
                bo_trend = "stable"
                
            # Add trends to stats
            hr_stats['trend'] = hr_trend
            bo_stats['trend'] = bo_trend
            
            # Create response
            return {
                'days_analyzed': days,
                'data_points': len(data_points),
                'heart_rate': hr_stats,
                'blood_oxygen': bo_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing health trends: {e}")
            return {
                'error': str(e),
                'message': 'Failed to analyze health trends'
            }
    
    @classmethod
    def reset_user_baseline(cls, user_id):
        """
        Reset a user's personalized baseline (useful for users with medical changes or errors in data)
        
        Args:
            user_id (str): User ID
            
        Returns:
            bool: Success status
        """
        try:
            # Remove from baseline cache if exists
            if user_id in cls._user_baselines:
                del cls._user_baselines[user_id]
                logger.info(f"Reset personalized baseline for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting user baseline: {e}")
            return False
            
    @classmethod
    def manually_set_user_baseline(cls, user_id, hr_mean=None, hr_std=None, hr_min=None, hr_max=None, 
                                 bo_mean=None, bo_std=None, bo_min=None, bo_max=None):
        """
        Manually set a user's baseline values (for users with known medical conditions)
        
        Args:
            user_id (str): User ID
            hr_mean (float, optional): Mean heart rate
            hr_std (float, optional): Standard deviation of heart rate
            hr_min (float, optional): Minimum normal heart rate
            hr_max (float, optional): Maximum normal heart rate
            bo_mean (float, optional): Mean blood oxygen
            bo_std (float, optional): Standard deviation of blood oxygen
            bo_min (float, optional): Minimum normal blood oxygen
            bo_max (float, optional): Maximum normal blood oxygen
            
        Returns:
            dict: Updated baseline
        """
        try:
            # Get current baseline
            baseline = cls.get_user_baseline(user_id)
            
            # Update heart rate values if provided
            if hr_mean is not None:
                baseline['heart_rate']['mean'] = float(hr_mean)
            if hr_std is not None:
                baseline['heart_rate']['std'] = float(hr_std)
            if hr_min is not None:
                baseline['heart_rate']['min'] = float(hr_min)
            if hr_max is not None:
                baseline['heart_rate']['max'] = float(hr_max)
                
            # Update blood oxygen values if provided
            if bo_mean is not None:
                baseline['blood_oxygen']['mean'] = float(bo_mean)
            if bo_std is not None:
                baseline['blood_oxygen']['std'] = float(bo_std)
            if bo_min is not None:
                baseline['blood_oxygen']['min'] = float(bo_min)
            if bo_max is not None:
                baseline['blood_oxygen']['max'] = float(bo_max)
            
            # Mark as manually set and recalculated
            baseline['calculated_at'] = datetime.now()
            baseline['manually_set'] = True
            baseline['data_points'] = max(baseline['data_points'], 25)  # Treat as well-established
            
            # Save to cache
            cls._user_baselines[user_id] = baseline
            
            logger.info(f"Manually set baseline for user {user_id}: HR {baseline['heart_rate']}, BO {baseline['blood_oxygen']}")
            
            return baseline
        except Exception as e:
            logger.error(f"Error setting user baseline: {e}")
            return None 