import numpy as np
import logging
from datetime import datetime, timedelta
from models.health_data import HealthData

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Service for feature extraction and engineering for health data analysis"""
    
    @staticmethod
    def extract_features(heart_rate, blood_oxygen, additional_metrics=None, user_context=None):
        """
        Extract features from health data for analysis
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            additional_metrics (dict, optional): Additional health metrics
            user_context (dict, optional): User context information
            
        Returns:
            list: Feature vector for model input
        """
        features = [heart_rate, blood_oxygen]
        
        # Process health conditions for additional features
        health_conditions_features = [0.0, 0.0, 0.0, 0.0]  # Default: no conditions
        
        if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
            conditions_text = " ".join([c.lower() for c in user_context['health_conditions']])
            
            # Set condition indicators
            has_copd = 1.0 if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']) else 0.0
            has_anxiety = 1.0 if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']) else 0.0
            has_heart_issue = 1.0 if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']) else 0.0
            is_athlete = 1.0 if 'athlete' in conditions_text else 0.0
            
            health_conditions_features = [has_copd, has_anxiety, has_heart_issue, is_athlete]
            
            logger.info(f"Extracted condition features: COPD={has_copd}, Anxiety={has_anxiety}, Heart={has_heart_issue}, Athlete={is_athlete}")
        
        # Add condition features
        features.extend(health_conditions_features)
        
        # Process additional metrics if available
        if additional_metrics and isinstance(additional_metrics, dict):
            # Extract any numerical features that might be useful
            if 'activity_level' in additional_metrics:
                try:
                    activity_level = float(additional_metrics['activity_level'])
                    features.append(activity_level)
                    logger.debug(f"Added activity_level feature: {activity_level}")
                except (ValueError, TypeError):
                    features.append(0.0)  # Default value
            else:
                features.append(0.0)  # Default value for activity level
            
            # Extract temperature if available
            if 'temperature' in additional_metrics:
                try:
                    temperature = float(additional_metrics['temperature'])
                    features.append(temperature)
                    logger.debug(f"Added temperature feature: {temperature}")
                except (ValueError, TypeError):
                    features.append(0.0)  # Default value
            else:
                features.append(0.0)  # Default value for temperature
        else:
            # Add default values for missing additional metrics
            features.extend([0.0, 0.0])  # Default for activity_level and temperature
        
        # Log feature extraction
        logger.debug(f"Extracted features: {features}")
        
        return features
    
    @staticmethod
    def get_historical_features(user_id, days=30):
        """
        Get historical health data features for a user
        
        Args:
            user_id (str): User ID
            days (int): Number of days to look back
            
        Returns:
            list: List of feature vectors
        """
        try:
            # Get health data for the specified period
            collection = HealthData.get_collection()
            start_date = datetime.now() - timedelta(days=days)
            
            # Query database for data points in the period
            cursor = collection.find({
                'user_id': user_id,
                'created_at': {'$gte': start_date}
            }).sort('created_at', -1)  # Sort by timestamp descending
            
            # Convert to list
            data_points = list(cursor)
            
            if not data_points:
                logger.info(f"No historical data found for user {user_id}")
                return []
            
            # Extract features from each data point
            features_list = []
            
            for dp in data_points:
                if 'heart_rate' in dp and 'blood_oxygen' in dp:
                    heart_rate = float(dp.get('heart_rate', 0))
                    blood_oxygen = float(dp.get('blood_oxygen', 0))
                    
                    # Extract additional metrics if available
                    additional_metrics = {}
                    for key in ['activity_level', 'temperature']:
                        if key in dp:
                            additional_metrics[key] = dp[key]
                    
                    # Create basic feature vector
                    feature_vector = [heart_rate, blood_oxygen]
                    
                    # Add risk class if available
                    if 'risk_class' in dp:
                        risk_class = int(dp['risk_class'])
                    elif 'analysis_result' in dp and 'risk_class' in dp['analysis_result']:
                        risk_class = int(dp['analysis_result']['risk_class'])
                    else:
                        risk_class = None
                    
                    features_list.append({
                        'features': feature_vector,
                        'risk_class': risk_class,
                        'timestamp': dp.get('created_at')
                    })
            
            logger.info(f"Retrieved {len(features_list)} historical feature vectors for user {user_id}")
            return features_list
            
        except Exception as e:
            logger.error(f"Error retrieving historical features: {e}")
            return []