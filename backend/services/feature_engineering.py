import numpy as np
from datetime import datetime, timedelta
from models.health_data import HealthData

class FeatureEngineering:
    """Feature engineering for health data"""
    
    @staticmethod
    def extract_features(heart_rate, blood_oxygen, additional_metrics=None, user_context=None):
        """Extract advanced features from raw health data"""
        # Base features
        features = [heart_rate, blood_oxygen]
        
        # Age-related feature
        if user_context and 'age' in user_context:
            age = user_context['age']
            hr_age_ratio = heart_rate / max(1, age) 
            features.append(hr_age_ratio)
        
        # Blood oxygen deviation
        bo_drop = 100 - blood_oxygen
        features.append(bo_drop)
        
        # Health condition indicators
        if user_context and 'health_conditions' in user_context:
            health_conditions = user_context['health_conditions']
            health_conditions_text = " ".join(health_conditions).lower()
            
            heart_condition = 1.0 if any(c in health_conditions_text for c in 
                                    ['heart', 'cardiac', 'arrhythmia']) else 0.0
            respiratory_condition = 1.0 if any(c in health_conditions_text for c in 
                                        ['asthma', 'copd', 'respiratory']) else 0.0
            anxiety_condition = 1.0 if any(c in health_conditions_text for c in 
                                    ['anxiety', 'panic', 'stress disorder']) else 0.0
            diabetes_condition = 1.0 if 'diabetes' in health_conditions_text else 0.0
            
            features.extend([heart_condition, respiratory_condition, anxiety_condition, diabetes_condition])
        
        # Add additional metrics as features
        if additional_metrics:
            if 'temperature' in additional_metrics:
                temp = float(additional_metrics['temperature'])
                temp_deviation = abs(temp - 37.0)  # Deviation from normal body temperature
                features.append(temp_deviation)
            
            if 'blood_pressure_systolic' in additional_metrics and 'blood_pressure_diastolic' in additional_metrics:
                systolic = float(additional_metrics['blood_pressure_systolic'])
                diastolic = float(additional_metrics['blood_pressure_diastolic'])
                pulse_pressure = systolic - diastolic
                features.append(pulse_pressure)
        
        return features
    
    @staticmethod
    def get_historical_features(user_id, days=30):
        """Extract features from historical health data"""
        # Get recent health data
        collection = HealthData.get_collection()
        start_date = datetime.now() - timedelta(days=days)
        
        cursor = collection.find({
            'user_id': user_id,
            'created_at': {'$gte': start_date}
        }).sort('created_at', 1)
        
        data_points = list(cursor)
        
        if not data_points:
            return []
        
        # Extract metric histories
        heart_rates = [float(dp.get('heart_rate', 0)) for dp in data_points if 'heart_rate' in dp]
        blood_oxygen = [float(dp.get('blood_oxygen', 0)) for dp in data_points if 'blood_oxygen' in dp]
        
        if not heart_rates or not blood_oxygen:
            return []
        
        hr_mean = np.mean(heart_rates)
        hr_std = np.std(heart_rates)
        hr_trend = 0
        
        bo_mean = np.mean(blood_oxygen)
        bo_std = np.std(blood_oxygen)
        bo_trend = 0
        
        if len(heart_rates) >= 3:
            x = list(range(len(heart_rates)))
            hr_trend = np.polyfit(x, heart_rates, 1)[0]
            bo_trend = np.polyfit(x, blood_oxygen, 1)[0]
        
        return [hr_mean, hr_std, hr_trend, bo_mean, bo_std, bo_trend]