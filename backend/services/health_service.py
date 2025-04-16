import numpy as np
from sklearn.ensemble import IsolationForest
import logging
from datetime import datetime
from config import DEBUG
from models.health_data import HealthData
from models.user import User
from gemini_client import gemini

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthService:
    """Service for handling health data analysis"""
    
    # Initialize anomaly detection model
    _anomaly_detector = None
    
    @classmethod
    def get_anomaly_detector(cls):
        """Get or initialize the anomaly detection model"""
        if cls._anomaly_detector is None:
            # Generate some sample training data for the model
            # These are example normal ranges for heart rate (60-100) and blood oxygen (95-100)
            np.random.seed(42)
            n_samples = 1000
            normal_heart_rates = np.random.uniform(60, 100, n_samples)
            normal_blood_oxygen = np.random.uniform(95, 100, n_samples)
            training_data = np.column_stack((normal_heart_rates, normal_blood_oxygen))
            
            # Initialize and train the model
            cls._anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            cls._anomaly_detector.fit(training_data)
            logger.info("Anomaly detection model initialized")
        
        return cls._anomaly_detector
    
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
            # Get anomaly detector
            anomaly_detector = cls.get_anomaly_detector()
            
            # Prepare data for analysis
            features = np.array([[heart_rate, blood_oxygen]])
            
            # Detect anomalies
            prediction = anomaly_detector.predict(features)
            is_anomaly = prediction[0] == -1
            
            # Calculate risk score
            risk_score = cls.calculate_risk_score(heart_rate, blood_oxygen)
            
            # Get user context for AI analysis
            user = User.get_by_id(user_id)
            user_context = {}
            if user:
                if 'age' in user:
                    user_context['age'] = user['age']
                if 'medical_history' in user:
                    user_context['medical_history'] = user['medical_history']
            
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
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score,
                'recommendations': cls.generate_recommendations(risk_score, heart_rate, blood_oxygen),
                'ai_analysis': ai_analysis
            }
            
            # Prepare additional metrics for saving to database
            metrics_to_save = {}
            if additional_metrics:
                metrics_to_save.update(additional_metrics)
            metrics_to_save['analysis_result'] = {
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score
            }
            
            # Save data to database
            health_data_id = HealthData.create(
                user_id=user_id,
                heart_rate=heart_rate,
                blood_oxygen=blood_oxygen,
                additional_metrics=metrics_to_save
            )
            
            result['health_data_id'] = health_data_id
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing health data: {e}")
            return {
                'error': str(e),
                'recommendations': ["Unable to analyze health data. Please try again later."]
            }
    
    @staticmethod
    def calculate_risk_score(heart_rate, blood_oxygen):
        """
        Calculate risk score based on health metrics
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            
        Returns:
            float: Risk score (0-100)
        """
        # Define normal ranges
        hr_normal_low, hr_normal_high = 60, 100
        bo_normal_low = 95
        
        # Calculate heart rate risk
        if hr_normal_low <= heart_rate <= hr_normal_high:
            hr_risk = 0
        else:
            # Calculate how far from normal range
            hr_deviation = min(abs(heart_rate - hr_normal_low), 
                             abs(heart_rate - hr_normal_high))
            hr_risk = min(100, (hr_deviation / 20) * 100)  # 20 BPM deviation = 100% risk
        
        # Calculate blood oxygen risk
        if blood_oxygen >= bo_normal_low:
            bo_risk = 0
        else:
            bo_deviation = bo_normal_low - blood_oxygen
            bo_risk = min(100, (bo_deviation / 5) * 100)  # 5% deviation = 100% risk
        
        # Weighted average (blood oxygen is more critical)
        return (hr_risk * 0.4 + bo_risk * 0.6)
    
    @staticmethod
    def generate_recommendations(risk_score, heart_rate, blood_oxygen):
        """
        Generate recommendations based on risk score and health metrics
        
        Args:
            risk_score (float): Risk score
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Severe conditions requiring immediate attention
        if blood_oxygen < 90 or heart_rate > 150 or heart_rate < 40:
            recommendations.extend([
                "URGENT: Immediate medical attention required",
                "Contact emergency services immediately",
                f"Critical values detected: HR={heart_rate}, SpO2={blood_oxygen}%"
            ])
        
        # High risk conditions
        elif risk_score > 70:
            recommendations.extend([
                "Contact your healthcare provider soon",
                "Monitor vital signs closely",
                "Rest and avoid physical exertion"
            ])
        
        # Moderate risk conditions
        elif risk_score > 40:
            recommendations.extend([
                "Continue monitoring your vital signs",
                "Consider contacting your healthcare provider if symptoms persist",
                "Take rest and stay hydrated"
            ])
        
        # Low risk or normal conditions
        else:
            recommendations.extend([
                "Vital signs are within normal range",
                "Continue normal activities",
                "Stay hydrated and maintain regular monitoring"
            ])
        
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