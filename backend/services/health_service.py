import numpy as np
from sklearn.ensemble import IsolationForest
import logging
from datetime import datetime, timedelta
import pandas as pd
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
            
            # Calculate base risk score
            risk_score = cls.calculate_risk_score(heart_rate, blood_oxygen)
            
            # If the ML model detected an anomaly, increase the risk score
            if is_anomaly and risk_score < 40:
                # Boost the risk score if ML detected an anomaly but calculated risk is low
                risk_score = max(risk_score, 40)  # Ensure at least moderate risk for ML-detected anomalies
            
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
            
            # Get recommendations based on risk score and metrics
            recommendations = cls.generate_recommendations(risk_score, heart_rate, blood_oxygen)
            
            # Store analysis results and recommendations at top level and in nested object
            metrics_to_save['analysis_result'] = {
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score,
                'recommendations': recommendations  # Include recommendations in analysis_result
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
                'risk_score': risk_score,
                'ai_analysis': ai_analysis
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
        
        # Calculate heart rate risk with improved sensitivity to high heart rates
        if hr_normal_low <= heart_rate <= hr_normal_high:
            hr_risk = 0
        else:
            # Calculate deviation and apply a non-linear scaling for higher values
            if heart_rate > hr_normal_high:
                # Higher risk for elevated heart rates (exponential scaling)
                hr_deviation = heart_rate - hr_normal_high
                hr_risk = min(100, 25 + (hr_deviation / 10) * 75)  # Start at 25% risk at 101 BPM
            else:
                # Lower heart rate (below normal)
                hr_deviation = hr_normal_low - heart_rate
                hr_risk = min(100, (hr_deviation / 20) * 100)
        
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
        
        # Add specific heart rate recommendations regardless of overall risk
        elif heart_rate > 100 and heart_rate <= 150:
            # Specific recommendations for elevated heart rate
            hr_severity = "significantly " if heart_rate > 120 else ""
            recommendations.extend([
                f"Heart rate is {hr_severity}elevated at {int(heart_rate)} BPM",
                "Consider resting and monitoring for other symptoms",
                "If heart rate remains elevated, contact your healthcare provider"
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