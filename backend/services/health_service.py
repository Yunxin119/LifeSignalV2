import numpy as np
from sklearn.ensemble import IsolationForest
import logging
from datetime import datetime, timedelta
import pandas as pd
from config import DEBUG
from models.health_data import HealthData
from models.user import User
from gemini_client import gemini

# Setup logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthService:
    """Health data analysis service"""
    
    # Anomaly detection model
    _anomaly_detector = None
    
    @classmethod
    def get_anomaly_detector(cls):
        """Initialize or return the anomaly detection model"""
        if cls._anomaly_detector is None:
            # Generate training data with normal heart rate and blood oxygen ranges
            np.random.seed(42)
            n_samples = 1000
            normal_heart_rates = np.random.uniform(60, 100, n_samples)
            normal_blood_oxygen = np.random.uniform(95, 100, n_samples)
            training_data = np.column_stack((normal_heart_rates, normal_blood_oxygen))
            
            # Train the model
            cls._anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            cls._anomaly_detector.fit(training_data)
            logger.info("Anomaly detection model initialized")
        
        return cls._anomaly_detector
    
    @classmethod
    def analyze_health_data(cls, user_id, heart_rate, blood_oxygen, additional_metrics=None):
        """Analyze health data using machine learning and rules"""
        try:
            logger.info(f"[ANALYSIS_START] User: {user_id}, HR: {heart_rate}, SpO2: {blood_oxygen}")
            start_time = datetime.now()
            
            # Get user context
            user = User.get_by_id(user_id)
            user_context = {}
            if user:
                if 'age' in user:
                    user_context['age'] = user['age']
                if 'health_conditions' in user:
                    user_context['health_conditions'] = user['health_conditions']
                if 'medical_history' in user:
                    user_context['medical_history'] = user['medical_history']
            
            # Log health conditions for personalized analysis
            if 'health_conditions' in user_context and user_context['health_conditions']:
                logger.info(f"[USER_CONTEXT] Analyzing health data with conditions: {user_context['health_conditions']}")
            if 'age' in user_context:
                logger.info(f"[USER_CONTEXT] User age: {user_context['age']}")
                    
            # Extract features
            from services.feature_engineering import FeatureEngineering
            features = FeatureEngineering.extract_features(
                heart_rate, blood_oxygen, additional_metrics, user_context
            )
            logger.info(f"[FEATURES] Extracted {len(features)} features")
            
            # Get historical features if available
            historical_features = FeatureEngineering.get_historical_features(user_id)
            history_count = len(historical_features)
            logger.info(f"[HISTORY] Found {history_count} historical data points for user {user_id}")
            
            # Legacy anomaly detection
            anomaly_detector = cls.get_anomaly_detector()
            legacy_features = np.array([[heart_rate, blood_oxygen]])
            prediction = anomaly_detector.predict(legacy_features)
            is_anomaly = prediction[0] == -1
            logger.info(f"[ANOMALY_DETECTION] Result: {'Anomaly detected' if is_anomaly else 'Normal'}")
            
            # Calculate rule-based risk
            rule_start_time = datetime.now()
            rule_risk_score = cls.calculate_risk_score(heart_rate, blood_oxygen, user_context)
            rule_duration = (datetime.now() - rule_start_time).total_seconds() * 1000
            logger.info(f"[RULE_RISK] Score: {rule_risk_score:.2f} (calculated in {rule_duration:.2f}ms)")
            
            # Get ML risk prediction
            ml_start_time = datetime.now()
            from services.health_ml_service import HealthMLService
            ml_risk_score = HealthMLService.predict_risk(user_id, features[:2], user_context)
            ml_duration = (datetime.now() - ml_start_time).total_seconds() * 1000
            logger.info(f"[ML_RISK] Score: {ml_risk_score:.2f} (calculated in {ml_duration:.2f}ms)")
            
            # Blend predictions with more weight to ML as we collect more data
            blend_start_time = datetime.now()
            if len(historical_features) > 0:
                # More data = more trust in ML model
                ml_weight = min(0.7, 0.3 + (len(historical_features) / 100))
                rule_weight = 1.0 - ml_weight
                logger.info(f"[RISK_BLEND] Using ML weight: {ml_weight:.2f}, Rule weight: {rule_weight:.2f} based on {len(historical_features)} historical points")
                risk_score = (ml_risk_score * ml_weight) + (rule_risk_score * rule_weight)
            else:
                # Not enough historical data, rely more on rules
                logger.info(f"[RISK_BLEND] Limited history: Using ML weight: 0.3, Rule weight: 0.7")
                risk_score = (ml_risk_score * 0.3) + (rule_risk_score * 0.7)
            
            blend_duration = (datetime.now() - blend_start_time).total_seconds() * 1000    
            logger.info(f"[RISK_BLEND] Final blended risk score: {risk_score:.2f} (blended in {blend_duration:.2f}ms)")
                    
            # Prepare data for analysis
            health_data = {
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen
            }
            if additional_metrics:
                health_data.update(additional_metrics)
                
            # Apply condition-specific adjustments to analysis    
            if user_context and 'health_conditions' in user_context:
                health_conditions = [c.lower() for c in user_context['health_conditions']]
                health_conditions_text = " ".join(health_conditions)
                
                if any(term in health_conditions_text for term in ['anxiety', 'panic disorder', 'stress disorder']):
                    logger.info(f"[CONDITION_ADJUST] Applying anxiety-adjusted heart rate thresholds for user {user_id}")
                    health_data['condition_note'] = "Anxiety may cause elevated heart rate readings"
                
                if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis']):
                    logger.info(f"[CONDITION_ADJUST] Applying COPD-adjusted blood oxygen thresholds for user {user_id}")
                    health_data['condition_note'] = "COPD may cause lower baseline blood oxygen levels"
                
                if any(term in health_conditions_text for term in ['heart disease', 'hypertension', 'cardiovascular']):
                    logger.info(f"[CONDITION_ADJUST] Adding heart condition context for user {user_id}")
                    health_data['condition_note'] = "Heart condition requires careful monitoring of vital signs"
            
            # Get AI recommendations
            ai_start_time = datetime.now()
            ai_analysis = gemini.generate_health_advice(health_data, user_context)
            ai_duration = (datetime.now() - ai_start_time).total_seconds() * 1000
            logger.info(f"[AI_ANALYSIS] Generated AI recommendations in {ai_duration:.2f}ms")
            
            recommendations = cls.generate_recommendations(risk_score, heart_rate, blood_oxygen, user_context)
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score,
                'ml_risk_score': ml_risk_score,
                'rule_risk_score': rule_risk_score,
                'recommendations': recommendations,
                'ai_analysis': ai_analysis
            }
            
            # Save to database
            metrics_to_save = {}
            if additional_metrics:
                metrics_to_save.update(additional_metrics)
                    
            metrics_to_save['analysis_result'] = {
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score,
                'ml_risk_score': ml_risk_score,
                'rule_risk_score': rule_risk_score,
                'recommendations': recommendations
            }
            
            health_data_id = HealthData.create(
                user_id=user_id,
                heart_rate=heart_rate,
                blood_oxygen=blood_oxygen,
                additional_metrics=metrics_to_save
            )
            
            HealthData.update(health_data_id, {
                'recommendations': recommendations,
                'is_anomaly': bool(is_anomaly),
                'risk_score': risk_score,
                'ml_risk_score': ml_risk_score,
                'rule_risk_score': rule_risk_score,
                'ai_analysis': ai_analysis,
                'health_conditions': user_context.get('health_conditions')
            })
            
            # Update ML model with new data (immediate training)
            try:
                HealthMLService.update_user_model(user_id, features[:2], risk_score, user_context)
                logger.info(f"[MODEL_UPDATE] Model updated for user {user_id} with new health data")
            except Exception as e:
                logger.warning(f"[MODEL_UPDATE] Failed to update user model: {e}")

            result['health_data_id'] = health_data_id
            
            # Log overall timing
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"[ANALYSIS_COMPLETE] Total analysis time: {total_duration:.2f}ms for user {user_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"[ANALYSIS_ERROR] Error analyzing health data: {e}", exc_info=True)
            return {
                'error': str(e),
                'recommendations': ["Unable to analyze health data. Please try again later."]
            }

    @staticmethod
    def calculate_risk_score(heart_rate, blood_oxygen, user_context=None):
        """Calculate health risk score based on metrics and user context"""
        # Normal ranges
        hr_normal_low, hr_normal_high = 60, 100
        bo_normal_low = 95

        condition_specific_adjustments = False
        condition_notes = []

        # Log starting parameters
        logging.debug(f"[RULE_RISK_START] HR: {heart_rate}, SpO2: {blood_oxygen}, Initial ranges: HR({hr_normal_low}-{hr_normal_high}), SpO2(≥{bo_normal_low})")

        if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
            health_conditions = [c.lower() for c in user_context['health_conditions']]
            health_conditions_text = " ".join(health_conditions)
            
            # Adjust heart rate range for anxiety
            if any(term in health_conditions_text for term in ['anxiety', 'panic disorder', 'stress disorder']):
                hr_normal_high += 15  # Allow higher heart rate for anxiety patients
                condition_specific_adjustments = True
                condition_notes.append("Adjusted heart rate threshold for anxiety")
                logging.debug(f"[RULE_ADJUST] Anxiety detected, increased HR upper threshold to {hr_normal_high}")
            
            # Adjust blood oxygen threshold for COPD
            if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis']):
                bo_normal_low = 92  # Lower threshold for COPD patients
                condition_specific_adjustments = True
                condition_notes.append("Adjusted blood oxygen threshold for COPD")
                logging.debug(f"[RULE_ADJUST] COPD detected, lowered SpO2 threshold to {bo_normal_low}")
                
            # Athletes might have lower resting heart rates
            if 'athlete' in health_conditions_text:
                hr_normal_low = 50  # Lower threshold for athletes
                condition_specific_adjustments = True
                condition_notes.append("Adjusted heart rate threshold for athletic condition")
                logging.debug(f"[RULE_ADJUST] Athletic condition detected, lowered HR lower threshold to {hr_normal_low}")
            
            # Diabetic patients may need different metrics
            if any(term in health_conditions_text for term in ['diabetes', 'diabetic']):
                condition_specific_adjustments = True
                condition_notes.append("Considering diabetic condition in assessment")
                logging.debug(f"[RULE_ADJUST] Diabetes condition noted in assessment")
            
            # Heart conditions need more careful monitoring
            if any(term in health_conditions_text for term in ['heart disease', 'hypertension', 'arrhythmia', 'cardiovascular']):
                condition_specific_adjustments = True
                condition_notes.append("Heart condition requires more careful monitoring")
                logging.debug(f"[RULE_ADJUST] Heart condition noted, applying more careful monitoring")
        
        # Heart rate risk calculation
        if hr_normal_low <= heart_rate <= hr_normal_high:
            hr_risk = 0
            logging.debug(f"[RULE_HR] Heart rate {heart_rate} is within normal range ({hr_normal_low}-{hr_normal_high}), risk = 0")
        else:
            if heart_rate > hr_normal_high:
                hr_deviation = heart_rate - hr_normal_high
                # More balanced scaling for elevated heart rate
                hr_risk = min(100, (hr_deviation / 20) * 100)
                logging.debug(f"[RULE_HR] Heart rate {heart_rate} exceeds normal range by {hr_deviation}, risk = {hr_risk:.2f}")
            else:
                hr_deviation = hr_normal_low - heart_rate
                hr_risk = min(100, (hr_deviation / 20) * 100)
                logging.debug(f"[RULE_HR] Heart rate {heart_rate} below normal range by {hr_deviation}, risk = {hr_risk:.2f}")
        
        # Blood oxygen risk calculation
        if blood_oxygen >= bo_normal_low:
            bo_risk = 0
            logging.debug(f"[RULE_SPO2] Blood oxygen {blood_oxygen}% is within normal range (≥{bo_normal_low}%), risk = 0")
        else:
            bo_deviation = bo_normal_low - blood_oxygen
            bo_risk = min(100, (bo_deviation / 5) * 100)
            logging.debug(f"[RULE_SPO2] Blood oxygen {blood_oxygen}% below normal by {bo_deviation}%, risk = {bo_risk:.2f}")
        
        # Base risk - equal weighting
        base_risk = (hr_risk * 0.5 + bo_risk * 0.5)
        logging.debug(f"[RULE_BASE] Base risk (50% HR, 50% SpO2): {base_risk:.2f}")
        
        # Combined risk factor when both metrics are abnormal OR there are specific conditions
        if condition_specific_adjustments and hr_risk > 0 and bo_risk > 0:
            prev_risk = base_risk
            base_risk = min(100, base_risk * 1.15)  # Increased multiplier from 1.1 to 1.15
            logging.info(f"[RULE_ADJUST] Applied condition-specific risk adjustment: {', '.join(condition_notes)}")
            logging.debug(f"[RULE_ADJUST] Risk increased from {prev_risk:.2f} to {base_risk:.2f} due to condition adjustments")
        elif hr_risk > 0 and bo_risk > 0:
            prev_risk = base_risk
            base_risk = min(100, base_risk * 1.1)
            logging.debug(f"[RULE_ADJUST] Risk increased from {prev_risk:.2f} to {base_risk:.2f} due to multiple abnormal vitals")
        
        logging.info(f"[RULE_FINAL] Final rule-based risk score: {base_risk:.2f}")
        return base_risk

    @staticmethod
    def generate_fallback_recommendations(risk_score, heart_rate, blood_oxygen, user_context=None):
        """Fallback recommendations if Gemini API fails or for low-risk situations"""
        # Start with basic recommendations based on risk score
        recommendations = []
        
        # Add risk-based recommendations
        if risk_score > 70:
            recommendations.extend([
                "Monitor vital signs closely",
                "Contact your healthcare provider soon",
                "Rest and avoid physical exertion"
            ])
        elif risk_score > 40:
            recommendations.extend([
                "Continue monitoring your vital signs",
                "Consider contacting your healthcare provider if symptoms persist",
                "Take rest and stay hydrated"
            ])
        else:
            recommendations.extend([
                "Vital signs are within normal range",
                "Continue normal activities",
                "Stay hydrated and maintain regular monitoring"
            ])
        
        # Add condition-specific recommendations if user context is available
        if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
            health_conditions = [c.lower() for c in user_context['health_conditions']]
            health_conditions_text = " ".join(health_conditions)
            
            # Anxiety-related recommendations
            if any(term in health_conditions_text for term in ['anxiety', 'panic disorder', 'stress disorder']):
                recommendations.append("Practice deep breathing exercises when feeling anxious")
                recommendations.append("Remember that anxiety can temporarily raise heart rate readings")
                
            # COPD-related recommendations
            if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis']):
                recommendations.append("Use supplemental oxygen as prescribed by your doctor")
                recommendations.append("Avoid exposure to smoke, air pollution, and respiratory irritants")
                
            # Heart condition recommendations
            if any(term in health_conditions_text for term in ['heart disease', 'hypertension', 'arrhythmia']):
                recommendations.append("Take prescribed heart medications consistently")
                recommendations.append("Limit sodium intake and maintain heart-healthy diet")
                
            # Diabetes recommendations
            if any(term in health_conditions_text for term in ['diabetes', 'diabetic']):
                recommendations.append("Monitor blood glucose levels regularly")
                recommendations.append("Maintain consistent meal timing and balanced diet")
                
            # Asthma recommendations
            if any(term in health_conditions_text for term in ['asthma']):
                recommendations.append("Keep rescue inhaler nearby")
                recommendations.append("Monitor and avoid exposure to asthma triggers")
        
        return recommendations[:6]

    @staticmethod
    def generate_recommendations(risk_score, heart_rate, blood_oxygen, user_context=None):
        """Generate personalized health recommendations using Gemini AI"""
        try:
            # Use fallback recommendations for low-risk situations (risk_score <= 40)
            if risk_score <= 40:
                logger.info(f"Using fallback recommendations for low-risk score: {risk_score}")
                return HealthService.generate_fallback_recommendations(risk_score, heart_rate, blood_oxygen, user_context)
                
            # Proceed with Gemini API for higher risk scores
            health_data = {
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen,
                'risk_score': risk_score
            }
            
            # Additional context
            if user_context:
                if 'age' in user_context:
                    health_data['age'] = user_context['age']
                
                if 'health_conditions' in user_context:
                    health_data['health_conditions'] = user_context['health_conditions']
                
                if 'medical_history' in user_context:
                    health_data['medical_history'] = user_context['medical_history']
            
            from gemini_client import gemini
            logger.info(f"Calling Gemini API for high-risk score: {risk_score}")
            recommendations = gemini.generate_health_recommendations(health_data, user_context)
            
            if not recommendations or len(recommendations) == 0:
                logger.warning("Gemini API returned empty recommendations, using fallback")
                return HealthService.generate_fallback_recommendations(risk_score, heart_rate, blood_oxygen, user_context)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return HealthService.generate_fallback_recommendations(risk_score, heart_rate, blood_oxygen, user_context)

    @classmethod
    def get_user_health_history(cls, user_id, limit=10):
        """Get user's health data history"""
        try:
            return HealthData.get_by_user_id(user_id, limit=limit)
        except Exception as e:
            logger.error(f"Error getting user health history: {e}")
            return []
    
    @classmethod
    def get_health_trends(cls, user_id, days=30):
        """Analyze health trends over time"""
        try:
            # Get recent health data
            collection = HealthData.get_collection()
            start_date = datetime.now() - timedelta(days=days)
            
            cursor = collection.find({
                'user_id': user_id,
                'created_at': {'$gte': start_date}
            }).sort('created_at', 1)
            
            data_points = list(cursor)
            
            if not data_points:
                return {
                    'error': 'Not enough data for analysis',
                    'message': f'No health data available for the past {days} days'
                }
            
            # Extract metrics
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
            
            # Calculate trends
            if len(heart_rates) >= 3:
                x = list(range(len(heart_rates)))
                hr_coef = np.polyfit(x, heart_rates, 1)[0]
                hr_trend = "increasing" if hr_coef > 0.5 else "decreasing" if hr_coef < -0.5 else "stable"
                
                bo_coef = np.polyfit(x, blood_oxygen, 1)[0]
                bo_trend = "increasing" if bo_coef > 0.05 else "decreasing" if bo_coef < -0.05 else "stable"
            else:
                hr_trend = "stable"
                bo_trend = "stable"
                
            hr_stats['trend'] = hr_trend
            bo_stats['trend'] = bo_trend
            
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