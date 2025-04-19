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
            
            # Extract features
            from services.feature_engineering import FeatureEngineering
            features = FeatureEngineering.extract_features(
                heart_rate, blood_oxygen, additional_metrics, user_context
            )
            
            # Get anomaly detection result (legacy model)
            anomaly_detector = cls.get_anomaly_detector()
            legacy_features = np.array([[heart_rate, blood_oxygen]])
            anomaly_prediction = anomaly_detector.predict(legacy_features)
            is_anomaly = anomaly_prediction[0] == -1
            
            # Get risk class prediction from ML model
            from services.classification_model import ClassificationModel
            prediction_result = ClassificationModel.predict_risk_class(
                user_id, features[:2], user_context
            )
            
            # Extract risk class and probabilities
            risk_class = prediction_result['risk_class']
            risk_category = prediction_result['risk_category']
            probabilities = prediction_result['probabilities']
            
            # Get recommendations based on risk class
            from services.risk_classification import RiskClassification
            recommendations = RiskClassification.get_recommendations_by_class(
                risk_class, heart_rate, blood_oxygen, user_context
            )
            
            # Calculate legacy risk score for backward compatibility
            rule_risk_score = cls.calculate_risk_score(heart_rate, blood_oxygen, user_context)
            
            # Prepare health data for AI analysis
            health_data = {
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen,
                'risk_class': risk_class,
                'risk_category': risk_category
            }
            if additional_metrics and isinstance(additional_metrics, dict):
                for key, value in additional_metrics.items():
                    health_data[key] = value
            
            # Get AI-generated analysis
            ai_analysis = gemini.generate_health_advice(health_data, user_context)
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': bool(is_anomaly),
                'risk_class': risk_class,
                'risk_category': risk_category,
                'risk_probabilities': probabilities,
                'legacy_risk_score': rule_risk_score,
                'recommendations': recommendations,
                'ai_analysis': ai_analysis
            }
            
            # Prepare additional metrics for saving to database
            metrics_to_save = {}
            if additional_metrics:
                metrics_to_save.update(additional_metrics)
            
            # Add analysis results to metrics
            metrics_to_save['analysis_result'] = {
                'is_anomaly': bool(is_anomaly),
                'risk_class': risk_class,
                'risk_category': risk_category,
                'risk_probabilities': probabilities,
                'legacy_risk_score': rule_risk_score
            }
            
            # Save data to database
            health_data_id = HealthData.create(
                user_id=user_id,
                heart_rate=heart_rate,
                blood_oxygen=blood_oxygen,
                additional_metrics=metrics_to_save
            )
            
            # Update document with recommendations and analysis
            HealthData.update(health_data_id, {
                'recommendations': recommendations,
                'is_anomaly': bool(is_anomaly),
                'risk_class': risk_class,
                'risk_category': risk_category,
                'risk_probabilities': probabilities,
                'legacy_risk_score': rule_risk_score,
                'ai_analysis': ai_analysis
            })
            
            # Update classification model with new data (immediate training)
            try:
                ClassificationModel.update_user_model(user_id, features[:2], risk_class, user_context)
                logger.info(f"[MODEL_UPDATE] Classification model updated for user {user_id}")
            except Exception as e:
                logger.warning(f"[MODEL_UPDATE] Failed to update classification model: {e}")
            
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
        """
        Calculate risk score based on health metrics and user context
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            user_context (dict, optional): User context information
            
        Returns:
            float: Risk score (0-100)
        """
        # Define normal ranges
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
            if any(term in health_conditions_text for c in ['anxiety', 'panic disorder', 'stress disorder']):
                hr_normal_high += 15  # Allow higher heart rate for anxiety patients
                condition_specific_adjustments = True
                condition_notes.append("Adjusted heart rate threshold for anxiety")
                logging.debug(f"[RULE_ADJUST] Anxiety detected, increased HR upper threshold to {hr_normal_high}")
            
            # Adjust blood oxygen threshold for COPD
            if any(term in health_conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']):
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
            
            # Extract risk classes if available
            risk_classes = [int(dp.get('risk_class', 0)) if 'risk_class' in dp else
                           (int(dp.get('analysis_result', {}).get('risk_class', 0)) 
                            if 'analysis_result' in dp and 'risk_class' in dp['analysis_result'] else 0)
                           for dp in data_points]
            
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
            
            # Risk class distribution
            if risk_classes:
                risk_distribution = {
                    'low_risk': risk_classes.count(0) / len(risk_classes) * 100,
                    'medium_risk': risk_classes.count(1) / len(risk_classes) * 100,
                    'high_risk': risk_classes.count(2) / len(risk_classes) * 100
                }
            else:
                risk_distribution = {'low_risk': 0, 'medium_risk': 0, 'high_risk': 0}
            
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
                'blood_oxygen': bo_stats,
                'risk_distribution': risk_distribution
            }
            
        except Exception as e:
            logger.error(f"Error analyzing health trends: {e}")
            return {
                'error': str(e),
                'message': 'Failed to analyze health trends'
            }

    @classmethod
    def evaluate_classification_model(cls, user_id, test_samples=None):
        """
        Evaluate classification model performance
        
        Args:
            user_id (str): User ID
            test_samples (list, optional): Test samples to use
            
        Returns:
            dict: Evaluation results
        """
        try:
            # Get user context
            user = User.get_by_id(user_id)
            user_context = {}
            if user:
                if 'age' in user:
                    user_context['age'] = user['age']
                if 'health_conditions' in user:
                    user_context['health_conditions'] = user['health_conditions']
            
            # Generate test data if not provided
            if test_samples is None:
                test_samples = []
                # Standard test cases with known risk levels
                test_cases = [
                    (60, 98, 0),   # Normal HR, normal O2 = Low risk
                    (90, 96, 0),   # Upper normal HR, normal O2 = Low risk
                    (105, 94, 1),  # Elevated HR, slightly low O2 = Medium risk
                    (120, 92, 1),  # High HR, low O2 = Medium risk
                    (140, 91, 2),  # Very high HR, low O2 = High risk
                    (55, 90, 2),   # Low HR, very low O2 = High risk
                ]
                
                for hr, bo, expected_class in test_cases:
                    test_samples.append({
                        'heart_rate': hr,
                        'blood_oxygen': bo,
                        'expected_class': expected_class
                    })
                
                # Add random test cases from full range
                np.random.seed(42)
                for _ in range(20):
                    hr = np.random.randint(40, 180)
                    bo = np.random.randint(85, 100)
                    
                    # Calculate expected class using rule-based approach
                    risk_score = cls.calculate_risk_score(hr, bo, user_context)
                    from services.risk_classification import RiskClassification
                    expected_class = RiskClassification.score_to_class(risk_score)
                    
                    test_samples.append({
                        'heart_rate': hr,
                        'blood_oxygen': bo,
                        'expected_class': expected_class
                    })
            
            # Initialize evaluation results
            confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            correct_predictions = 0
            rule_correct = 0
            ml_correct = 0
            
            detailed_results = []
            
            # Test each sample
            for sample in test_samples:
                heart_rate = sample['heart_rate']
                blood_oxygen = sample['blood_oxygen']
                expected_class = sample['expected_class']
                
                # Get ML classification result
                from services.classification_model import ClassificationModel
                prediction = ClassificationModel.predict_risk_class(user_id, [heart_rate, blood_oxygen], user_context)
                
                # Extract results
                predicted_class = prediction['risk_class']
                ml_class = np.argmax(prediction['ml_probabilities'])
                rule_class = np.argmax(prediction['rule_probabilities'])
                
                # Update confusion matrix
                confusion_matrix[expected_class][predicted_class] += 1
                
                # Count correct predictions
                if predicted_class == expected_class:
                    correct_predictions += 1
                if ml_class == expected_class:
                    ml_correct += 1
                if rule_class == expected_class:
                    rule_correct += 1
                
                # Add detailed result
                detailed_results.append({
                    'heart_rate': heart_rate,
                    'blood_oxygen': blood_oxygen,
                    'expected_class': expected_class,
                    'predicted_class': int(predicted_class),
                    'ml_class': int(ml_class),
                    'rule_class': int(rule_class),
                    'probabilities': prediction['probabilities']
                })
            
            # Calculate accuracy
            accuracy = correct_predictions / len(test_samples) if test_samples else 0
            ml_accuracy = ml_correct / len(test_samples) if test_samples else 0
            rule_accuracy = rule_correct / len(test_samples) if test_samples else 0
            
            # Calculate class-specific metrics
            precision = [0, 0, 0]
            recall = [0, 0, 0]
            
            for i in range(3):
                # Precision = TP / (TP + FP)
                predicted_as_i = sum(confusion_matrix[j][i] for j in range(3))
                if predicted_as_i > 0:
                    precision[i] = confusion_matrix[i][i] / predicted_as_i
                
                # Recall = TP / (TP + FN)
                actual_i = sum(confusion_matrix[i][j] for j in range(3))
                if actual_i > 0:
                    recall[i] = confusion_matrix[i][i] / actual_i
            
            # Create response
            return {
                'samples_tested': len(test_samples),
                'accuracy': accuracy,
                'ml_only_accuracy': ml_accuracy,
                'rule_only_accuracy': rule_accuracy,
                'confusion_matrix': confusion_matrix,
                'precision': precision,
                'recall': recall,
                'detailed_results': detailed_results[:10],  # Show only first 10 results
                'hybrid_improvement': (accuracy - max(ml_accuracy, rule_accuracy)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {
                'error': str(e),
                'message': 'Failed to evaluate classification model'
            }