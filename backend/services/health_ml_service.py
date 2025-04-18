import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import logging
from config import DEBUG
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthMLService:
    """Machine learning service for health risk prediction"""
    
    # Model storage directory
    MODEL_DIR = "models"
    
    @classmethod
    def get_or_create_user_model(cls, user_id, user_context=None):
        """Get existing user model or create a new one"""
        # Check for ensemble model first (for shared risk calculation)
        ensemble_path = os.path.join(cls.MODEL_DIR, "ensemble_model.pkl")
        has_ensemble = os.path.exists(ensemble_path)
        
        # Check for user-specific model
        model_path = os.path.join(cls.MODEL_DIR, f"user_{user_id}_model.pkl")
        
        # Check if directory exists first
        if not os.path.exists(cls.MODEL_DIR):
            logger.info(f"Creating model directory: {cls.MODEL_DIR}")
            os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        # Load existing model if available
        if os.path.exists(model_path):
            try:
                logger.info(f"[MODEL_USAGE] Loading existing user-specific model for user {user_id} from {model_path}")
                model = joblib.load(model_path)
                model_type = type(model).__name__
                logger.info(f"[MODEL_DETAILS] User model type: {model_type}")
                if hasattr(model, 'n_estimators'):
                    logger.info(f"[MODEL_DETAILS] Model estimators: {model.n_estimators}")
                return model
            except Exception as e:
                logger.error(f"[MODEL_ERROR] Error loading model for user {user_id}: {e}")
                # Fall through to create new model
        else:
            logger.info(f"[MODEL_USAGE] No existing model found for user {user_id} at {model_path}")
            # Log ensemble availability
            if has_ensemble:
                logger.info(f"[MODEL_USAGE] Ensemble model is available at {ensemble_path}")
            else:
                logger.info(f"[MODEL_USAGE] No ensemble model found at {ensemble_path}")
        
        # Create new model
        logger.info(f"[MODEL_USAGE] Creating new base model for user {user_id}")
        model = cls._create_base_model(user_context)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"[MODEL_USAGE] Created and saved new model for user {user_id} to {model_path}")
        
        return model

    @classmethod
    def _create_base_model(cls, user_context=None):
        """Create initial model based on user context"""
        # Check for ensemble model first
        ensemble_path = os.path.join(cls.MODEL_DIR, "ensemble_model.pkl")
        default_model_path = os.path.join(cls.MODEL_DIR, "default_model.pkl")
        
        # Try to use ensemble if available
        if os.path.exists(ensemble_path):
            try:
                logger.info(f"[MODEL_USAGE] Loading ensemble model as base from {ensemble_path}")
                ensemble = joblib.load(ensemble_path)
                logger.info(f"[MODEL_DETAILS] Loaded ensemble model with rule_weight: {ensemble.rule_weight}")
                return ensemble
            except Exception as e:
                logger.error(f"[MODEL_ERROR] Could not load ensemble model: {e}")
                # Fall through to regular model creation
        
        # Try to use default model if available
        if os.path.exists(default_model_path):
            try:
                logger.info(f"[MODEL_USAGE] Loading default model from {default_model_path}")
                model = joblib.load(default_model_path)
                return model
            except Exception as e:
                logger.error(f"[MODEL_ERROR] Could not load default model: {e}")
                # Fall through to regular model creation
        
        # Create a new model from scratch if no other options available
        logger.info("[MODEL_USAGE] Creating new GradientBoostingRegressor model from scratch")
        
        # Check if we're creating a model for a user with specific conditions
        has_special_conditions = False
        if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
            health_conditions = [c.lower() for c in user_context['health_conditions']]
            health_conditions_text = " ".join(health_conditions)
            
            # Check for conditions that require special thresholds
            has_special_conditions = any(term in health_conditions_text for term in 
                                     ['copd', 'emphysema', 'chronic bronchitis', 'heart disease', 
                                      'arrhythmia', 'anxiety', 'athlete', 'diabetes'])
            
            if has_special_conditions:
                logger.info(f"[MODEL_TRAINING] Creating model with special consideration for conditions: {health_conditions}")
        
        # For users with special conditions, we'll rely more on rule-based approach rather than ML model
        # So we'll create a basic model that will be overridden by the rule-based calculation in predict_risk
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        X_pretrain = []
        y_pretrain = []
        
        # Import here to avoid circular import
        from services.health_service import HealthService
        
        # Generate samples across the full range of heart rates and blood oxygen
        for hr in range(40, 180, 5):  # Heart rate 40-180
            for bo in range(80, 101):  # Blood oxygen 80-100
                X_pretrain.append([hr, bo])
                
                # Calculate risk score using the rule-based approach with appropriate context
                y_pretrain.append(HealthService.calculate_risk_score(hr, bo, user_context))
        
        # Add more condition-specific samples if needed
        if has_special_conditions:
            logger.info(f"[MODEL_TRAINING] Adding condition-specific training samples")
            
            # Sample more heavily in the regions that are important for this user's conditions
            # This creates a more specialized model for their condition
            for i in range(100):  # Add 100 more condition-specific samples
                # Generate samples focusing on the thresholds for this condition
                if 'copd' in health_conditions_text or 'emphysema' in health_conditions_text:
                    # For COPD, focus on the 92-95% blood oxygen range
                    hr = np.random.uniform(60, 100)
                    bo = np.random.uniform(90, 95)
                elif 'anxiety' in health_conditions_text:
                    # For anxiety, focus on higher heart rates
                    hr = np.random.uniform(90, 120)
                    bo = np.random.uniform(94, 100)
                elif 'athlete' in health_conditions_text:
                    # For athletes, focus on lower heart rates
                    hr = np.random.uniform(45, 70)
                    bo = np.random.uniform(95, 100)
                else:
                    # For other conditions, add general samples
                    hr = np.random.uniform(50, 120)
                    bo = np.random.uniform(90, 100)
                
                X_pretrain.append([hr, bo])
                y_pretrain.append(HealthService.calculate_risk_score(hr, bo, user_context))
        
        # Convert to numpy arrays and fit the model
        X_pretrain = np.array(X_pretrain)
        y_pretrain = np.array(y_pretrain)
        model.fit(X_pretrain, y_pretrain)
        
        logger.info(f"[MODEL_TRAINING] Model trained with {len(X_pretrain)} samples")
        
        return model
    
    @classmethod
    def _generate_age_appropriate_samples(cls, age):
        """Generate synthetic training data based on age"""
        n_samples = 100
        X = []
        y = []
        
        # Adjust normal ranges based on age
        if age < 18:
            hr_range = (70, 120)
            bo_range = (96, 100)
        elif age < 40:
            hr_range = (60, 100)
            bo_range = (95, 100)
        elif age < 65:
            hr_range = (60, 90)
            bo_range = (94, 99)
        else:
            hr_range = (55, 90)
            bo_range = (92, 98)
        
        # Generate normal samples
        for _ in range(n_samples):
            hr = np.random.uniform(hr_range[0], hr_range[1])
            bo = np.random.uniform(bo_range[0], bo_range[1])
            X.append([hr, bo])
            
            # Calculate risk score for normal values
            if hr < hr_range[0] or hr > hr_range[1] or bo < bo_range[0]:
                risk = np.random.uniform(40, 70)  # Moderate risk
            else:
                risk = np.random.uniform(0, 30)  # Low risk
            y.append(risk)
            
        # Add abnormal samples
        for _ in range(n_samples // 5):
            # Generate abnormal heart rate
            hr = np.random.choice([
                np.random.uniform(40, hr_range[0]-1),  # Low heart rate
                np.random.uniform(hr_range[1]+1, 150)  # High heart rate
            ])
            
            # Generate abnormal blood oxygen
            bo = np.random.uniform(85, bo_range[0]-1)
            
            X.append([hr, bo])
            y.append(np.random.uniform(60, 100))  # High risk
        
        return np.array(X), np.array(y)
    
    @classmethod
    def update_user_model(cls, user_id, features, risk_score, user_context=None):
        """Update user model with new data"""
        # Get current model
        model = cls.get_or_create_user_model(user_id, user_context)
        
        # Prepare training data
        X = np.array([features])
        y = np.array([risk_score])
        
        # Update model
        try:
            model.fit(X, y)
            # Save updated model
            model_path = os.path.join(cls.MODEL_DIR, f"user_{user_id}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Updated model for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating model for user {user_id}: {e}")
            return False
    
    @classmethod
    def predict_risk(cls, user_id, features, user_context=None):
        """Predict risk score using user's model"""
        start_time = datetime.now()
        try:
            # Get user model
            model = cls.get_or_create_user_model(user_id, user_context)
            model_type = type(model).__name__
            
            # Log model details
            logger.info(f"[PREDICTION_START] User: {user_id}, Model type: {model_type}")
            
            # Check for ensemble model
            is_ensemble = hasattr(model, 'models') and hasattr(model, 'rule_weight')
            if is_ensemble:
                logger.info(f"[ENSEMBLE_USED] Using ensemble model with rule_weight: {model.rule_weight}")
                
                # For ensembles, we need to handle prediction differently
                heart_rate, blood_oxygen = features[:2] if len(features) >= 2 else (0, 0)
                prediction = model.predict(heart_rate, blood_oxygen, user_context)
                logger.info(f"[PREDICTION_RESULT] Ensemble prediction for user {user_id}: {prediction}")
                
                # Capture processing time
                duration = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(f"[PREDICTION_TIMING] Ensemble prediction completed in {duration:.2f}ms")
                return float(prediction)
            
            # IMPORTANT: Check if we should use the rule-based approach instead due to health conditions
            if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
                health_conditions = [c.lower() for c in user_context['health_conditions']]
                health_conditions_text = " ".join(health_conditions)
                
                # For specific conditions, we should prioritize the rule-based approach
                # since ML models don't properly account for condition-specific thresholds
                if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis', 
                                                                  'heart disease', 'arrhythmia', 'anxiety']):
                    heart_rate, blood_oxygen = features[:2] if len(features) >= 2 else (0, 0)
                    logger.info(f"[PREDICTION_OVERRIDE] Using rule-based calculation for user with conditions: {health_conditions}")
                    from services.health_service import HealthService
                    rule_result = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
                    logger.info(f"[PREDICTION_RESULT] Rule-based result for user with conditions: {rule_result}")
                    return rule_result
            
            # Standard model prediction
            # Ensure features is correctly formatted for prediction
            if isinstance(features, list):
                # Convert list to numpy array
                X = np.array([features[:2]])  # For now, only use heart rate and blood oxygen
                logger.info(f"[PREDICTION_FEATURES] Heart rate: {features[0]}, Blood oxygen: {features[1]}")
            else:
                # Assume it's already a numpy array
                X = features
                logger.info(f"[PREDICTION_FEATURES] Using numpy array features of shape {X.shape}")
            
            prediction = model.predict(X)[0]
            
            # Log prediction details
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                logger.info(f"[MODEL_DETAILS] Feature importances: {importances}")
            
            logger.info(f"[PREDICTION_RESULT] Risk score for user {user_id}: {prediction}")
            
            # Capture processing time
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"[PREDICTION_TIMING] Prediction completed in {duration:.2f}ms")
            
            return float(prediction)
        except Exception as e:
            import services.health_service as HealthService
            logger.error(f"[PREDICTION_ERROR] Error predicting risk for user {user_id}: {e}")
            logger.error(f"[PREDICTION_ERROR] Stack trace: ", exc_info=True)
            
            # Fallback to rule-based approach
            if isinstance(features, list) and len(features) >= 2:
                heart_rate, blood_oxygen = features[:2]
                logger.info(f"[PREDICTION_FALLBACK] Using rule-based fallback with HR: {heart_rate}, SpO2: {blood_oxygen}")
                fallback_result = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
                logger.info(f"[PREDICTION_FALLBACK] Rule-based fallback result: {fallback_result}")
                return fallback_result
            else:
                logger.error(f"[PREDICTION_ERROR] Invalid features format: {features}")
                return 50.0  # Default mid-range risk