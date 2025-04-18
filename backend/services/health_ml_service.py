import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import logging
from config import DEBUG

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
        model_path = os.path.join(cls.MODEL_DIR, f"user_{user_id}_model.pkl")
        
        # Load existing model if available
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading model for user {user_id}: {e}")
                # Fall through to create new model
        
        # Create new model
        model = cls._create_base_model(user_context)
        
        # Ensure directory exists
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Created new model for user {user_id}")
        
        return model
    
    @classmethod
    def _create_base_model(cls, user_context=None):
        """Create initial model based on user context"""
        # Create gradient boosting regressor for risk prediction
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Pre-train with synthetic data if context available
        if user_context and 'age' in user_context:
            X_pretrain, y_pretrain = cls._generate_age_appropriate_samples(user_context['age'])
            if len(X_pretrain) > 0:
                model.fit(X_pretrain, y_pretrain)
        
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
        try:
            # Get user model
            model = cls.get_or_create_user_model(user_id, user_context)
            
            # Make prediction
            X = np.array([features])
            prediction = model.predict(X)[0]
            logger.info(f"Predicted risk score for user {user_id}: {prediction}")
            
            return float(prediction)
        except Exception as e:
            logger.error(f"Error predicting risk for user {user_id}: {e}")
            from services.health_service import HealthService
            heart_rate, blood_oxygen = features
            return HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)