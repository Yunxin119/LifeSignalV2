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
    MODEL_DIR = "user_models"
    
    @classmethod
    def get_or_create_user_model(cls, user_id, user_context=None):
        """Get existing user model or create a new one"""
        # Check if directory exists first
        if not os.path.exists(cls.MODEL_DIR):
            logger.info(f"Creating model directory: {cls.MODEL_DIR}")
            os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        # Check for user-specific model
        model_path = os.path.join(cls.MODEL_DIR, f"user_{user_id}_model.pkl")
        
        # Load existing model if available
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading user model for {user_id}")
                model = joblib.load(model_path)
                return model
            except Exception as e:
                logger.error(f"Error loading model for user {user_id}: {e}")
                # Fall through to create new model
        
        # Create new model
        logger.info(f"Creating new model for user {user_id}")
        model = cls._create_base_model(user_context)
        
        # Save model
        joblib.dump(model, model_path)
        
        return model

    @classmethod
    def _create_base_model(cls, user_context=None):
        """Create personalized base model using user health context"""
        logger.info("Creating new personalized model")
        
        # Extract health conditions if available
        health_conditions = []
        if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
            health_conditions = [c.lower() for c in user_context['health_conditions']]
        
        # Create base model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Generate training data with condition-specific examples
        X_train, y_train = cls._generate_training_data(health_conditions, user_context)
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info(f"Base model trained with {len(X_train)} samples")
        
        return model
    
    @classmethod
    def _generate_training_data(cls, health_conditions, user_context=None):
        """Generate synthetic training data with condition-specific adjustments"""
        from services.health_service import HealthService
        
        # Combine all conditions into a single string for easier checking
        conditions_text = " ".join(health_conditions).lower()
        
        # Generate base samples across the range of heart rates and blood oxygen
        X_samples = []
        y_samples = []
        
        # Generate normal range samples first
        for hr in range(40, 180, 5):  # Heart rate from 40 to 180
            for bo in range(80, 101):  # Blood oxygen from 80 to 100
                features = [hr, bo]
                
                # We'll add condition markers as features
                has_copd = 1.0 if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']) else 0.0
                has_anxiety = 1.0 if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']) else 0.0
                has_heart_issue = 1.0 if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']) else 0.0
                is_athlete = 1.0 if 'athlete' in conditions_text else 0.0
                
                # Complete feature vector with condition indicators
                features.extend([has_copd, has_anxiety, has_heart_issue, is_athlete])
                
                X_samples.append(features)
                
                # Get rule-based risk score using health conditions
                risk_score = HealthService.calculate_risk_score(hr, bo, user_context)
                y_samples.append(risk_score)
        
        # Add extra condition-specific samples to boost learning for special cases
        if len(health_conditions) > 0:
            # COPD-specific samples: need more examples with lowered blood oxygen
            if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']):
                for _ in range(200):
                    hr = np.random.uniform(60, 100)
                    bo = np.random.uniform(88, 94)  # COPD patients have lower baseline
                    
                    features = [hr, bo, 1.0, 0.0, 0.0, 0.0]  # COPD=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(HealthService.calculate_risk_score(hr, bo, user_context))
            
            # Anxiety-specific samples: need more examples with elevated heart rate
            if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']):
                for _ in range(200):
                    hr = np.random.uniform(70, 115)  # Higher baseline heart rate
                    bo = np.random.uniform(95, 100)
                    
                    features = [hr, bo, 0.0, 1.0, 0.0, 0.0]  # Anxiety=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(HealthService.calculate_risk_score(hr, bo, user_context))
            
            # Heart condition samples
            if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']):
                for _ in range(200):
                    # Higher risk for unusual heart rates
                    hr = np.random.choice([
                        np.random.uniform(40, 55),   # Low heart rate
                        np.random.uniform(100, 130)  # High heart rate
                    ])
                    bo = np.random.uniform(93, 100)
                    
                    features = [hr, bo, 0.0, 0.0, 1.0, 0.0]  # Heart issue=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(HealthService.calculate_risk_score(hr, bo, user_context))
            
            # Athlete samples: lower resting heart rate
            if 'athlete' in conditions_text:
                for _ in range(200):
                    hr = np.random.uniform(40, 70)  # Athletes often have lower heart rates
                    bo = np.random.uniform(95, 100)
                    
                    features = [hr, bo, 0.0, 0.0, 0.0, 1.0]  # Athlete=1, others=0
                    
                    X_samples.append(features)
                    y_samples.append(HealthService.calculate_risk_score(hr, bo, user_context))
        
        return np.array(X_samples), np.array(y_samples)
    
    @classmethod
    def update_user_model(cls, user_id, features, risk_score, user_context=None):
        """Update user model with new data point"""
        try:
            # Get current model
            model = cls.get_or_create_user_model(user_id, user_context)
            
            # Extract health conditions for feature expansion
            health_conditions_features = [0.0, 0.0, 0.0, 0.0]  # Default: no conditions
            
            if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
                conditions_text = " ".join([c.lower() for c in user_context['health_conditions']])
                
                # Set condition indicators
                has_copd = 1.0 if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']) else 0.0
                has_anxiety = 1.0 if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']) else 0.0
                has_heart_issue = 1.0 if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']) else 0.0
                is_athlete = 1.0 if 'athlete' in conditions_text else 0.0
                
                health_conditions_features = [has_copd, has_anxiety, has_heart_issue, is_athlete]
            
            # Prepare expanded features with condition indicators
            expanded_features = features[:2] + health_conditions_features
            
            # Prepare training data
            X = np.array([expanded_features])
            y = np.array([risk_score])
            
            # Update model
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
        """Predict risk score using user's model with health condition consideration"""
        start_time = datetime.now()
        try:
            # Get or create user model
            model = cls.get_or_create_user_model(user_id, user_context)
            
            # Extract base features
            heart_rate, blood_oxygen = features[:2] if len(features) >= 2 else (0, 0)
            
            # Process health conditions for expanded features
            health_conditions_features = [0.0, 0.0, 0.0, 0.0]  # Default: no conditions
            
            if user_context and 'health_conditions' in user_context and user_context['health_conditions']:
                conditions_text = " ".join([c.lower() for c in user_context['health_conditions']])
                
                # Set condition indicators
                has_copd = 1.0 if any(c in conditions_text for c in ['copd', 'emphysema', 'chronic bronchitis']) else 0.0
                has_anxiety = 1.0 if any(c in conditions_text for c in ['anxiety', 'panic', 'stress']) else 0.0
                has_heart_issue = 1.0 if any(c in conditions_text for c in ['heart', 'cardiac', 'arrhythmia']) else 0.0
                is_athlete = 1.0 if 'athlete' in conditions_text else 0.0
                
                health_conditions_features = [has_copd, has_anxiety, has_heart_issue, is_athlete]
                
                logger.info(f"Predicting with condition features: COPD={has_copd}, Anxiety={has_anxiety}, Heart={has_heart_issue}, Athlete={is_athlete}")
            
            # Prepare expanded features
            expanded_features = [heart_rate, blood_oxygen] + health_conditions_features
            X = np.array([expanded_features])
            
            # Get ML prediction
            prediction = model.predict(X)[0]
            
            # Record processing time
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"ML prediction: {prediction:.2f} (completed in {duration:.2f}ms)")
            
            return float(prediction)
        except Exception as e:
            logger.error(f"Error in ML prediction for user {user_id}: {e}", exc_info=True)
            
            # Fallback to rule-based approach
            from services.health_service import HealthService
            fallback_result = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
            logger.info(f"Using rule-based fallback: {fallback_result}")
            return fallback_result