import os
import sys
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_simulator import HealthDataSimulator
from services.health_service import HealthService
from services.health_ml_service import HealthMLService
from services.feature_engineering import FeatureEngineering
import joblib
import logging


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    num_users = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'simulated_data.json'
    train_model = True if len(sys.argv) > 4 and sys.argv[4].lower() == 'train' else False
    
    logger.info(f"Generating data for {num_users} users over {days} days...")
    
    # Generate data
    data = HealthDataSimulator.generate_training_dataset(num_users=num_users, days_per_user=days)
    
    # Calculate risk scores
    logger.info("Calculating risk scores...")
    data = HealthDataSimulator.calculate_risk_scores(data)
    
    # Prepare serializable data for JSON
    serializable_data = []
    for record in data:
        # Create a copy of the record
        record_copy = {}
        for key, value in record.items():
            if key == '_id':
                record_copy[key] = str(value)
            elif key in ['created_at', 'updated_at']:
                record_copy[key] = value.isoformat()
            else:
                record_copy[key] = value
        
        serializable_data.append(record_copy)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    logger.info(f"Generated {len(data)} records and saved to {output_file}")
    
    # Optionally train a model with the generated data
    if train_model:
        logger.info("Training ML model with simulated data...")
        
        # Prepare features and targets
        X_train = []
        y_train = []
        
        for record in data:
            # Get features
            features = FeatureEngineering.extract_features(
                record['heart_rate'],
                record['blood_oxygen'],
                None,  # No additional metrics
                record.get('user_context')
            )
            
            X_train.append(features[:2])  # Just basic features for now
            y_train.append(record['risk_score'])
        
        # Train default model
        logger.info("Creating and training default model...")
        model = HealthMLService._create_base_model(None)
        model.fit(np.array(X_train), np.array(y_train))
        
        # Create models directory if it doesn't exist
        os.makedirs(HealthMLService.MODEL_DIR, exist_ok=True)
        
        # Save default model
        model_path = os.path.join(HealthMLService.MODEL_DIR, "default_model.pkl")
        joblib.dump(model, model_path)
        
        logger.info(f"Trained and saved default model with {len(X_train)} data points")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()