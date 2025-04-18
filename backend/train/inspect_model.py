#!/usr/bin/env python3
"""
Script to test the health risk ML model
"""
import os
import sys
import numpy as np
import logging

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from services.health_ml_service import HealthMLService
from services.health_service import HealthService
from services.feature_engineering import FeatureEngineering

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the health risk prediction model"""
    # User ID for testing
    test_user_id = "test_user_1"
    
    # Create test user context
    user_context = {
        'age': 50,
        'health_conditions': []
    }
    
    # Test cases with different heart rates and blood oxygen levels
    test_cases = [
        (60, 98, "Normal"),
        (100, 98, "Upper normal heart rate"),
        (120, 98, "Elevated heart rate"),
        (60, 92, "Lower blood oxygen"),
        (130, 90, "High heart rate, low blood oxygen"),
        (40, 95, "Low heart rate, normal blood oxygen"),
        (150, 85, "Very high heart rate, very low blood oxygen")
    ]
    
    # Print header
    print("\nTesting Health Risk Prediction Model")
    print("=" * 80)
    print(f"{'Heart Rate':<12} {'Blood O2':<10} {'Rule Risk':<12} {'ML Risk':<10} {'Diff':<8} {'Description'}")
    print("-" * 80)
    
    # Test each case
    for hr, bo, desc in test_cases:
        # Calculate rule-based risk
        rule_risk = HealthService.calculate_risk_score(hr, bo, user_context)
        
        # Extract features
        features = FeatureEngineering.extract_features(hr, bo, None, user_context)
        
        # Make ML prediction
        ml_risk = HealthMLService.predict_risk(test_user_id, features, user_context)
        
        # Calculate difference
        diff = ml_risk - rule_risk
        
        # Print results
        print(f"{hr:<12} {bo:<10} {rule_risk:<12.2f} {ml_risk:<10.2f} {diff:<8.2f} {desc}")
    
    print("=" * 80)
    print("Done")

if __name__ == "__main__":
    main()