import numpy as np
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class RiskClassification:
    """Service for health risk classification using low, medium, and high risk categories"""
    
    # Risk category definitions
    LOW_RISK = 0
    MEDIUM_RISK = 1
    HIGH_RISK = 2
    
    # Risk category names
    RISK_CATEGORY_NAMES = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Risk score thresholds
    LOW_RISK_THRESHOLD = 30
    MEDIUM_RISK_THRESHOLD = 70
    
    @staticmethod
    def score_to_class(risk_score):
        """
        Convert numerical risk score to risk class
        
        Args:
            risk_score (float): Risk score (0-100)
            
        Returns:
            int: Risk class (0=low, 1=medium, 2=high)
        """
        if risk_score <= RiskClassification.LOW_RISK_THRESHOLD:
            return RiskClassification.LOW_RISK
        elif risk_score <= RiskClassification.MEDIUM_RISK_THRESHOLD:
            return RiskClassification.MEDIUM_RISK
        else:
            return RiskClassification.HIGH_RISK
    
    @staticmethod
    def score_to_probabilities(risk_score):
        """
        Convert risk score to probability distribution across risk classes
        
        Args:
            risk_score (float): Risk score (0-100)
            
        Returns:
            list: Probability distribution [low_prob, medium_prob, high_prob]
        """
        probs = [0, 0, 0]  # [low, medium, high]
        
        if risk_score <= RiskClassification.LOW_RISK_THRESHOLD:
            # Low risk area - lower score means higher probability
            center_dist = RiskClassification.LOW_RISK_THRESHOLD - risk_score
            probs[0] = 0.5 + (center_dist / 60) * 0.5  # 50-100% range
            probs[1] = 1.0 - probs[0]  # Remaining probability to medium risk
        elif risk_score <= RiskClassification.MEDIUM_RISK_THRESHOLD:
            # Medium risk area
            if risk_score < 50:
                # Medium risk closer to low
                center_dist = 50 - risk_score  # Distance to center point
                ratio = center_dist / 20.0  # Normalized factor
                probs[1] = 0.6  # Base probability for medium risk
                probs[0] = 0.4 * ratio  # Low risk probability
                probs[2] = 0.4 * (1 - ratio)  # High risk probability
            else:
                # Medium risk closer to high
                center_dist = risk_score - 50  # Distance to center point
                ratio = center_dist / 20.0  # Normalized factor
                probs[1] = 0.6  # Base probability for medium risk
                probs[2] = 0.4 * ratio  # High risk probability
                probs[0] = 0.4 * (1 - ratio)  # Low risk probability
        else:
            # High risk area - higher score means higher probability
            center_dist = risk_score - RiskClassification.MEDIUM_RISK_THRESHOLD
            probs[2] = 0.5 + (center_dist / 60) * 0.5  # 50-100% range
            probs[1] = 1.0 - probs[2]  # Remaining probability to medium risk
        
        return probs
    
    @staticmethod
    def calculate_risk_probabilities(heart_rate, blood_oxygen, user_context=None):
        """
        Calculate risk probabilities based on health metrics
        
        Args:
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            user_context (dict, optional): User context information
            
        Returns:
            list: Probability distribution [low_prob, medium_prob, high_prob]
        """
        from services.health_service import HealthService
        
        # Calculate risk score using existing method
        risk_score = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
        
        # Convert to probabilities
        return RiskClassification.score_to_probabilities(risk_score)
    
    @staticmethod
    def blend_probabilities(ml_probs, rule_probs, ml_weight=0.5):
        """
        Blend ML and rule-based probabilities
        
        Args:
            ml_probs (list): ML model probability distribution [low, medium, high]
            rule_probs (list): Rule-based probability distribution [low, medium, high]
            ml_weight (float): Weight to give ML probabilities (0.0-1.0)
            
        Returns:
            list: Blended probability distribution [low, medium, high]
        """
        rule_weight = 1.0 - ml_weight
        
        # Blend probabilities
        blended_probs = []
        for i in range(len(ml_probs)):
            blended_probs.append(ml_probs[i] * ml_weight + rule_probs[i] * rule_weight)
        
        # Ensure probabilities sum to 1
        total = sum(blended_probs)
        if total > 0:
            normalized_probs = [p/total for p in blended_probs]
            return normalized_probs
        else:
            # Default to equal probabilities if something went wrong
            return [1/3, 1/3, 1/3]
    
    @staticmethod
    def get_recommendations_by_class(risk_class, heart_rate, blood_oxygen, user_context=None):
        """
        Get recommendations based on risk class
        
        Args:
            risk_class (int): Risk class (0=low, 1=medium, 2=high)
            heart_rate (float): Heart rate measurement
            blood_oxygen (float): Blood oxygen level measurement
            user_context (dict, optional): User context information
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Check for severe conditions requiring immediate attention
        if blood_oxygen < 90 or heart_rate > 150 or heart_rate < 40:
            recommendations.extend([
                "URGENT: Immediate medical attention required",
                "Contact emergency services immediately",
                f"Critical values detected: HR={heart_rate}, SpO2={blood_oxygen}%"
            ])
            return recommendations
        
        # Recommendations based on risk class
        if risk_class == RiskClassification.HIGH_RISK:
            recommendations.extend([
                "Contact your healthcare provider soon",
                "Monitor vital signs closely",
                "Rest and avoid physical exertion"
            ])
        elif risk_class == RiskClassification.MEDIUM_RISK:
            recommendations.extend([
                "Continue monitoring your vital signs",
                "Consider contacting your healthcare provider if symptoms persist",
                "Take rest and stay hydrated"
            ])
        else:  # LOW_RISK
            recommendations.extend([
                "Vital signs are within normal range",
                "Continue normal activities",
                "Stay hydrated and maintain regular monitoring"
            ])
        
        # Add specific heart rate recommendations if needed
        if heart_rate > 100 and heart_rate <= 150 and risk_class != RiskClassification.HIGH_RISK:
            # Specific recommendations for elevated heart rate
            hr_severity = "significantly " if heart_rate > 120 else ""
            recommendations.append(f"Heart rate is {hr_severity}elevated at {int(heart_rate)} BPM")
            recommendations.append("Consider resting and monitoring for other symptoms")
        
        # Add specific blood oxygen recommendations if needed
        if blood_oxygen < 95 and blood_oxygen >= 90 and risk_class != RiskClassification.HIGH_RISK:
            recommendations.append(f"Blood oxygen level ({blood_oxygen}%) is slightly below normal range")
            recommendations.append("Monitor for breathing difficulties or other symptoms")
        
        return recommendations[:6]  