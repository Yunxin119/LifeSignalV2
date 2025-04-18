import numpy as np
import random
from datetime import datetime, timedelta
from bson import ObjectId
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class HealthDataSimulator:
    """Simulator for generating realistic health data"""
    
    @staticmethod
    def generate_user_profile(age_range=(18, 80), conditions_prob=0.3):
        """Generate a random user profile with age and health conditions"""
        # Generate random age
        age = random.randint(*age_range)
        
        # Common health conditions
        all_conditions = [
            "Hypertension", "Type 2 Diabetes", "Asthma", "COPD", 
            "Heart Disease", "Arrhythmia", "Anxiety", "Depression",
            "Obesity", "Sleep Apnea", "Hypothyroidism", "Anemia"
        ]
        
        # Select random conditions based on probability
        health_conditions = []
        for condition in all_conditions:
            if random.random() < conditions_prob:
                health_conditions.append(condition)
        
        # Return user profile
        return {
            'age': age,
            'health_conditions': health_conditions
        }
    
    @staticmethod
    def generate_normal_vitals(user_profile):
        """Generate normal vital signs based on user profile"""
        age = user_profile.get('age', 40)
        conditions = user_profile.get('health_conditions', [])
        
        # Adjust normal ranges based on age
        if age < 18:
            hr_base = 75
            hr_var = 20
            bo_base = 97
            bo_var = 2
        elif age < 40:
            hr_base = 70
            hr_var = 15
            bo_base = 97
            bo_var = 2
        elif age < 65:
            hr_base = 65
            hr_var = 15
            bo_base = 96
            bo_var = 3
        else:
            hr_base = 60
            hr_var = 20
            bo_base = 95
            bo_var = 3
        
        # Adjust for conditions
        for condition in conditions:
            condition = condition.lower()
            if any(c in condition for c in ['heart', 'arrhythmia']):
                hr_var += 5
                hr_base += 5 if random.random() < 0.6 else -5
            if any(c in condition for c in ['asthma', 'copd', 'sleep apnea']):
                bo_var += 1
                bo_base -= 1
        
        # Generate vitals with random variation
        heart_rate = max(40, min(150, np.random.normal(hr_base, hr_var/3)))
        blood_oxygen = min(100, max(85, np.random.normal(bo_base, bo_var/3)))
        
        return {
            'heart_rate': round(heart_rate, 1),
            'blood_oxygen': round(blood_oxygen, 1)
        }
    
    @staticmethod
    def generate_abnormal_vitals(user_profile, severity='moderate'):
        """Generate abnormal vital signs based on user profile and severity"""
        age = user_profile.get('age', 40)
        conditions = user_profile.get('health_conditions', [])
        
        # Select which vital to make abnormal
        abnormal_type = random.choice(['heart_rate_high', 'heart_rate_low', 'blood_oxygen_low', 'both'])
        
        # Normal baselines
        hr_base = 70 if age < 65 else 65
        bo_base = 97 if age < 65 else 95
        
        # Adjust based on severity
        if severity == 'mild':
            hr_high = hr_base + random.uniform(20, 30)
            hr_low = hr_base - random.uniform(10, 20)
            bo_low = bo_base - random.uniform(2, 4)
        elif severity == 'moderate':
            hr_high = hr_base + random.uniform(30, 50)
            hr_low = hr_base - random.uniform(20, 30)
            bo_low = bo_base - random.uniform(4, 8)
        else:  # severe
            hr_high = hr_base + random.uniform(50, 80)
            hr_low = hr_base - random.uniform(30, 40)
            bo_low = bo_base - random.uniform(8, 15)
        
        # Generate abnormal vitals
        if abnormal_type == 'heart_rate_high':
            heart_rate = hr_high
            blood_oxygen = random.uniform(95, 100)
        elif abnormal_type == 'heart_rate_low':
            heart_rate = hr_low
            blood_oxygen = random.uniform(95, 100)
        elif abnormal_type == 'blood_oxygen_low':
            heart_rate = random.uniform(hr_base - 10, hr_base + 10)
            blood_oxygen = bo_low
        else:  # both abnormal
            if random.random() < 0.5:
                heart_rate = hr_high
            else:
                heart_rate = hr_low
            blood_oxygen = bo_low
        
        # Ensure values are within realistic ranges
        heart_rate = max(30, min(180, heart_rate))
        blood_oxygen = max(75, min(100, blood_oxygen))
        
        return {
            'heart_rate': round(heart_rate, 1),
            'blood_oxygen': round(blood_oxygen, 1)
        }
    
    @staticmethod
    def generate_health_timeline(user_profile, days=30, abnormal_prob=0.15):
        """Generate a timeline of health data for a user"""
        timeline = []
        
        # Start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate daily records
        current_date = start_date
        while current_date <= end_date:
            # Determine if this reading is abnormal
            is_abnormal = random.random() < abnormal_prob
            
            # Choose severity if abnormal
            if is_abnormal:
                severity = random.choices(
                    ['mild', 'moderate', 'severe'], 
                    weights=[0.6, 0.3, 0.1]
                )[0]
                vitals = HealthDataSimulator.generate_abnormal_vitals(user_profile, severity)
            else:
                vitals = HealthDataSimulator.generate_normal_vitals(user_profile)
            
            # Add timestamp
            timestamp = current_date + timedelta(
                hours=random.randint(8, 22),
                minutes=random.randint(0, 59)
            )
            
            # Create record
            record = {
                '_id': ObjectId(),
                'user_id': 'simulated',
                'heart_rate': vitals['heart_rate'],
                'blood_oxygen': vitals['blood_oxygen'],
                'created_at': timestamp,
                'updated_at': timestamp,
                'is_simulated': True
            }
            
            # Add to timeline
            timeline.append(record)
            
            # Move to next date (possibly multiple readings per day)
            if random.random() < 0.3:  # 30% chance of multiple readings per day
                # Add another reading a few hours later
                timestamp = timestamp + timedelta(hours=random.randint(2, 8))
                
                # Second reading has higher chance of being abnormal if first was abnormal
                second_abnormal_prob = 0.6 if is_abnormal else 0.1
                
                if random.random() < second_abnormal_prob:
                    severity = random.choices(
                        ['mild', 'moderate', 'severe'], 
                        weights=[0.5, 0.3, 0.2]
                    )[0]
                    vitals = HealthDataSimulator.generate_abnormal_vitals(user_profile, severity)
                else:
                    vitals = HealthDataSimulator.generate_normal_vitals(user_profile)
                
                # Create second record
                record = {
                    '_id': ObjectId(),
                    'user_id': 'simulated',
                    'heart_rate': vitals['heart_rate'],
                    'blood_oxygen': vitals['blood_oxygen'],
                    'created_at': timestamp,
                    'updated_at': timestamp,
                    'is_simulated': True
                }
                
                # Add to timeline
                timeline.append(record)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return timeline
    
    @classmethod
    def generate_training_dataset(cls, num_users=10, days_per_user=60):
        """Generate a comprehensive training dataset from multiple users"""
        all_data = []
        
        for i in range(num_users):
            # Create diverse user profiles
            if i < num_users * 0.3:  # 30% elderly
                profile = cls.generate_user_profile(age_range=(65, 85), conditions_prob=0.4)
            elif i < num_users * 0.6:  # 30% middle-aged
                profile = cls.generate_user_profile(age_range=(40, 64), conditions_prob=0.3)
            elif i < num_users * 0.9:  # 30% young adults
                profile = cls.generate_user_profile(age_range=(18, 39), conditions_prob=0.2)
            else:  # 10% teenagers
                profile = cls.generate_user_profile(age_range=(13, 17), conditions_prob=0.1)
            
            # Generate timeline for this user
            user_data = cls.generate_health_timeline(
                profile, 
                days=days_per_user,
                abnormal_prob=0.2 if len(profile['health_conditions']) > 0 else 0.1
            )
            
            # Add user context to each record
            for record in user_data:
                record['user_context'] = profile
            
            # Add to dataset
            all_data.extend(user_data)
        
        return all_data
    
    @classmethod
    def calculate_risk_scores(cls, data):
        """Calculate risk scores for the simulated data"""
        # Import health service
        from services.health_service import HealthService
        
        for record in data:
            # Extract vitals
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            
            # Calculate risk score
            risk_score = HealthService.calculate_risk_score(
                heart_rate, 
                blood_oxygen, 
                record.get('user_context')
            )
            
            # Add risk score to record
            record['risk_score'] = risk_score
            
            # Determine if anomaly based on risk score
            record['is_anomaly'] = risk_score > 50
        
        return data