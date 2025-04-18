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
    def generate_health_timeline(user_profile, days=30, abnormal_prob=0.15, simulation_params=None):
        """Generate a timeline of health data for a user with condition-specific patterns"""
        timeline = []
        
        # Start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Default simulation parameters
        hr_variability_factor = 1.0
        hr_baseline_shift = 0
        bo_variability_factor = 1.0
        bo_baseline_shift = 0
        
        # Special episode patterns
        anxiety_episodes = False
        arrhythmia_episodes = False
        altitude_sensitive = False
        glucose_related_fluctuations = False
        
        # Apply simulation parameters if provided
        if simulation_params:
            hr_variability_factor = simulation_params.get('hr_variability_factor', 1.0)
            hr_baseline_shift = simulation_params.get('hr_baseline_shift', 0)
            bo_variability_factor = simulation_params.get('bo_variability_factor', 1.0)
            bo_baseline_shift = simulation_params.get('bo_baseline_shift', 0)
            
            anxiety_episodes = simulation_params.get('anxiety_episodes', False)
            arrhythmia_episodes = simulation_params.get('arrhythmia_episodes', False)
            altitude_sensitive = simulation_params.get('altitude_sensitive', False)
            glucose_related_fluctuations = simulation_params.get('glucose_related_fluctuations', False)
        
        # Extract user context information
        age = user_profile.get('age', 40)
        conditions = user_profile.get('health_conditions', [])
        conditions_text = " ".join([c.lower() for c in conditions])
        
        # Generate daily records
        current_date = start_date
        day_counter = 0
        
        while current_date <= end_date:
            day_counter += 1
            # Number of readings for this day (random)
            num_readings = random.randint(1, 3)
            
            # Special episode probability increases for certain conditions
            episode_probability = 0.05  # Base probability
            if anxiety_episodes and 'anxiety' in conditions_text:
                episode_probability = 0.2  # Higher chance of anxiety episode
            
            # Determine if this is a special episode day
            is_special_episode = random.random() < episode_probability
            
            for reading in range(num_readings):
                # Determine if this reading is abnormal (standard threshold)
                is_standard_abnormal = random.random() < abnormal_prob
                
                # Base vitals generation
                if is_standard_abnormal:
                    severity = random.choices(
                        ['mild', 'moderate', 'severe'], 
                        weights=[0.6, 0.3, 0.1]
                    )[0]
                    vitals = HealthDataSimulator.generate_abnormal_vitals(user_profile, severity)
                else:
                    vitals = HealthDataSimulator.generate_normal_vitals(user_profile)
                
                # Apply condition-specific adjustments
                heart_rate = vitals['heart_rate']
                blood_oxygen = vitals['blood_oxygen']
                
                # Apply baseline shifts
                heart_rate += hr_baseline_shift
                blood_oxygen += bo_baseline_shift
                
                # Apply variability factors
                if hr_variability_factor != 1.0:
                    # Apply increased variability around the mean
                    hr_deviation = heart_rate - (70 + hr_baseline_shift)  # Deviation from adjusted baseline
                    heart_rate = (70 + hr_baseline_shift) + (hr_deviation * hr_variability_factor)
                
                if bo_variability_factor != 1.0:
                    # Apply increased variability around the mean
                    bo_deviation = blood_oxygen - (97 + bo_baseline_shift)  # Deviation from adjusted baseline
                    blood_oxygen = (97 + bo_baseline_shift) + (bo_deviation * bo_variability_factor)
                
                # Special episode patterns
                if is_special_episode:
                    if anxiety_episodes and 'anxiety' in conditions_text:
                        # Anxiety episode - temporary elevated heart rate
                        heart_rate += random.uniform(15, 30)
                        # Slightly lower oxygen due to faster breathing
                        blood_oxygen -= random.uniform(0, 2)
                    
                    if arrhythmia_episodes and any(c in conditions_text for c in ['heart', 'arrhythmia']):
                        # Arrhythmia episode - irregular heart rate (high or low)
                        if random.random() < 0.5:
                            heart_rate += random.uniform(20, 40)  # Tachycardia
                        else:
                            heart_rate -= random.uniform(15, 25)  # Bradycardia
                    
                    if altitude_sensitive and 'copd' in conditions_text:
                        # COPD patients more affected by environmental factors
                        blood_oxygen -= random.uniform(3, 7)
                    
                    if glucose_related_fluctuations and 'diabetes' in conditions_text:
                        # Blood sugar affecting heart rate
                        if random.random() < 0.5:  # High sugar
                            heart_rate += random.uniform(5, 15)
                        else:  # Low sugar
                            heart_rate -= random.uniform(5, 10)
                
                # Daily patterns (time of day effects)
                hour = random.randint(6, 23)  # Time between 6am and 11pm
                
                if 6 <= hour <= 9:  # Morning
                    heart_rate += random.uniform(0, 5)  # Slightly higher in morning
                elif 22 <= hour <= 23:  # Night
                    heart_rate -= random.uniform(0, 8)  # Lower at night
                    
                # Weekly patterns (more stress on weekdays)
                weekday = (current_date.weekday() < 5)  # Monday-Friday
                if weekday and 'stress' in conditions_text:
                    heart_rate += random.uniform(0, 5)  # Stress effect
                
                # Ensure values are within realistic ranges
                heart_rate = max(40, min(180, heart_rate))
                blood_oxygen = max(75, min(100, blood_oxygen))
                
                # Add timestamp
                timestamp = current_date + timedelta(
                    hours=hour,
                    minutes=random.randint(0, 59)
                )
                
                # Create record
                record = {
                    '_id': ObjectId(),
                    'user_id': 'simulated',
                    'heart_rate': round(heart_rate, 1),
                    'blood_oxygen': round(blood_oxygen, 1),
                    'created_at': timestamp,
                    'updated_at': timestamp,
                    'is_simulated': True,
                    'simulation_note': 'Enhanced simulation' + (' with special episode' if is_special_episode else '')
                }
                
                # Add to timeline
                timeline.append(record)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x['created_at'])
        
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