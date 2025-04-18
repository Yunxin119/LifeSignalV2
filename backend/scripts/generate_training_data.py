#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse

# Define how different pre-existing conditions affect heart rate and blood oxygen
CONDITION_EFFECTS = {
    'none': {'heart_rate': 0, 'blood_oxygen': 0},
    'hypertension': {'heart_rate': 5, 'blood_oxygen': -0.5},
    'asthma': {'heart_rate': 3, 'blood_oxygen': -2},
    'COPD': {'heart_rate': 6, 'blood_oxygen': -3},
    'heart_disease': {'heart_rate': 8, 'blood_oxygen': -1},
    'sleep_apnea': {'heart_rate': 2, 'blood_oxygen': -2.5},
    'anemia': {'heart_rate': 7, 'blood_oxygen': -2},
    'diabetes': {'heart_rate': 3, 'blood_oxygen': -0.5}
}

# Define how activity levels affect heart rate and blood oxygen
ACTIVITY_EFFECTS = {
    'sedentary': {'heart_rate': -5, 'blood_oxygen': -0.5},
    'light': {'heart_rate': 0, 'blood_oxygen': 0},
    'moderate': {'heart_rate': 10, 'blood_oxygen': 0.5},
    'high': {'heart_rate': 20, 'blood_oxygen': 0.2}
}

def age_effect(age):
    """Calculate effect of age on heart rate and blood oxygen"""
    # Heart rate typically decreases with age
    hr_effect = max(-15, -0.4 * (age - 20))
    
    # Blood oxygen slightly decreases with age
    bo_effect = max(-1.5, -0.02 * (age - 20))
    
    return {'heart_rate': hr_effect, 'blood_oxygen': bo_effect}

def generate_normal_reading(age, gender, condition, activity_level):
    """Generate normal heart rate and blood oxygen readings based on demographics"""
    # Base values for a healthy adult
    base_hr = 72
    base_bo = 97
    
    # Adjust for gender
    if gender == 'female':
        base_hr += 3
    
    # Adjust for age
    age_effects = age_effect(age)
    base_hr += age_effects['heart_rate']
    base_bo += age_effects['blood_oxygen']
    
    # Adjust for condition
    condition_effect = CONDITION_EFFECTS.get(condition, {'heart_rate': 0, 'blood_oxygen': 0})
    base_hr += condition_effect['heart_rate']
    base_bo += condition_effect['blood_oxygen']
    
    # Adjust for activity level
    activity_effect = ACTIVITY_EFFECTS.get(activity_level, {'heart_rate': 0, 'blood_oxygen': 0})
    base_hr += activity_effect['heart_rate']
    base_bo += activity_effect['blood_oxygen']
    
    # Add random variation (more realistic noise)
    # Increased noise for heart rate - some normal readings might appear anomalous
    hr_noise = np.random.normal(0, 7.0)  # Increased standard deviation
    bo_noise = np.random.normal(0, 1.0)  # Increased standard deviation
    
    # Add occasional larger deviations to normal readings (borderline cases)
    if np.random.random() < 0.05:  # 5% chance of a borderline normal reading
        if np.random.random() < 0.5:  # Half of these will have high heart rate
            hr_noise += np.random.uniform(15, 25)
        else:  # Half will have low blood oxygen
            bo_noise -= np.random.uniform(2, 4)
    
    heart_rate = max(40, min(180, round(base_hr + hr_noise)))
    blood_oxygen = max(85, min(100, round(base_bo + bo_noise, 1)))
    
    return heart_rate, blood_oxygen

def generate_anomaly_reading(age, gender, condition, activity_level):
    """Generate anomalous heart rate and blood oxygen readings"""
    # Start with a normal reading as baseline
    base_hr, base_bo = generate_normal_reading(age, gender, condition, activity_level)
    
    # Generate varying severities of anomalies (not all anomalies are extreme)
    anomaly_type = np.random.choice(['mild', 'moderate', 'severe'], p=[0.3, 0.4, 0.3])
    
    # Determine which vital sign(s) to make anomalous
    anomaly_target = np.random.choice(['heart_rate', 'blood_oxygen', 'both'], p=[0.4, 0.4, 0.2])
    
    # Create mild anomalies that are barely outside normal range
    if anomaly_type == 'mild':
        if anomaly_target in ['heart_rate', 'both']:
            hr_shift = np.random.choice([-1, 1]) * np.random.uniform(10, 20)
            base_hr += hr_shift
        if anomaly_target in ['blood_oxygen', 'both']:
            bo_shift = -np.random.uniform(2, 4)  # Only decreases for anomalies
            base_bo += bo_shift
    
    # Create moderate anomalies
    elif anomaly_type == 'moderate':
        if anomaly_target in ['heart_rate', 'both']:
            hr_shift = np.random.choice([-1, 1]) * np.random.uniform(20, 35)
            base_hr += hr_shift
        if anomaly_target in ['blood_oxygen', 'both']:
            bo_shift = -np.random.uniform(4, 7)
            base_bo += bo_shift
    
    # Create severe anomalies
    else:
        if anomaly_target in ['heart_rate', 'both']:
            hr_shift = np.random.choice([-1, 1]) * np.random.uniform(35, 60)
            base_hr += hr_shift
        if anomaly_target in ['blood_oxygen', 'both']:
            bo_shift = -np.random.uniform(7, 12)
            base_bo += bo_shift
    
    # Add random noise to make anomalies less distinct
    hr_noise = np.random.normal(0, 5)
    bo_noise = np.random.normal(0, 0.8)
    
    # Ensure values are within realistic physiological limits
    heart_rate = max(30, min(220, round(base_hr + hr_noise)))
    blood_oxygen = max(70, min(99, round(base_bo + bo_noise, 1)))
    
    return heart_rate, blood_oxygen

def generate_dataset(num_samples=10000, anomaly_ratio=0.1, output_file='training_data.csv'):
    """Generate a synthetic dataset of heart rate and blood oxygen readings"""
    data = []
    
    # Define distributions for demographic variables
    ages = np.random.randint(18, 85, num_samples)
    genders = np.random.choice(['male', 'female'], num_samples)
    conditions = np.random.choice(list(CONDITION_EFFECTS.keys()), num_samples, p=[0.5, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05])
    activity_levels = np.random.choice(list(ACTIVITY_EFFECTS.keys()), num_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Track counts for summary
    normal_count = 0
    anomaly_count = 0
    condition_counts = {k: 0 for k in CONDITION_EFFECTS.keys()}
    activity_counts = {k: 0 for k in ACTIVITY_EFFECTS.keys()}
    
    # Generate data
    for i in range(num_samples):
        is_anomaly = np.random.random() < anomaly_ratio
        
        if is_anomaly:
            heart_rate, blood_oxygen = generate_anomaly_reading(
                ages[i], genders[i], conditions[i], activity_levels[i]
            )
            anomaly_count += 1
        else:
            heart_rate, blood_oxygen = generate_normal_reading(
                ages[i], genders[i], conditions[i], activity_levels[i]
            )
            normal_count += 1
        
        condition_counts[conditions[i]] += 1
        activity_counts[activity_levels[i]] += 1
        
        data.append({
            'heart_rate': heart_rate,
            'blood_oxygen': blood_oxygen,
            'age': ages[i],
            'gender': genders[i],
            'condition': conditions[i],
            'activity_level': activity_levels[i],
            'is_anomaly': int(is_anomaly)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Generated {num_samples} records:")
    print(f"  Normal readings: {normal_count}")
    print(f"  Anomalous readings: {anomaly_count}")
    print("\nCondition distribution:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} ({count/num_samples*100:.1f}%)")
    print("\nActivity level distribution:")
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count} ({count/num_samples*100:.1f}%)")
    
    print(f"\nData saved to {output_file}")
    print("\nSample of generated data:")
    print(df.head())
    
    return df

def generate_validation_data(num_samples=2000, training_data_file='training_data.csv', output_file='validation_data.csv'):
    """Generate validation data with similar distribution to training data"""
    # Load training data to match its distribution
    train_df = pd.read_csv(training_data_file)
    
    # Analyze the distribution of conditions and activity levels in training data
    condition_counts = train_df['condition'].value_counts(normalize=True).to_dict()
    activity_counts = train_df['activity_level'].value_counts(normalize=True).to_dict()
    
    # Generate validation data with similar distribution
    data = []
    
    # Define distributions for demographic variables (slightly different from training)
    ages = np.random.randint(18, 85, num_samples)
    genders = np.random.choice(['male', 'female'], num_samples)
    
    # Use distributions from training data
    conditions = np.random.choice(
        list(condition_counts.keys()), 
        num_samples, 
        p=list(condition_counts.values())
    )
    
    activity_levels = np.random.choice(
        list(activity_counts.keys()), 
        num_samples, 
        p=list(activity_counts.values())
    )
    
    # Generate mixture of normal and anomalous readings (50/50 for good evaluation)
    for i in range(num_samples):
        # 50% anomalies in validation set for better evaluation
        is_anomaly = np.random.choice([0, 1])
        
        if is_anomaly:
            heart_rate, blood_oxygen = generate_anomaly_reading(
                ages[i], genders[i], conditions[i], activity_levels[i]
            )
        else:
            heart_rate, blood_oxygen = generate_normal_reading(
                ages[i], genders[i], conditions[i], activity_levels[i]
            )
        
        data.append({
            'heart_rate': heart_rate,
            'blood_oxygen': blood_oxygen,
            'age': ages[i],
            'gender': genders[i],
            'condition': conditions[i],
            'activity_level': activity_levels[i],
            'is_anomaly': is_anomaly
        })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nGenerated {num_samples} validation samples")
    print(f"Validation data saved to {output_file}")
    
    return df

def generate_time_series_data(num_users=5, days=7, anomaly_freq=0.05, output_file='time_series_data.csv'):
    """Generate time series data for multiple users over multiple days"""
    data = []
    
    # Generate user profiles
    user_profiles = []
    for user_id in range(1, num_users + 1):
        age = np.random.randint(18, 85)
        gender = np.random.choice(['male', 'female'])
        condition = np.random.choice(list(CONDITION_EFFECTS.keys()))
        activity_level = np.random.choice(list(ACTIVITY_EFFECTS.keys()))
        
        user_profiles.append({
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'condition': condition,
            'activity_level': activity_level
        })
    
    # Generate readings for each user over time
    start_date = datetime.now() - timedelta(days=days)
    
    for user in user_profiles:
        # 6-12 readings per day
        daily_readings = np.random.randint(6, 13)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for _ in range(daily_readings):
                # Random time during the day
                hour = np.random.randint(7, 23)
                minute = np.random.randint(0, 60)
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                # Determine if this should be an anomalous reading
                is_anomaly = np.random.random() < anomaly_freq
                
                if is_anomaly:
                    heart_rate, blood_oxygen = generate_anomaly_reading(
                        user['age'], user['gender'], user['condition'], user['activity_level']
                    )
                else:
                    heart_rate, blood_oxygen = generate_normal_reading(
                        user['age'], user['gender'], user['condition'], user['activity_level']
                    )
                
                data.append({
                    'user_id': user['user_id'],
                    'timestamp': timestamp,
                    'heart_rate': heart_rate,
                    'blood_oxygen': blood_oxygen,
                    'age': user['age'],
                    'gender': user['gender'],
                    'condition': user['condition'],
                    'activity_level': user['activity_level'],
                    'is_anomaly': int(is_anomaly)
                })
    
    # Convert to DataFrame and sort by user_id and timestamp
    df = pd.DataFrame(data)
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nGenerated time series data for {num_users} users over {days} days")
    print(f"Time series data saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic health data for training')
    parser.add_argument('--num_samples', type=int, default=10000, 
                        help='Number of samples to generate')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, 
                        help='Ratio of anomalous readings in training data')
    parser.add_argument('--output_dir', type=str, default='../data', 
                        help='Directory to save the generated data')
    parser.add_argument('--validation_samples', type=int, default=2000, 
                        help='Number of validation samples to generate')
    parser.add_argument('--time_series_users', type=int, default=5, 
                        help='Number of users for time series data')
    parser.add_argument('--time_series_days', type=int, default=7, 
                        help='Number of days for time series data')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of the data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data
    training_file = os.path.join(args.output_dir, 'training_data.csv')
    df = generate_dataset(args.num_samples, args.anomaly_ratio, training_file)
    
    # Generate validation data
    validation_file = os.path.join(args.output_dir, 'validation_data.csv')
    generate_validation_data(args.validation_samples, training_file, validation_file)
    
    # Generate time series data
    time_series_file = os.path.join(args.output_dir, 'time_series_data.csv')
    generate_time_series_data(args.time_series_users, args.time_series_days, 0.05, time_series_file)
    
    # Visualize data if requested
    if args.visualize:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df, 
            x='heart_rate', 
            y='blood_oxygen', 
            hue='is_anomaly',
            palette={0: 'green', 1: 'red'},
            alpha=0.7
        )
        plt.title('Heart Rate vs. Blood Oxygen (Green: Normal, Red: Anomaly)')
        plt.xlabel('Heart Rate (bpm)')
        plt.ylabel('Blood Oxygen (%)')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(args.output_dir, 'data_visualization.png'))
        print(f"Data visualization saved to {os.path.join(args.output_dir, 'data_visualization.png')}")
    
    print("\nData generation complete.")

if __name__ == "__main__":
    main() 