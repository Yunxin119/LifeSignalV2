# Health Analysis System Documentation

## Overview

The health monitoring and analysis system has been enhanced with advanced capabilities for better health insights and personalized recommendations. The system now includes:

1. **Personalized User Models**:

   - Individual anomaly detection models for each user
   - Models that learn and adapt to each user's normal health patterns
   - Automatic model updates as new health data is recorded

2. **Time Series Analysis**:

   - Trend detection for health metrics (increasing, stable, decreasing)
   - Predictive forecasting of future health readings
   - Statistical analysis of health patterns over time

3. **Enhanced AI Analysis**:
   - More detailed and contextual health assessments using Gemini AI
   - Personalized recommendations based on user context and medical history
   - Long-term trend analysis with actionable insights

## API Endpoints

### Existing Endpoints

- `POST /api/health/analyze`: Analyze current health readings
- `GET /api/health/history`: Get health data history

### New Endpoints

- `GET /api/health/trends`: Get statistical trend analysis for a specified time period
- `GET /api/health/trends/analyze`: Get AI-powered analysis of health trends with recommendations

## Integrating Public Health Datasets

To enhance the health analysis system with public health datasets, follow these guidelines:

### Recommended Datasets

1. **MIMIC-III Clinical Database**

   - Source: [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
   - Contains de-identified health data from ICU patients
   - Useful for: Normal ranges, correlations between vital signs

2. **UK Biobank**

   - Source: [UK Biobank](https://www.ukbiobank.ac.uk/)
   - Large prospective study with detailed health data
   - Useful for: Population reference ranges, long-term health outcomes

3. **NHANES**

   - Source: [CDC NHANES](https://www.cdc.gov/nchs/nhanes/index.htm)
   - National Health and Nutrition Examination Survey
   - Useful for: General population health references, demographic factors

4. **HealthData.gov**
   - Source: [HealthData.gov](https://healthdata.gov/)
   - Various healthcare datasets
   - Useful for: Population-level health statistics

### Integration Process

1. **Data Preprocessing**:

   - Download and preprocess datasets to extract relevant metrics
   - Clean and normalize the data to match your system's format
   - Create reference ranges based on demographic factors

2. **Create Reference Models**:

   ```python
   def create_reference_models(dataset_path, output_dir):
       """Process public dataset and create reference models"""
       # Load dataset
       data = pd.read_csv(dataset_path)

       # Group by demographic factors
       for age_group in age_groups:
           for gender in genders:
               subset = data[(data['age_group'] == age_group) &
                            (data['gender'] == gender)]

               # Calculate reference ranges
               metrics = {
                   'heart_rate': {
                       'mean': subset['heart_rate'].mean(),
                       'std': subset['heart_rate'].std(),
                       'p5': subset['heart_rate'].quantile(0.05),
                       'p95': subset['heart_rate'].quantile(0.95)
                   },
                   'blood_oxygen': {
                       'mean': subset['blood_oxygen'].mean(),
                       'std': subset['blood_oxygen'].std(),
                       'p5': subset['blood_oxygen'].quantile(0.05),
                       'p95': subset['blood_oxygen'].quantile(0.95)
                   }
               }

               # Save reference model
               model_path = os.path.join(
                   output_dir,
                   f"reference_{age_group}_{gender}.json"
               )
               with open(model_path, 'w') as f:
                   json.dump(metrics, f)
   ```

3. **Integration with Analysis System**:

   ```python
   def get_reference_ranges(user_context):
       """Get reference ranges for a specific user"""
       age = user_context.get('age')
       gender = user_context.get('gender')

       # Map user to appropriate age group
       age_group = map_to_age_group(age)

       # Load reference model
       model_path = os.path.join(
           REFERENCE_DIR,
           f"reference_{age_group}_{gender}.json"
       )

       if os.path.exists(model_path):
           with open(model_path, 'r') as f:
               return json.load(f)
       else:
           # Return default reference ranges
           return DEFAULT_REFERENCE_RANGES
   ```

4. **Enhanced Risk Assessment**:

   ```python
   def calculate_risk_with_references(metrics, user_context):
       """Calculate risk score with reference to population data"""
       # Get reference ranges for user
       references = get_reference_ranges(user_context)

       # Calculate z-scores relative to reference population
       z_scores = {}
       for metric, value in metrics.items():
           if metric in references:
               mean = references[metric]['mean']
               std = references[metric]['std']
               z_scores[metric] = (value - mean) / std

       # Calculate risk based on deviation from reference
       risk_score = calculate_risk_from_z_scores(z_scores)

       return risk_score
   ```

### Data Privacy Considerations

When working with health data:

1. Ensure all public datasets are properly de-identified
2. Never combine public data with private user data in a way that could re-identify individuals
3. Follow relevant regulations (HIPAA, GDPR, etc.) for health data processing
4. Clearly document data sources and processing methods

## Implementation Notes

### Machine Learning Models

The system uses several types of models:

1. **IsolationForest** (Base model):

   - General anomaly detection
   - Used when insufficient user data is available

2. **LocalOutlierFactor** (Personalized model):

   - User-specific anomaly detection
   - Adapts to individual normal ranges

3. **ExponentialSmoothing** (Time series):
   - Forecasting future health readings
   - Detecting trends in health data

### Model Storage

User models are stored in the `user_models` directory as serialized files:

- `user_model_{user_id}.joblib`: Personalized model for each user
- Models are automatically updated as new data is received

## Machine Learning System for Health Risk Prediction

### Overview

The system uses a hybrid approach that combines rule-based and machine learning predictions to assess health risks. The importance of each approach is weighted dynamically based on the amount of historical data available for the user.

### Machine Learning Implementation

#### Improved Condition-Aware ML Model (Current)

The current implementation incorporates health conditions directly into the machine learning model:

1. **Feature Engineering**: Health conditions are encoded as additional features in the input vector

   - Base features: Heart rate, blood oxygen
   - Condition indicators: COPD, anxiety, heart issues, athlete status

2. **Training Data Generation**: The system generates training data with emphasis on condition-specific regions

   - For COPD patients: More samples with lower blood oxygen levels (88-94%)
   - For anxiety patients: More samples with elevated heart rates (70-115 bpm)
   - For heart conditions: More samples with very low or high heart rates
   - For athletes: More samples with lower resting heart rates (40-70 bpm)

3. **Model Personalization**: Each user has their own model that is continually updated with new data
   - Model stored in user_models directory
   - Each model is pre-trained with condition-specific data before first use
   - Models are updated with real user data over time

#### Risk Score Calculation

The final risk score is calculated as a weighted average of:

- Machine learning prediction (ml_risk_score)
- Rule-based calculation (rule_risk_score)

The weight given to the ML model increases as more historical data becomes available:

```python
if len(historical_features) > 0:
    ml_weight = min(0.7, 0.3 + (len(historical_features) / 100))
    rule_weight = 1.0 - ml_weight
    risk_score = (ml_risk_score * ml_weight) + (rule_risk_score * rule_weight)
else:
    risk_score = (ml_risk_score * 0.3) + (rule_risk_score * 0.7)
```

This means:

- New users start with 30% ML weight, 70% rule weight
- As data accumulates, ML weight can increase up to 70%

### Resetting Models

To reset all models and start with fresh machine learning models:

```bash
python reset_models.py
```

This script:

1. Backs up all existing models to a timestamped directory
2. Removes old model directories
3. Creates fresh, empty model directories
4. Also cleans up test result directories

### Development Notes

When developing new features or adjusting the ML model:

1. Test with different health condition scenarios
2. Verify that special health conditions are properly accounted for
3. Monitor the model's performance as it learns from new data
4. Check the relative weights of ML vs. rule-based predictions

## Future Improvements

1. **Multivariate Analysis**:

   - Analyze correlations between different health metrics
   - Detect complex health patterns involving multiple variables

2. **Seasonal Pattern Detection**:

   - Identify daily, weekly, or seasonal health patterns
   - Adjust recommendations based on time-specific factors

3. **Federated Learning**:

   - Improve models across users while maintaining privacy
   - Allow opt-in aggregated learning from anonymized data

4. **Advanced Alerting System**:

   - Multi-level alert system based on risk severity
   - Customizable alert thresholds for different users

5. **Integration with Wearable Data**:
   - Support for continuous monitoring from wearable devices
   - Real-time analysis and feedback
