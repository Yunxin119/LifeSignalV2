from flask import Blueprint, request, jsonify
from services.health_service import HealthService
from routes.auth_routes import token_required
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a health blueprint
health_bp = Blueprint('health', __name__, url_prefix='/api/health')

@health_bp.route('/analyze', methods=['POST'])
@token_required
def analyze_health_data():
    """Analyze health data"""
    data = request.get_json()
    
    # Extract user_id from token (added by token_required decorator)
    user_id = request.user_id
    
    # Validate required fields
    required_fields = ['heart_rate', 'blood_oxygen']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Extract required fields
    heart_rate = float(data.get('heart_rate'))
    blood_oxygen = float(data.get('blood_oxygen'))
    
    # Extract additional metrics
    additional_metrics = {}
    for key, value in data.items():
        if key not in required_fields and key != 'user_id':
            additional_metrics[key] = value
    
    # Call health service
    result = HealthService.analyze_health_data(
        user_id=user_id,
        heart_rate=heart_rate,
        blood_oxygen=blood_oxygen,
        additional_metrics=additional_metrics if additional_metrics else None
    )
    
    if 'error' in result:
        return jsonify({'error': result.get('error')}), 400
    
    return jsonify(result), 200

@health_bp.route('/history', methods=['GET'])
@token_required
def get_health_history():
    """Get health history for current user"""
    # Extract user_id from token (added by token_required decorator)
    user_id = request.user_id
    
    # Get limit parameter
    limit = request.args.get('limit', default=10, type=int)
    
    # Call health service
    history = HealthService.get_user_health_history(user_id, limit=limit)
    
    # Convert ObjectIds to strings for JSON serialization
    serializable_history = []
    for item in history:
        item['_id'] = str(item['_id'])
        
        # Check if recommendations are missing, and generate them if needed
        if 'recommendations' not in item:
            risk_score = 0
            # Get risk score from analysis_result if available
            if 'analysis_result' in item and 'risk_score' in item['analysis_result']:
                risk_score = item['analysis_result']['risk_score']
            
            # Generate recommendations based on health data
            recommendations = HealthService.generate_recommendations(
                risk_score=risk_score,
                heart_rate=item.get('heart_rate', 0),
                blood_oxygen=item.get('blood_oxygen', 0)
            )
            
            # Add recommendations to the item
            item['recommendations'] = recommendations
        
        serializable_history.append(item)
    
    return jsonify({
        'history': serializable_history,
        'count': len(serializable_history)
    }), 200

@health_bp.route('/trends', methods=['GET'])
@token_required
def get_health_trends():
    """Get health trend analysis for current user"""
    # Extract user_id from token (added by token_required decorator)
    user_id = request.user_id
    
    # Get days parameter
    days = request.args.get('days', default=30, type=int)
    
    # Call health service
    trends = HealthService.get_health_trends(user_id, days=days)
    
    # Check for errors
    if 'error' in trends:
        return jsonify(trends), 400
    
    return jsonify(trends), 200

@health_bp.route('/trends/analyze', methods=['GET'])
@token_required
def analyze_trends_with_ai():
    """Get AI-powered analysis of health trends"""
    user_id = request.user_id
    
    # Get days
    days = request.args.get('days', default=30, type=int)
    
    trends = HealthService.get_health_trends(user_id, days=days)
    
    if 'error' in trends:
        return jsonify(trends), 400
    
    from models.user import User
    user = User.get_by_id(user_id)
    user_context = {}
    if user:
        if 'age' in user:
            user_context['age'] = user['age']
        if 'gender' in user:
            user_context['gender'] = user['gender']
        if 'medical_history' in user:
            user_context['medical_history'] = user['medical_history']
    
    # Call Gemini for analysis
    from gemini_client import gemini
    analysis = gemini.analyze_health_trends(trends, user_context)
    
    response = {
        'trends': trends,
        'ai_analysis': analysis
    }
    
    return jsonify(response), 200 

@health_bp.route('/evaluate-model', methods=['GET'])
@token_required
def evaluate_user_model():
    """Evaluate user's health prediction model with simulated test data"""
    user_id = request.user_id
    
    try:
        # Get user information
        from models.user import User
        user = User.get_by_id(user_id)
        user_context = {}
        if user:
            if 'age' in user:
                user_context['age'] = user['age']
            if 'health_conditions' in user:
                user_context['health_conditions'] = user['health_conditions']
        
        # Generate test data
        test_data = []
        test_cases = [
            (60, 98),   # Normal
            (100, 98),  # Upper normal heart rate
            (120, 98),  # Elevated heart rate
            (60, 92),   # Lower blood oxygen
            (130, 90),  # High heart rate, low blood oxygen
            (40, 95),   # Low heart rate, normal blood oxygen
        ]

        from datetime import datetime
        for hr, bo in test_cases:
            test_data.append({
                'heart_rate': hr,
                'blood_oxygen': bo,
                'created_at': datetime.now()
            })

        # Add random cases from the simulator
        from train.data_simulator import HealthDataSimulator
        simulator_data = HealthDataSimulator.generate_health_timeline(
            user_context if user else HealthDataSimulator.generate_user_profile(),
            days=5,
            abnormal_prob=0.3
        )
        test_data.extend(simulator_data)
        
        # Calculate true risk scores
        from services.health_service import HealthService
        for record in test_data:
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            record['true_risk'] = HealthService.calculate_risk_score(
                heart_rate, blood_oxygen, user_context
            )
        
        # Get ML predictions
        from services.health_ml_service import HealthMLService
        from services.feature_engineering import FeatureEngineering

        for record in test_data:
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            
            # Extract all features
            features = FeatureEngineering.extract_features(
                heart_rate,
                blood_oxygen,
                None,
                user_context
            )
            
            # Use all features for prediction
            record['ml_risk'] = HealthMLService.predict_risk(
                user_id, features, user_context  # Use all features, not just features[:2]
            )
            
            # Calculate hybrid risk score
            record['hybrid_risk'] = (record['ml_risk'] * 0.7) + (record['true_risk'] * 0.3)
        
        # Calculate evaluation metrics
        import numpy as np
        true_risks = np.array([r['true_risk'] for r in test_data])
        ml_risks = np.array([r['ml_risk'] for r in test_data])
        hybrid_risks = np.array([r['hybrid_risk'] for r in test_data])
        
        # Calculate mean absolute error
        ml_mae = np.mean(np.abs(ml_risks - true_risks))
        hybrid_mae = np.mean(np.abs(hybrid_risks - true_risks))
        
        # Prepare result
        evaluation = {
            'test_points': len(test_data),
            'ml_model_error': float(ml_mae),
            'hybrid_model_error': float(hybrid_mae),
            'improvement': float((1 - (hybrid_mae / ml_mae)) * 100) if ml_mae > 0 else 0,
            'sample_data': [{
                'heart_rate': r['heart_rate'],
                'blood_oxygen': r['blood_oxygen'],
                'true_risk': r['true_risk'],
                'ml_risk': r['ml_risk'],
                'hybrid_risk': r['hybrid_risk']
            } for r in test_data[:5]]  # Only show first 5 samples
        }
        
        return jsonify(evaluation), 200
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to evaluate health risk model'
        }), 500
    
@health_bp.route('/simulate', methods=['POST'])
@token_required
def simulate_specific_user_data():
    """
    Simulate health data for a specific user profile with given conditions.
    This is for demo/testing purposes only.
    """
    data = request.get_json()
    user_id = request.user_id
    
    days = data.get('days', 30)
    abnormal_prob = data.get('abnormal_prob', 0.1)
    readings_per_day = data.get('readings_per_day', 2)
    
    user_profile = {}

    if 'age' in data:
        user_profile['age'] = int(data.get('age'))
    else:
        from models.user import User
        user = User.get_by_id(user_id)
        if user and 'age' in user:
            user_profile['age'] = user['age']
        else:
            user_profile['age'] = 50  # Default age
    
    # Use provided health conditions
    if 'health_conditions' in data:
        user_profile['health_conditions'] = data.get('health_conditions')
    else:
        from models.user import User
        user = User.get_by_id(user_id)
        if user and 'health_conditions' in user:
            user_profile['health_conditions'] = user['health_conditions']
        else:
            user_profile['health_conditions'] = []
    
    # Enhanced simulation parameters
    simulation_params = {}
    condition_specific_adjustments = []
    
    # Check for special conditions that need specific simulation parameters
    if user_profile['health_conditions']:
        health_conditions = [c.lower() for c in user_profile['health_conditions']]
        health_conditions_text = " ".join(health_conditions)
        
        # Anxiety adjustments
        if any(term in health_conditions_text for term in ['anxiety', 'panic disorder', 'stress disorder']):
            simulation_params['hr_variability_factor'] = 1.5  # More heart rate variability
            simulation_params['hr_baseline_shift'] = 10  # Higher baseline heart rate
            simulation_params['anxiety_episodes'] = True  # Generate occasional episodes of high HR
            condition_specific_adjustments.append("Added anxiety-specific heart rate patterns")
        
        # COPD adjustments
        if any(term in health_conditions_text for term in ['copd', 'emphysema', 'chronic bronchitis']):
            simulation_params['bo_variability_factor'] = 1.5  # More blood oxygen variability
            simulation_params['bo_baseline_shift'] = -3  # Lower baseline blood oxygen
            simulation_params['altitude_sensitive'] = True  # More affected by environmental factors
            condition_specific_adjustments.append("Added COPD-specific blood oxygen patterns")
            
        # Athlete adjustments
        if any(term in health_conditions_text for term in ['athlete', 'athletic']):
            simulation_params['hr_baseline_shift'] = -10  # Lower resting heart rate
            simulation_params['recovery_factor'] = 1.5  # Better recovery from exertion
            condition_specific_adjustments.append("Added athletic-specific heart metrics")
            
        # Diabetes adjustments
        if any(term in health_conditions_text for term in ['diabetes', 'diabetic']):
            simulation_params['glucose_related_fluctuations'] = True  # Heart rate affected by glucose
            simulation_params['hr_variability_factor'] = 1.3  # More heart rate variability
            condition_specific_adjustments.append("Added diabetes-specific health variations")
            
        # Heart condition adjustments
        if any(term in health_conditions_text for term in ['heart disease', 'hypertension', 'arrhythmia']):
            simulation_params['arrhythmia_episodes'] = True  # Occasional irregular patterns
            simulation_params['stress_sensitivity'] = 1.5  # More sensitive to stress factors
            condition_specific_adjustments.append("Added heart condition-specific patterns")
    
    # Generate timeline with specific user profile and enhanced parameters
    from train.data_simulator import HealthDataSimulator
    
    # Check if the simulator has the enhanced method available
    if hasattr(HealthDataSimulator, 'generate_enhanced_health_timeline') and simulation_params:
        logger.info(f"Using enhanced simulation with condition-specific parameters: {simulation_params}")
        timeline = HealthDataSimulator.generate_enhanced_health_timeline(
            user_profile,
            days=days,
            abnormal_prob=abnormal_prob,
            simulation_params=simulation_params
        )
    else:
        # Fallback to standard method
        logger.info("Using standard simulation method")
        timeline = HealthDataSimulator.generate_health_timeline(
            user_profile,
            days=days,
            abnormal_prob=abnormal_prob
        )
    
    # Save simulated data to database
    created_records = []
    from models.health_data import HealthData
    
    # Track distribution of values
    hr_values = []
    bo_values = []
    
    for record in timeline:
        # Extract vitals
        heart_rate = record['heart_rate']
        blood_oxygen = record['blood_oxygen']
        
        # Track for statistics
        hr_values.append(heart_rate)
        bo_values.append(blood_oxygen)
        
        try:
            # Calculate risk score and analyze
            result = HealthService.analyze_health_data(
                user_id=user_id,
                heart_rate=heart_rate,
                blood_oxygen=blood_oxygen,
                additional_metrics={'is_simulated': True}
            )
            
            # Add to list of created records
            if 'health_data_id' in result:
                created_records.append({
                    'id': str(result['health_data_id']),
                    'heart_rate': heart_rate,
                    'blood_oxygen': blood_oxygen,
                    'risk_score': result['risk_score'],
                    'is_anomaly': result['is_anomaly'],
                    'timestamp': result['timestamp']
                })
        except Exception as e:
            logger.error(f"Error creating simulated record: {e}")
    
    # Calculate statistics on generated data
    import numpy as np
    data_statistics = {
        'heart_rate': {
            'min': np.min(hr_values),
            'max': np.max(hr_values),
            'mean': np.mean(hr_values),
            'std': np.std(hr_values)
        },
        'blood_oxygen': {
            'min': np.min(bo_values),
            'max': np.max(bo_values),
            'mean': np.mean(bo_values),
            'std': np.std(bo_values)
        }
    }
    
    # Call the model evaluation endpoint to get updated metrics
    try:
        from services.health_ml_service import HealthMLService
        from services.feature_engineering import FeatureEngineering
        
        # Get user information for evaluation
        from models.user import User
        user = User.get_by_id(user_id)
        user_context = {}
        if user:
            if 'age' in user:
                user_context['age'] = user['age']
            if 'health_conditions' in user:
                user_context['health_conditions'] = user['health_conditions']
        
        # Generate test data
        test_data = timeline[:min(20, len(timeline))]  # Use subset of the generated data
        
        # Calculate true risk scores
        for record in test_data:
            heart_rate = record['heart_rate']
            blood_oxygen = record['blood_oxygen']
            record['true_risk'] = HealthService.calculate_risk_score(
                heart_rate, blood_oxygen, user_context
            )
        
        # Get ML predictions
        for record in test_data:
            features = FeatureEngineering.extract_features(
                record['heart_rate'],
                record['blood_oxygen'],
                None,
                user_context
            )
            
            record['ml_risk'] = HealthMLService.predict_risk(
                user_id, features[:2], user_context
            )
            
            # Calculate hybrid risk score
            record['hybrid_risk'] = (record['ml_risk'] * 0.7) + (record['true_risk'] * 0.3)
        
        true_risks = np.array([r['true_risk'] for r in test_data])
        ml_risks = np.array([r['ml_risk'] for r in test_data])
        hybrid_risks = np.array([r['hybrid_risk'] for r in test_data])
        
        ml_mae = np.mean(np.abs(ml_risks - true_risks))
        hybrid_mae = np.mean(np.abs(hybrid_risks - true_risks))
        
        # Calculate additional metrics to understand the differences
        rule_vs_ml_diff = np.mean(np.abs(ml_risks - true_risks))
        
        # Track cases where ML significantly differs from rules
        significant_differences = []
        for i, (true, ml) in enumerate(zip(true_risks, ml_risks)):
            if abs(true - ml) > 15:  # Consider a difference of 15+ points significant
                significant_differences.append({
                    'heart_rate': test_data[i]['heart_rate'],
                    'blood_oxygen': test_data[i]['blood_oxygen'],
                    'true_risk': true,
                    'ml_risk': ml,
                    'difference': ml - true
                })
        
        model_evaluation = {
            'ml_model_error': float(ml_mae),
            'hybrid_model_error': float(hybrid_mae),
            'improvement': float((1 - (hybrid_mae / ml_mae)) * 100) if ml_mae > 0 else 0,
            'rule_vs_ml_difference': float(rule_vs_ml_diff),
            'significant_differences': significant_differences[:5] if significant_differences else []
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        model_evaluation = {"error": str(e)}
    
    return jsonify({
        'message': f'Successfully created {len(created_records)} simulated health records',
        'user_profile': user_profile,
        'condition_specific_adjustments': condition_specific_adjustments,
        'data_statistics': data_statistics,
        'records_created': len(created_records),
        'samples': created_records[:5],
        'model_evaluation': model_evaluation
    }), 200