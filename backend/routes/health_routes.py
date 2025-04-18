from flask import Blueprint, request, jsonify
from services.health_service import HealthService
from routes.auth_routes import token_required

# Create health blueprint
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
    
    # Add model information
    result['using_new_model'] = getattr(HealthService, '_using_new_model', False)
    result['model_type'] = 'RandomForestClassifier' if result['using_new_model'] else 'IsolationForest'
    
    # Add explanation of how risk was calculated
    result['risk_calculation'] = {
        'method': 'hybrid',
        'components': {
            'rule_based': 'Condition and age-adjusted threshold analysis',
            'ml_based': result['model_type'] + ' anomaly detection'
        },
        'ml_contribution': result.get('ml_contribution', 'minor')
    }
    
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
        
        # Check if risk_factors are missing (for historical data), and add them
        if 'risk_factors' not in item:
            # Get basic data
            heart_rate = item.get('heart_rate', 0)
            blood_oxygen = item.get('blood_oxygen', 0)
            
            # For historical data, create simplified explanation of risk factors
            item['risk_factors'] = []
            
            if 'risk_score' in item:
                risk_score = item['risk_score']
                if risk_score >= 40:
                    if heart_rate > 100:
                        item['risk_factors'].append(f"Heart rate {heart_rate} is elevated")
                    elif heart_rate < 60:
                        item['risk_factors'].append(f"Heart rate {heart_rate} is below normal range")
                    
                    if blood_oxygen < 95:
                        item['risk_factors'].append(f"Blood oxygen {blood_oxygen}% is below normal levels")
            
            # Add a generic message if we have no specific factors
            if not item['risk_factors']:
                if item.get('risk_score', 0) > 20:
                    item['risk_factors'].append("Vital signs outside normal range")
                else:
                    item['risk_factors'].append("Vital signs within normal range")
        
        # Check if recommendations are missing, and generate them if needed
        if 'recommendations' not in item:
            risk_score = 0
            # Get risk score from analysis_result if available
            if 'analysis_result' in item and 'risk_score' in item['analysis_result']:
                risk_score = item['analysis_result']['risk_score']
            elif 'risk_score' in item:
                risk_score = item['risk_score']
            
            # Generate recommendations based on health data
            recommendations = HealthService.generate_recommendations(
                risk_score=risk_score,
                heart_rate=item.get('heart_rate', 0),
                blood_oxygen=item.get('blood_oxygen', 0)
            )
            
            # Add recommendations to the item
            item['recommendations'] = recommendations
            
        # Add severity level if missing
        if 'severity' not in item and 'risk_score' in item:
            risk_score = item['risk_score']
            if risk_score >= 70:
                item['severity'] = 'severe'
            elif risk_score >= 40:
                item['severity'] = 'moderate'
            elif risk_score >= 20:
                item['severity'] = 'mild'
            else:
                item['severity'] = 'normal'
        
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
        if 'medical_conditions' in user and user['medical_conditions']:
            user_context['medical_conditions'] = user['medical_conditions']
        if 'activity_level' in user:
            user_context['activity_level'] = user['activity_level']
    
    # Call Gemini for analysis
    from gemini_client import gemini
    analysis = gemini.analyze_health_trends(trends, user_context)
    
    response = {
        'trends': trends,
        'ai_analysis': analysis
    }
    
    return jsonify(response), 200

@health_bp.route('/baseline/reset', methods=['POST'])
@token_required
def reset_user_baseline():
    """Reset a user's personalized baseline"""
    # Extract user_id from token (added by token_required decorator)
    user_id = request.user_id
    
    # Call health service to reset baseline
    success = HealthService.reset_user_baseline(user_id)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Your personalized baseline has been reset. New readings will be used to establish your baseline.'
        }), 200
    else:
        return jsonify({
            'error': 'Failed to reset baseline',
            'message': 'There was an error resetting your baseline. Please try again.'
        }), 500
        
@health_bp.route('/baseline/set', methods=['POST'])
@token_required
def set_user_baseline():
    """Manually set a user's baseline values (requires admin or healthcare provider role)"""
    # Extract user_id from token (added by token_required decorator)
    user_id = request.user_id
    
    # Get request data
    data = request.get_json()
    
    # Validate input
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    # Extract parameters
    params = {}
    if 'heart_rate' in data:
        hr_data = data['heart_rate']
        if 'mean' in hr_data:
            params['hr_mean'] = hr_data['mean']
        if 'std' in hr_data:
            params['hr_std'] = hr_data['std']
        if 'min' in hr_data:
            params['hr_min'] = hr_data['min']
        if 'max' in hr_data:
            params['hr_max'] = hr_data['max']
            
    if 'blood_oxygen' in data:
        bo_data = data['blood_oxygen']
        if 'mean' in bo_data:
            params['bo_mean'] = bo_data['mean']
        if 'std' in bo_data:
            params['bo_std'] = bo_data['std']
        if 'min' in bo_data:
            params['bo_min'] = bo_data['min']
        if 'max' in bo_data:
            params['bo_max'] = bo_data['max']
    
    # Call health service to set baseline
    baseline = HealthService.manually_set_user_baseline(user_id, **params)
    
    if baseline:
        return jsonify({
            'success': True,
            'message': 'Your personalized baseline has been updated.',
            'baseline': baseline
        }), 200
    else:
        return jsonify({
            'error': 'Failed to update baseline',
            'message': 'There was an error updating your baseline. Please try again.'
        }), 500
        
@health_bp.route('/baseline', methods=['GET'])
@token_required
def get_user_baseline():
    """Get a user's current baseline values"""
    # Extract user_id from token (added by token_required decorator)
    user_id = request.user_id
    
    # Get user's baseline
    baseline = HealthService.get_user_baseline(user_id)
    
    # Return baseline data
    return jsonify({
        'baseline': baseline,
        'is_personalized': baseline['data_points'] >= 5,
        'is_well_established': baseline['data_points'] >= 20,
        'data_points': baseline['data_points'],
        'manually_set': baseline.get('manually_set', False)
    }), 200

@health_bp.route('/model-info', methods=['GET'])
@token_required
def get_model_info():
    """Get information about the currently active anomaly detection model"""
    # Get information about which model is being used
    using_new_model = getattr(HealthService, '_using_new_model', False)
    
    response = {
        'using_new_model': using_new_model,
        'model_type': 'RandomForestClassifier' if using_new_model else 'IsolationForest',
        'features': {
            'count': len(HealthService._feature_names) if using_new_model and HealthService._feature_names else 2,
            'names': HealthService._feature_names if using_new_model and HealthService._feature_names else ['heart_rate', 'blood_oxygen']
        },
        'conditions_supported': list(HealthService.CONDITION_EFFECTS.keys()) if hasattr(HealthService, 'CONDITION_EFFECTS') else []
    }
    
    return jsonify(response), 200 