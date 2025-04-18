from flask import Blueprint, request, jsonify
from services.health_service import HealthService
from routes.auth_routes import token_required
import logging

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
        from train.data_simulator import HealthDataSimulator
        test_data = HealthDataSimulator.generate_health_timeline(
            user_context if user else HealthDataSimulator.generate_user_profile(),
            days=10,
            abnormal_prob=0.3  # Increase abnormal ratio for better testing
        )
        
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