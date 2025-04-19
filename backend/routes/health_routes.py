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
def evaluate_classification_model():
    """Evaluate classification model performance"""
    user_id = request.user_id
    
    # Get optional test size parameter
    test_size = request.args.get('test_size', default=30, type=int)
    
    # Run evaluation
    evaluation_results = HealthService.evaluate_classification_model(user_id)
    
    if 'error' in evaluation_results:
        return jsonify(evaluation_results), 400
    
    return jsonify(evaluation_results), 200

@health_bp.route('/test-classification', methods=['POST'])
@token_required
def test_classification():
    """Test classification model with specific vital signs"""
    data = request.get_json()
    user_id = request.user_id
    
    # Validate required fields
    required_fields = ['heart_rate', 'blood_oxygen']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Extract fields
    heart_rate = float(data.get('heart_rate'))
    blood_oxygen = float(data.get('blood_oxygen'))
    
    # Get user context
    from models.user import User
    user = User.get_by_id(user_id)
    user_context = {}
    if user:
        if 'age' in user:
            user_context['age'] = user['age']
        if 'health_conditions' in user:
            user_context['health_conditions'] = user['health_conditions']
    
    # Get rule-based risk score
    from services.health_service import HealthService
    rule_risk_score = HealthService.calculate_risk_score(heart_rate, blood_oxygen, user_context)
    
    # Get rule-based risk class and probabilities
    from services.risk_classification import RiskClassification
    rule_class = RiskClassification.score_to_class(rule_risk_score)
    rule_probabilities = RiskClassification.score_to_probabilities(rule_risk_score)
    
    # Get ML classification
    from services.classification_model import ClassificationModel
    ml_prediction = ClassificationModel.predict_risk_class(user_id, [heart_rate, blood_oxygen], user_context)
    
    # Prepare result
    result = {
        'heart_rate': heart_rate,
        'blood_oxygen': blood_oxygen,
        'rule_risk_score': rule_risk_score,
        'rule_risk_class': rule_class,
        'rule_risk_category': RiskClassification.RISK_CATEGORY_NAMES[rule_class],
        'rule_probabilities': {
            'low': float(rule_probabilities[0]),
            'medium': float(rule_probabilities[1]),
            'high': float(rule_probabilities[2])
        },
        'ml_prediction': ml_prediction
    }
    
    return jsonify(result), 200