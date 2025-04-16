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
        serializable_history.append(item)
    
    return jsonify({
        'history': serializable_history,
        'count': len(serializable_history)
    }), 200 