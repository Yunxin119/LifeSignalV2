from flask import Blueprint, request, jsonify
from services.auth_service import AuthService

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Extract fields
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Extract optional fields
    kwargs = {}
    optional_fields = ['full_name', 'age', 'medical_history', 'health_conditions']
    for field in optional_fields:
        if field in data:
            kwargs[field] = data.get(field)
    
    # Call register service
    result = AuthService.register(username, email, password, **kwargs)
    
    if result.get('success'):
        return jsonify(result), 201
    else:
        return jsonify({'error': result.get('message', 'Registration failed')}), 400

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login a user"""
    data = request.get_json()
    
    # Validate required fields
    if not data.get('username_or_email') or not data.get('password'):
        return jsonify({'error': 'Username/email and password are required'}), 400
    
    # Extract fields
    username_or_email = data.get('username_or_email')
    password = data.get('password')
    
    # Call authenticate service
    result = AuthService.authenticate(username_or_email, password)
    
    if result.get('success'):
        return jsonify(result), 200
    else:
        return jsonify({'error': result.get('message', 'Authentication failed')}), 401


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Logout a user"""
    data = request.get_json()
    token = data.get('token')
    AuthService.logout(token)
    return jsonify({'success': True}), 200

@auth_bp.route('/update-health', methods=['POST'])
def update_health_conditions():
    """Update user health conditions"""
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication token is missing'}), 401
    
    token = auth_header.split(' ')[1]
    
    # Validate token
    payload = AuthService.validate_token(token)
    if not payload:
        return jsonify({'error': 'Invalid or expired token'}), 401
    
    # Get user ID from token
    user_id = payload.get('user_id')
    
    # Get health conditions from request
    data = request.get_json()
    health_conditions = data.get('health_conditions')
    
    if not health_conditions or not isinstance(health_conditions, list):
        return jsonify({'error': 'Health conditions must be a list'}), 400
    
    # Update health conditions
    result = AuthService.update_health_conditions(user_id, health_conditions)
    
    if result.get('success'):
        return jsonify(result), 200
    else:
        return jsonify({'error': result.get('message', 'Failed to update health conditions')}), 400

# Middleware function to verify JWT token
def token_required(f):
    """Decorator for routes that require authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        # Validate token
        payload = AuthService.validate_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user_id to request for downstream use
        request.user_id = payload.get('user_id')
        
        return f(*args, **kwargs)
    
    return decorated 