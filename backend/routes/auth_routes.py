from flask import Blueprint, request, jsonify
from services.auth_service import AuthService

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Define standard condition mapping
STANDARD_CONDITIONS = {
    'none': ['none', 'healthy', 'no condition', 'no medical condition'],
    'hypertension': ['hypertension', 'high blood pressure', 'hbp'],
    'asthma': ['asthma'],
    'COPD': ['copd', 'emphysema', 'chronic bronchitis', 'chronic obstructive pulmonary disease'],
    'heart_disease': ['heart disease', 'heart failure', 'chf', 'coronary artery disease', 'cad'],
    'sleep_apnea': ['sleep apnea', 'osa'],
    'anemia': ['anemia'],
    'diabetes': ['diabetes', 'type 1 diabetes', 'type 2 diabetes']
}

def normalize_condition(condition):
    """Normalize condition string to standard format"""
    if not condition:
        return 'none'
        
    condition_lower = condition.lower().strip()
    
    # Look for match in standard conditions
    for standard, variations in STANDARD_CONDITIONS.items():
        if any(variation in condition_lower for variation in variations):
            return standard
            
    # Default to none if no match found
    return 'none'

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
    optional_fields = ['full_name', 'age', 'medical_history']
    for field in optional_fields:
        if field in data:
            kwargs[field] = data.get(field)
    
    # Handle health profile data if provided
    health_profile = data.get('health_profile')
    if health_profile and isinstance(health_profile, dict):
        # Extract health profile fields
        if 'age' in health_profile:
            kwargs['age'] = health_profile.get('age')
        if 'gender' in health_profile:
            kwargs['gender'] = health_profile.get('gender')
        if 'activity_level' in health_profile:
            activity_level = health_profile.get('activity_level', '').lower()
            
            # Normalize activity level to match our categories
            if activity_level in ['inactive', 'sedentary']:
                kwargs['activity_level'] = 'sedentary'
            elif activity_level in ['light', 'low']:
                kwargs['activity_level'] = 'light'
            elif activity_level in ['moderate', 'medium']:
                kwargs['activity_level'] = 'moderate'
            elif activity_level in ['high', 'active', 'intense', 'vigorous']:
                kwargs['activity_level'] = 'high'
            else:
                kwargs['activity_level'] = 'light'  # Default
                
        if 'medical_history' in health_profile:
            kwargs['medical_history'] = health_profile.get('medical_history')
            
        # Handle pre-existing medical conditions 
        if 'medical_conditions' in health_profile:
            conditions = health_profile.get('medical_conditions')
            
            if isinstance(conditions, list):
                # Normalize all conditions in the list
                normalized_conditions = [normalize_condition(condition) for condition in conditions]
                # Filter out duplicates and 'none' if other conditions exist
                normalized_conditions = list(set(normalized_conditions))
                if len(normalized_conditions) > 1 and 'none' in normalized_conditions:
                    normalized_conditions.remove('none')
                kwargs['medical_conditions'] = normalized_conditions
            elif isinstance(conditions, str) and conditions.strip():
                # Single condition provided as string
                kwargs['medical_conditions'] = [normalize_condition(conditions)]
            else:
                kwargs['medical_conditions'] = ['none']
    
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