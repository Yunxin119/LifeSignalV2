from flask import Blueprint, request, jsonify
from services.auth_service import AuthService
from services.pair_service import PairingService
from routes.auth_routes import token_required
from bson.objectid import ObjectId
import logging

logger = logging.getLogger(__name__)

# Create pairing blueprint
pair_bp = Blueprint('pair', __name__, url_prefix='/api/pair')

@pair_bp.route('/generate-code', methods=['POST'])
@token_required
def generate_pairing_code():
    """Generate a pairing code for the user"""
    user_id = request.user_id
    
    try:
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
            
        pairing_code = PairingService.generate_pairing_code(user_id)
        
        if pairing_code is not None:
            return jsonify({
                'success': True,
                'pairing_code': pairing_code,
                'expiration_minutes': PairingService.EXPIRATION_MINUTES
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate pairing code'
            }), 400
            
    except Exception as e:
        logger.error(f"Error generating pairing code: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while generating the pairing code'
        }), 500

@pair_bp.route('/validate', methods=['POST'])
def validate_pairing_code():
    """Validate a pairing code and pair a device"""
    data = request.get_json()
    
    # Check if pairing code is provided
    if not data or 'pairing_code' not in data:
        return jsonify({
            'success': False,
            'error': 'Pairing code is required'
        }), 400
        
    pairing_code = data.get('pairing_code')
    device_type = data.get('device_type', 'apple_watch')
    
    try:
        # Validate the pairing code
        user_info = PairingService.validate_pairing_code(pairing_code, device_type)
        
        if user_info is not None:
            # Generate a token for the paired device
            token = AuthService.generate_token(user_info)  # 生成 token，返回的是字符串
            
            return jsonify({
                'success': True,
                'message': 'Device paired successfully',
                'user': user_info,  # 直接返回 user_info 对象
                'token': token  # 直接返回生成的 token
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired pairing code'
            }), 400

    except Exception as e:
        logger.error(f"Error validating pairing code: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while validating the pairing code'
        }), 500

@pair_bp.route('/unpair', methods=['POST'])
@token_required
def unpair_device():
    """Unpair a device from the user"""
    user_id = request.user_id
    
    data = request.get_json()
    device_type = data.get('device_type', 'apple_watch')
    
    try:
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
            
        # Unpair the device
        success = PairingService.unpair_device(user_id, device_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{device_type.capitalize()} unpaired successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to unpair {device_type}'
            }), 400
            
    except Exception as e:
        logger.error(f"Error unpairing device: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while unpairing the device'
        }), 500

@pair_bp.route('/status', methods=['GET'])
@token_required
def check_pairing_status():
    """Check the pairing status of a device"""
    user_id = request.user_id
    
    device_type = request.args.get('device_type', 'apple_watch')
    
    try:
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
            
        # Check pairing status
        is_paired = PairingService.check_pairing_status(user_id, device_type)
        
        return jsonify({
            'success': True,
            'is_paired': is_paired,
            'device_type': device_type
        }), 200
            
    except Exception as e:
        logger.error(f"Error checking pairing status: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while checking the pairing status'
        }), 500
