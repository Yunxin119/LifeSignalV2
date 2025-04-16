from flask import Flask, jsonify
import logging
import json  # Import standard json module instead
from datetime import datetime
from bson import ObjectId
from flask_cors import CORS

# Import configuration
from config import FLASK_ENV, FLASK_HOST, FLASK_PORT, DEBUG

# Import routes
from routes.auth_routes import auth_bp
from routes.health_routes import health_bp

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder for MongoDB ObjectId and datetime
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Set custom JSON encoder for Flask 2.3.3
    app.json_encoder = CustomJSONEncoder
    
    # Register error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(error=str(e)), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Server error: {e}")
        return jsonify(error="Internal server error"), 500
    
    # Simple health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'ok',
            'environment': FLASK_ENV,
            'version': '1.0.0'
        })
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    logger.info(f"Starting LifeSignal backend in {FLASK_ENV} mode")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG == 1) 