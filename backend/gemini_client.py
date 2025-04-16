import google.generativeai as genai
import logging
from config import GEMINI_API_KEY, DEBUG

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiClient, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """Initialize Gemini API client"""
        try:
            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key is not configured")
            
            # Configure the Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Default to Gemini Pro model
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini API client: {e}")
            raise
    
    def generate_health_advice(self, health_data, user_context=None):
        """
        Generate health advice based on health data using Gemini
        
        Args:
            health_data (dict): Health metrics like heart_rate, blood_oxygen, etc.
            user_context (dict, optional): Additional user context like age, medical history, etc.
            
        Returns:
            dict: Generated advice and recommendations
        """
        try:
            # Format data for prompt
            metrics = []
            for key, value in health_data.items():
                metrics.append(f"{key}: {value}")
            
            metrics_str = "\n".join(metrics)
            
            context = ""
            if user_context:
                context_items = []
                for key, value in user_context.items():
                    context_items.append(f"{key}: {value}")
                context = "User context:\n" + "\n".join(context_items)
            
            # Create prompt for Gemini
            prompt = f"""
            Based on the following health metrics, provide an analysis and recommendations:
            
            {metrics_str}
            
            {context}
            
            Analyze these health metrics and provide:
            1. A brief assessment of current health status
            2. Any potential health concerns
            3. Actionable recommendations to improve or maintain health
            4. When the user should seek medical attention, if applicable
            
            Format your response as JSON with the following structure:
            {{
                "status": "normal/moderate/severe",
                "assessment": "brief assessment text",
                "concerns": ["concern 1", "concern 2", ...],
                "recommendations": ["recommendation 1", "recommendation 2", ...],
                "seek_medical_attention": "none/if_symptoms_persist/soon/immediately"
            }}
            """
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            # Try to parse response as JSON, otherwise return the text
            try:
                # Return the text content which should be formatted as JSON
                return {"ai_response": response.text}
            except Exception as parse_err:
                logger.warning(f"Could not parse Gemini response as expected format: {parse_err}")
                return {"ai_response": response.text, "error": "Response format unexpected"}
                
        except Exception as e:
            logger.error(f"Error generating health advice with Gemini: {e}")
            return {"error": str(e)}

# Create a singleton instance
gemini = GeminiClient() 