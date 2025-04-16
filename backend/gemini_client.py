import google.generativeai as genai
import logging
from config import GEMINI_API_KEY, DEBUG
import json

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
            self.model = genai.GenerativeModel('gemini-2.0-flash')
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
            
            # Extract any trend information
            trend_info = ""
            if 'heart_rate_trend' in health_data or 'blood_oxygen_trend' in health_data:
                trend_info = "Trend information:\n"
                if 'heart_rate_trend' in health_data:
                    trend_info += f"- Heart rate trend: {health_data['heart_rate_trend']}\n"
                if 'blood_oxygen_trend' in health_data:
                    trend_info += f"- Blood oxygen trend: {health_data['blood_oxygen_trend']}\n"
            
            # Create enhanced prompt for Gemini
            prompt = f"""
            Based on the following health metrics, provide a comprehensive analysis and personalized recommendations:
            
            {metrics_str}
            
            {trend_info}
            
            {context}
            
            You are a medical AI assistant tasked with analyzing these health metrics. Please provide:
            
            1. A brief but comprehensive assessment of current health status
            2. A detailed explanation of any potential health concerns based on the metrics
            3. Specific and personalized actionable recommendations considering age, medical history (if available), and current trends
            4. Clear guidance on when the user should seek medical attention
            5. An explanation of what the trends in their data mean (if trend data is available)
            6. Contextual information about normal ranges for these metrics
            
            Format your response as JSON with the following structure:
            {{
                "status": "normal/moderate/severe",
                "assessment": "brief assessment text",
                "explanation": "detailed explanation of the analysis with normal ranges context",
                "concerns": ["concern 1", "concern 2", ...],
                "recommendations": ["recommendation 1", "recommendation 2", ...],
                "trend_analysis": "explanation of what the trends indicate (if applicable)",
                "seek_medical_attention": "none/if_symptoms_persist/soon/immediately",
                "follow_up": "when to take next readings"
            }}
            
            Important: Your analysis should be medically sound but accessible to non-medical professionals. Focus on actionable insights.
            """
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            # Try to parse response as JSON
            try:
                # Check if response is structured as expected
                content = response.text.strip()
                
                # Some Gemini responses might include code block markers
                if content.startswith("```json"):
                    content = content.replace("```json", "", 1)
                if content.endswith("```"):
                    content = content[:-3]
                
                # Clean up and parse the JSON
                content = content.strip()
                parsed_response = json.loads(content)
                
                # Return the parsed response
                return parsed_response
            except Exception as parse_err:
                logger.warning(f"Could not parse Gemini response as JSON: {parse_err}")
                # Return raw text if parsing fails
                return {"ai_response": response.text, "error": "Response format unexpected"}
                
        except Exception as e:
            logger.error(f"Error generating health advice with Gemini: {e}")
            return {"error": str(e)}
    
    def analyze_health_trends(self, trend_data, user_context=None):
        """
        Generate insights from long-term health trends
        
        Args:
            trend_data (dict): Health trend data over time
            user_context (dict, optional): Additional user context
            
        Returns:
            dict: Long-term health insights and recommendations
        """
        try:
            # Convert trend data to string format
            trend_str = json.dumps(trend_data, indent=2)
            
            # Format user context
            context = ""
            if user_context:
                context_items = []
                for key, value in user_context.items():
                    context_items.append(f"{key}: {value}")
                context = "User context:\n" + "\n".join(context_items)
            
            # Create prompt for Gemini
            prompt = f"""
            Based on the following long-term health trend data, provide insights and personalized recommendations:
            
            {trend_str}
            
            {context}
            
            As a medical AI assistant, analyze these health trends and provide:
            
            1. A summary of the long-term trends identified
            2. An assessment of overall health trajectory
            3. Potential health concerns that may be developing
            4. Personalized recommendations based on these trends
            5. Lifestyle adjustments that might be beneficial
            
            Format your response as JSON with the following structure:
            {{
                "trend_summary": "summary of identified trends",
                "health_trajectory": "improving/stable/concerning",
                "assessment": "overall assessment of health patterns",
                "potential_concerns": ["concern 1", "concern 2", ...],
                "recommendations": ["recommendation 1", "recommendation 2", ...],
                "lifestyle_adjustments": ["adjustment 1", "adjustment 2", ...],
                "medical_consultation": "whether medical consultation is advised and why"
            }}
            """
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            # Try to parse response as JSON
            try:
                # Clean up and parse the JSON
                content = response.text.strip()
                
                # Some Gemini responses might include code block markers
                if content.startswith("```json"):
                    content = content.replace("```json", "", 1)
                if content.endswith("```"):
                    content = content[:-3]
                
                # Parse the JSON
                content = content.strip()
                parsed_response = json.loads(content)
                
                return parsed_response
            except Exception as parse_err:
                logger.warning(f"Could not parse Gemini trend analysis response as JSON: {parse_err}")
                return {"ai_response": response.text, "error": "Response format unexpected"}
                
        except Exception as e:
            logger.error(f"Error analyzing health trends with Gemini: {e}")
            return {"error": str(e)}

# Create a singleton instance
gemini = GeminiClient() 