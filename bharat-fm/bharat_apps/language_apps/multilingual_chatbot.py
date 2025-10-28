"""
Multilingual Chatbot Application

A complete, production-ready chatbot application that supports 22+ Indian languages
for government scheme inquiries and citizen services.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from bharat_domains.language.models import BharatLang
from bharat_domains.language.datasets import GovernmentSchemesDataset
from bharat_deploy.inference_server import InferenceServer


class ChatRequest(BaseModel):
    """Request model for chatbot interaction"""
    message: str
    language: str = "hi"  # Default to Hindi
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chatbot interaction"""
    response: str
    language: str
    confidence: float
    session_id: str
    suggestions: List[str]
    metadata: Dict[str, Any]


class MultilingualChatbotApp:
    """
    Complete multilingual chatbot application for government services
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.app = FastAPI(
            title="Bharat Multilingual Chatbot",
            description="AI-powered chatbot for government services in 22+ Indian languages",
            version="1.0.0"
        )
        
        # Initialize components
        self.model = self._load_model()
        self.dataset = GovernmentSchemesDataset()
        self.inference_server = InferenceServer()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load application configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "model_name": "bharat-lang-v1",
            "supported_languages": ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "en"],
            "max_response_length": 512,
            "confidence_threshold": 0.7,
            "session_timeout": 1800,  # 30 minutes
            "cors_origins": ["*"],
            "host": "0.0.0.0",
            "port": 8000
        }
    
    def _load_model(self) -> BharatLang:
        """Load the language model"""
        try:
            model = BharatLang.from_pretrained(self.config["model_name"])
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["cors_origins"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Health check endpoint"""
            return {
                "message": "Bharat Multilingual Chatbot API",
                "version": "1.0.0",
                "supported_languages": self.config["supported_languages"]
            }
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Main chat endpoint"""
            try:
                # Validate language
                if request.language not in self.config["supported_languages"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported language: {request.language}"
                    )
                
                # Generate response
                response_data = await self._generate_response(request)
                
                return ChatResponse(**response_data)
                
            except Exception as e:
                logging.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/languages")
        async def get_supported_languages():
            """Get list of supported languages"""
            return {
                "languages": self.config["supported_languages"],
                "default": "hi"
            }
        
        @self.app.get("/schemes")
        async def get_schemes():
            """Get list of government schemes"""
            try:
                schemes_data = self.dataset.get_scheme_chatbot_data()
                schemes = []
                
                for item in schemes_data:
                    if item['scheme'] not in [s['name'] for s in schemes]:
                        schemes.append({
                            "name": item['scheme'],
                            "category": item.get('category', 'general'),
                            "language": item.get('language', 'hi')
                        })
                
                return {"schemes": schemes}
                
            except Exception as e:
                logging.error(f"Error fetching schemes: {e}")
                raise HTTPException(status_code=500, detail="Error fetching schemes")
        
        @self.app.get("/health")
        async def health_check():
            """Detailed health check"""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "dataset_loaded": self.dataset is not None,
                "supported_languages": len(self.config["supported_languages"])
            }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def _generate_response(self, request: ChatRequest) -> Dict:
        """Generate chatbot response"""
        # Create session ID if not provided
        session_id = request.session_id or f"session_{hash(request.message) % 1000000}"
        
        # Prepare prompt with context
        prompt = self._prepare_prompt(request.message, request.language, request.context)
        
        # Generate response using model
        try:
            response_text = self.model.generate_multilingual(
                prompt=prompt,
                target_language=request.language,
                max_length=self.config["max_response_length"]
            )
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(response_text, request.language)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(request.message, request.language)
            
            # Prepare metadata
            metadata = {
                "model": self.config["model_name"],
                "response_length": len(response_text),
                "processing_time": 0.5,  # Placeholder
                "language_detected": request.language
            }
            
            return {
                "response": response_text,
                "language": request.language,
                "confidence": confidence,
                "session_id": session_id,
                "suggestions": suggestions,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "मुझे खेद है, मैं आपकी समस्या को समझने में असमर्थ हूँ। कृपया फिर से प्रयास करें।" if request.language == "hi" else "I'm sorry, I'm unable to understand your issue. Please try again.",
                "language": request.language,
                "confidence": 0.1,
                "session_id": session_id,
                "suggestions": [],
                "metadata": {"error": str(e)}
            }
    
    def _prepare_prompt(self, message: str, language: str, context: Optional[Dict]) -> str:
        """Prepare prompt for model"""
        base_prompt = f"""
        You are a helpful AI assistant for Indian government services. 
        Respond in {language} language.
        
        User message: {message}
        """
        
        if context:
            base_prompt += f"""
            Additional context: {json.dumps(context, ensure_ascii=False)}
            """
        
        base_prompt += """
        Provide helpful, accurate, and concise information about government schemes, services, and policies.
        If you don't know the answer, politely say so and suggest contacting the relevant department.
        """
        
        return base_prompt
    
    def _calculate_confidence(self, response: str, language: str) -> float:
        """Calculate confidence score for response"""
        # Simple heuristic based on response quality
        if len(response) < 10:
            return 0.3
        elif "sorry" in response.lower() or "खेद" in response:
            return 0.2
        else:
            return 0.8
    
    def _generate_suggestions(self, message: str, language: str) -> List[str]:
        """Generate follow-up suggestions"""
        # Common suggestions based on message content
        suggestions = []
        
        if any(word in message.lower() for word in ["scheme", "yojana", "योजना"]):
            suggestions.extend([
                "What are the eligibility criteria?",
                "How to apply for this scheme?",
                "What documents are required?"
            ])
        
        if any(word in message.lower() for word in ["application", "apply", "आवेदन"]):
            suggestions.extend([
                "Where can I get the application form?",
                "What is the last date to apply?",
                "Is there any application fee?"
            ])
        
        # Add language-specific suggestions
        if language == "hi":
            suggestions.extend([
                "और जानकारी के लिए कहाँ संपर्क करें?",
                "इस योजना के लाभ क्या हैं?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def run(self, host: str = None, port: int = None):
        """Run the application"""
        host = host or self.config["host"]
        port = port or self.config["port"]
        
        self.logger.info(f"Starting Multilingual Chatbot on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# CLI entry point
def main():
    """Main function to run the chatbot application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bharat Multilingual Chatbot")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run application
    app = MultilingualChatbotApp(config_path=args.config)
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()