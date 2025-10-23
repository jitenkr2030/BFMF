"""
RTI Assistant Application

A complete application for automating RTI (Right to Information) response generation
and management for government departments.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from bharat_domains.governance.models import BharatGov
from bharat_deploy.inference_server import InferenceServer


class RTIRequest(BaseModel):
    """RTI request model"""
    applicant_name: str
    applicant_address: str
    contact_details: str
    query_text: str
    department: str
    urgency_level: str = "normal"  # normal, urgent, very_urgent
    language: str = "hi"
    attachments: Optional[List[str]] = None


class RTIResponse(BaseModel):
    """RTI response model"""
    response_id: str
    reference_number: str
    response_text: str
    status: str
    estimated_processing_time: int
    department_contact: Dict[str, str]
    fees_required: bool
    fee_amount: Optional[float] = None
    next_steps: List[str]


class RTIStatusUpdate(BaseModel):
    """RTI status update model"""
    response_id: str
    new_status: str
    remarks: Optional[str] = None
    updated_by: str


class RTIAssistantApp:
    """
    Complete RTI assistant application for government departments
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.app = FastAPI(
            title="Bharat RTI Assistant",
            description="AI-powered RTI response generation and management system",
            version="1.0.0"
        )
        
        # Initialize components
        self.model = self._load_model()
        self.inference_server = InferenceServer()
        self.rti_database = {}  # In-memory database (replace with real DB)
        
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
            "model_name": "bharat-gov-v1",
            "supported_languages": ["hi", "en", "bn", "ta", "te"],
            "max_response_length": 1024,
            "standard_processing_time": 30,  # days
            "urgent_processing_time": 15,   # days
            "very_urgent_processing_time": 7,  # days
            "rti_fee": 10.0,  # Standard RTI application fee
            "departments": {
                "education": {
                    "name": "Ministry of Education",
                    "contact": "rti.education@gov.in",
                    "phone": "+91-11-26172483"
                },
                "health": {
                    "name": "Ministry of Health and Family Welfare",
                    "contact": "rti.health@gov.in",
                    "phone": "+91-11-23061357"
                },
                "finance": {
                    "name": "Ministry of Finance",
                    "contact": "rti.finance@gov.in",
                    "phone": "+91-11-23092432"
                }
            },
            "cors_origins": ["*"],
            "host": "0.0.0.0",
            "port": 8001
        }
    
    def _load_model(self) -> BharatGov:
        """Load the governance model"""
        try:
            model = BharatGov.from_pretrained(self.config["model_name"])
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
                "message": "Bharat RTI Assistant API",
                "version": "1.0.0",
                "supported_languages": self.config["supported_languages"]
            }
        
        @self.app.post("/rti/submit", response_model=RTIResponse)
        async def submit_rti_request(request: RTIRequest):
            """Submit RTI request and generate response"""
            try:
                # Validate department
                if request.department not in self.config["departments"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported department: {request.department}"
                    )
                
                # Generate RTI response
                response_data = await self._generate_rti_response(request)
                
                return RTIResponse(**response_data)
                
            except Exception as e:
                logging.error(f"RTI submission error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/rti/upload")
        async def upload_rti_document(
            file: UploadFile = File(...),
            department: str = "education",
            language: str = "hi"
        ):
            """Upload RTI request document"""
            try:
                # Read and process uploaded file
                content = await file.read()
                
                # Extract text from document (simplified)
                extracted_text = self._extract_text_from_document(content, file.filename)
                
                # Generate RTI response
                rti_request = RTIRequest(
                    applicant_name="Uploaded Document",
                    applicant_address="To be provided",
                    contact_details="To be provided",
                    query_text=extracted_text,
                    department=department,
                    language=language
                )
                
                response_data = await self._generate_rti_response(rti_request)
                
                return RTIResponse(**response_data)
                
            except Exception as e:
                logging.error(f"Document upload error: {e}")
                raise HTTPException(status_code=500, detail="Error processing document")
        
        @self.app.get("/rti/status/{response_id}")
        async def get_rti_status(response_id: str):
            """Get RTI request status"""
            try:
                if response_id not in self.rti_database:
                    raise HTTPException(status_code=404, detail="RTI request not found")
                
                return self.rti_database[response_id]
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Status check error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.put("/rti/status", response_model=Dict)
        async def update_rti_status(update: RTIStatusUpdate):
            """Update RTI request status"""
            try:
                if update.response_id not in self.rti_database:
                    raise HTTPException(status_code=404, detail="RTI request not found")
                
                # Update status
                self.rti_database[update.response_id]["status"] = update.new_status
                self.rti_database[update.response_id]["last_updated"] = datetime.now().isoformat()
                if update.remarks:
                    self.rti_database[update.response_id]["remarks"] = update.remarks
                
                return {"message": "Status updated successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Status update error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/rti/departments")
        async def get_departments():
            """Get list of supported departments"""
            return {
                "departments": self.config["departments"],
                "supported_languages": self.config["supported_languages"]
            }
        
        @self.app.get("/rti/stats")
        async def get_rti_statistics():
            """Get RTI processing statistics"""
            total_requests = len(self.rti_database)
            status_counts = {}
            
            for request in self.rti_database.values():
                status = request.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_requests": total_requests,
                "status_distribution": status_counts,
                "average_processing_time": self.config["standard_processing_time"]
            }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def _generate_rti_response(self, request: RTIRequest) -> Dict:
        """Generate RTI response"""
        # Generate unique response ID
        response_id = f"RTI_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.query_text) % 10000}"
        
        # Generate reference number
        ref_number = self._generate_reference_number(request.department)
        
        # Determine processing time based on urgency
        processing_time = self._get_processing_time(request.urgency_level)
        
        # Generate RTI response using model
        try:
            response_text = self.model.generate_rti_response(
                rti_query=request.query_text,
                department=request.department,
                relevant_data="",  # Could be enhanced with actual data retrieval
                max_length=self.config["max_response_length"]
            )
            
            # Store in database
            self.rti_database[response_id] = {
                "response_id": response_id,
                "reference_number": ref_number,
                "applicant_name": request.applicant_name,
                "department": request.department,
                "query_text": request.query_text,
                "response_text": response_text,
                "status": "submitted",
                "submitted_at": datetime.now().isoformat(),
                "estimated_completion": (datetime.now() + timedelta(days=processing_time)).isoformat(),
                "urgency_level": request.urgency_level,
                "language": request.language
            }
            
            # Get department contact info
            dept_contact = self.config["departments"][request.department]
            
            # Determine if fees are required
            fees_required = self._check_fees_required(request.query_text)
            
            # Generate next steps
            next_steps = self._generate_next_steps(request.department, fees_required)
            
            return {
                "response_id": response_id,
                "reference_number": ref_number,
                "response_text": response_text,
                "status": "submitted",
                "estimated_processing_time": processing_time,
                "department_contact": {
                    "name": dept_contact["name"],
                    "email": dept_contact["contact"],
                    "phone": dept_contact["phone"]
                },
                "fees_required": fees_required,
                "fee_amount": self.config["rti_fee"] if fees_required else None,
                "next_steps": next_steps
            }
            
        except Exception as e:
            self.logger.error(f"Error generating RTI response: {e}")
            return {
                "response_id": response_id,
                "reference_number": ref_number,
                "response_text": "We acknowledge receipt of your RTI application. Your request is being processed and you will receive a detailed response within the stipulated time frame.",
                "status": "error",
                "estimated_processing_time": processing_time,
                "department_contact": self.config["departments"][request.department],
                "fees_required": False,
                "fee_amount": None,
                "next_steps": ["Contact department for follow-up"]
            }
    
    def _generate_reference_number(self, department: str) -> str:
        """Generate RTI reference number"""
        dept_code = department[:3].upper()
        date_code = datetime.now().strftime("%Y%m%d")
        sequence = len([r for r in self.rti_database.values() if r.get("department") == department]) + 1
        
        return f"RTI/{dept_code}/{date_code}/{sequence:04d}"
    
    def _get_processing_time(self, urgency_level: str) -> int:
        """Get processing time based on urgency level"""
        time_map = {
            "normal": self.config["standard_processing_time"],
            "urgent": self.config["urgent_processing_time"],
            "very_urgent": self.config["very_urgent_processing_time"]
        }
        return time_map.get(urgency_level, self.config["standard_processing_time"])
    
    def _check_fees_required(self, query_text: str) -> bool:
        """Check if RTI fees are required"""
        # Simple heuristic - most RTI requests require fees
        fee_exempt_keywords = ["below poverty line", "bpl", "मुफ्त", "free"]
        
        if any(keyword in query_text.lower() for keyword in fee_exempt_keywords):
            return False
        
        return True
    
    def _generate_next_steps(self, department: str, fees_required: bool) -> List[str]:
        """Generate next steps for applicant"""
        steps = [
            "Keep your RTI reference number safe for future reference",
            f"Contact {self.config['departments'][department]['name']} for follow-up"
        ]
        
        if fees_required:
            steps.extend([
                f"Pay RTI fee of ₹{self.config['rti_fee']} at designated office",
                "Submit fee payment receipt to the department"
            ])
        
        steps.append("Wait for response within stipulated time period")
        
        return steps
    
    def _extract_text_from_document(self, content: bytes, filename: str) -> str:
        """Extract text from uploaded document"""
        # Simplified text extraction
        # In a real implementation, this would use OCR for PDFs/images
        # and proper text extraction for documents
        
        try:
            # Try to decode as text
            text = content.decode('utf-8')
            return text[:2000]  # Limit to first 2000 characters
        except UnicodeDecodeError:
            # If not text, return placeholder
            return "Document uploaded. Please provide specific query in text format."
    
    def run(self, host: str = None, port: int = None):
        """Run the application"""
        host = host or self.config["host"]
        port = port or self.config["port"]
        
        self.logger.info(f"Starting RTI Assistant on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# CLI entry point
def main():
    """Main function to run the RTI assistant application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bharat RTI Assistant")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run application
    app = RTIAssistantApp(config_path=args.config)
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()