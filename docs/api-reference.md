# API Reference

This document provides comprehensive API reference for the Bharat Foundation Model Framework (BFMF).

## 📋 Table of Contents

- [Core API](#core-api)
- [Domain Modules API](#domain-modules-api)
- [Utilities API](#utilities-api)
- [Configuration API](#configuration-api)
- [Error Handling](#error-handling)

## 🏗️ Core API

### BharatFM Class

The main entry point for BFMF functionality.

```python
class BharatFM:
    """
    Main BFMF class that provides access to all framework functionality.
    """
    
    def __init__(self, modules=None, config=None):
        """
        Initialize BFMF with optional modules and configuration.
        
        Args:
            modules (list): List of modules to initialize
            config (dict): Configuration dictionary
        """
        pass
    
    @property
    def version(self) -> str:
        """Get BFMF version."""
        pass
    
    def chat(self, message: str, language: str = None) -> str:
        """
        Send a chat message and get response.
        
        Args:
            message (str): Input message
            language (str): Language code (optional)
            
        Returns:
            str: Response message
        """
        pass
    
    def process(self, input_data: dict, modules: list = None) -> dict:
        """
        Process input data using specified modules.
        
        Args:
            input_data (dict): Input data
            modules (list): List of modules to use
            
        Returns:
            dict: Processed data
        """
        pass
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        pass
    
    def configure_module(self, module_name: str, config: dict):
        """
        Configure a specific module.
        
        Args:
            module_name (str): Name of the module
            config (dict): Configuration dictionary
        """
        pass
    
    def register_module(self, module):
        """
        Register a custom module.
        
        Args:
            module: Module instance to register
        """
        pass
```

### Usage Example

```python
from bharat_fm import BharatFM

# Initialize BFMF
bfmf = BharatFM()

# Basic chat
response = bfmf.chat("Hello, how are you?")
print(response)

# Process with specific modules
result = bfmf.process(
    input_data={"text": "नमस्ते", "task": "translate"},
    modules=["language_ai"]
)
print(result)

# Get supported languages
languages = bfmf.get_supported_languages()
print(languages)
```

## 🌐 Language AI API

### LanguageAI Class

```python
class LanguageAI:
    """
    Language AI module for multi-language processing.
    """
    
    def __init__(self, config=None):
        """
        Initialize Language AI.
        
        Args:
            config (dict): Configuration dictionary
        """
        pass
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of given text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code
        """
        pass
    
    def translate(self, text: str, target_language: str, source_language: str = None) -> str:
        """
        Translate text to target language.
        
        Args:
            text (str): Text to translate
            target_language (str): Target language code
            source_language (str): Source language code (optional)
            
        Returns:
            str: Translated text
        """
        pass
    
    def tokenize(self, text: str, language: str = None) -> list:
        """
        Tokenize text into words/tokens.
        
        Args:
            text (str): Input text
            language (str): Language code (optional)
            
        Returns:
            list: List of tokens
        """
        pass
    
    def process_mixed_language(self, text: str) -> dict:
        """
        Process mixed-language text.
        
        Args:
            text (str): Mixed-language text
            
        Returns:
            dict: Processed result with language segments
        """
        pass
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        pass
```

### Usage Example

```python
from bharat_fm.domain_modules.language_ai import LanguageAI

# Initialize Language AI
lang_ai = LanguageAI()

# Language detection
language = lang_ai.detect_language("नमस्ते, आप कैसे हैं?")
print(f"Language: {language}")

# Translation
translated = lang_ai.translate("Hello world", "hi")
print(f"Translated: {translated}")

# Tokenization
tokens = lang_ai.tokenize("भारत एक महान देश है")
print(f"Tokens: {tokens}")
```

## 🏛️ Governance AI API

### GovernanceAI Class

```python
class GovernanceAI:
    """
    Governance AI module for digital governance applications.
    """
    
    def __init__(self, config=None):
        """
        Initialize Governance AI.
        
        Args:
            config (dict): Configuration dictionary
        """
        pass
    
    def process_rti_request(self, query: str, department: str = None) -> dict:
        """
        Process RTI (Right to Information) request.
        
        Args:
            query (str): RTI query
            department (str): Department name (optional)
            
        Returns:
            dict: RTI response
        """
        pass
    
    def analyze_policy(self, policy_text: str, aspects: list = None) -> dict:
        """
        Analyze government policy.
        
        Args:
            policy_text (str): Policy text
            aspects (list): Aspects to analyze (optional)
            
        Returns:
            dict: Policy analysis
        """
        pass
    
    def check_compliance(self, document: str, regulations: list) -> dict:
        """
        Check document compliance with regulations.
        
        Args:
            document (str): Document path or text
            regulations (list): List of regulations
            
        Returns:
            dict: Compliance status
        """
        pass
    
    def generate_citizen_response(self, query: str, context: dict = None) -> str:
        """
        Generate response for citizen query.
        
        Args:
            query (str): Citizen query
            context (dict): Context information (optional)
            
        Returns:
            str: Generated response
        """
        pass
```

### Usage Example

```python
from bharat_fm.domain_modules.governance_ai import GovernanceAI

# Initialize Governance AI
gov_ai = GovernanceAI()

# Process RTI request
rti_response = gov_ai.process_rti_request(
    query="Information about PM Awas Yojana",
    department="Housing and Urban Affairs"
)
print(f"RTI Response: {rti_response}")

# Analyze policy
policy_analysis = gov_ai.analyze_policy(
    policy_text="National Education Policy 2020",
    aspects=["impact", "implementation", "budget"]
)
print(f"Policy Analysis: {policy_analysis}")
```

## 🎓 Education AI API

### EducationAI Class

```python
class EducationAI:
    """
    Education AI module for AI-powered education.
    """
    
    def __init__(self, config=None):
        """
        Initialize Education AI.
        
        Args:
            config (dict): Configuration dictionary
        """
        pass
    
    def create_learning_plan(self, student_id: str, subject: str, grade_level: str, 
                           learning_style: str = None) -> dict:
        """
        Create personalized learning plan.
        
        Args:
            student_id (str): Student identifier
            subject (str): Subject name
            grade_level (str): Grade level
            learning_style (str): Learning style (optional)
            
        Returns:
            dict: Learning plan
        """
        pass
    
    def generate_content(self, topic: str, language: str = None, 
                        content_type: str = "lesson") -> str:
        """
        Generate educational content.
        
        Args:
            topic (str): Topic for content generation
            language (str): Language code (optional)
            content_type (str): Type of content (optional)
            
        Returns:
            str: Generated content
        """
        pass
    
    def assess_student(self, student_answers: list, correct_answers: list, 
                      subject: str = None) -> dict:
        """
        Assess student performance.
        
        Args:
            student_answers (list): Student's answers
            correct_answers (list): Correct answers
            subject (str): Subject name (optional)
            
        Returns:
            dict: Assessment results
        """
        pass
    
    def provide_feedback(self, student_work: str, rubric: dict = None) -> str:
        """
        Provide feedback on student work.
        
        Args:
            student_work (str): Student's work
            rubric (dict): Assessment rubric (optional)
            
        Returns:
            str: Feedback
        """
        pass
```

### Usage Example

```python
from bharat_fm.domain_modules.education_ai import EducationAI

# Initialize Education AI
edu_ai = EducationAI()

# Create learning plan
learning_plan = edu_ai.create_learning_plan(
    student_id="student123",
    subject="Mathematics",
    grade_level="10",
    learning_style="visual"
)
print(f"Learning Plan: {learning_plan}")

# Generate content
content = edu_ai.generate_content(
    topic="Quadratic Equations",
    language="hi",
    content_type="lesson_plan"
)
print(f"Generated Content: {content}")
```

## 💰 Financial AI API

### FinancialAI Class

```python
class FinancialAI:
    """
    Financial AI module for financial services.
    """
    
    def __init__(self, config=None):
        """
        Initialize Financial AI.
        
        Args:
            config (dict): Configuration dictionary
        """
        pass
    
    def analyze_market(self, market: str, sector: str = None, 
                      time_period: str = "1M") -> dict:
        """
        Analyze market trends.
        
        Args:
            market (str): Market name
            sector (str): Sector name (optional)
            time_period (str): Time period (optional)
            
        Returns:
            dict: Market analysis
        """
        pass
    
    def detect_fraud(self, transaction_data: dict) -> dict:
        """
        Detect fraudulent transactions.
        
        Args:
            transaction_data (dict): Transaction data
            
        Returns:
            dict: Fraud detection result
        """
        pass
    
    def assess_risk(self, portfolio: list, risk_tolerance: str = "moderate") -> dict:
        """
        Assess portfolio risk.
        
        Args:
            portfolio (list): List of assets
            risk_tolerance (str): Risk tolerance level (optional)
            
        Returns:
            dict: Risk assessment
        """
        pass
    
    def generate_investment_advice(self, user_profile: dict, 
                                 financial_goals: list) -> dict:
        """
        Generate investment advice.
        
        Args:
            user_profile (dict): User profile
            financial_goals (list): Financial goals
            
        Returns:
            dict: Investment advice
        """
        pass
```

### Usage Example

```python
from bharat_fm.domain_modules.financial_ai import FinancialAI

# Initialize Financial AI
fin_ai = FinancialAI()

# Analyze market
market_analysis = fin_ai.analyze_market(
    market="NSE",
    sector="IT",
    time_period="1M"
)
print(f"Market Analysis: {market_analysis}")

# Detect fraud
fraud_result = fin_ai.detect_fraud({
    "amount": 100000,
    "account": "ACC123",
    "location": "Mumbai",
    "time": "2024-01-15 14:30:00"
})
print(f"Fraud Detection: {fraud_result}")
```

## 🧠 Memory System API

### ConversationMemory Class

```python
class ConversationMemory:
    """
    Memory system for storing and retrieving conversations.
    """
    
    def __init__(self, config=None):
        """
        Initialize conversation memory.
        
        Args:
            config (dict): Configuration dictionary
        """
        pass
    
    def add_message(self, role: str, content: str, metadata: dict = None):
        """
        Add message to memory.
        
        Args:
            role (str): Message role (user/assistant)
            content (str): Message content
            metadata (dict): Additional metadata (optional)
        """
        pass
    
    def get_conversation_history(self, limit: int = None) -> list:
        """
        Get conversation history.
        
        Args:
            limit (int): Number of messages to retrieve (optional)
            
        Returns:
            list: Conversation history
        """
        pass
    
    def search_conversations(self, query: str, limit: int = 10) -> list:
        """
        Search conversations.
        
        Args:
            query (str): Search query
            limit (int): Number of results (optional)
            
        Returns:
            list: Search results
        """
        pass
    
    def clear_memory(self):
        """Clear all memory."""
        pass
    
    def save_to_file(self, filepath: str):
        """
        Save memory to file.
        
        Args:
            filepath (str): File path
        """
        pass
    
    def load_from_file(self, filepath: str):
        """
        Load memory from file.
        
        Args:
            filepath (str): File path
        """
        pass
```

### Usage Example

```python
from bharat_fm.memory import ConversationMemory

# Initialize memory
memory = ConversationMemory()

# Add messages
memory.add_message("user", "Hello, how are you?")
memory.add_message("assistant", "I'm doing well, thank you!")

# Get conversation history
history = memory.get_conversation_history()
print(f"History: {history}")

# Search conversations
results = memory.search_conversations("hello")
print(f"Search Results: {results}")
```

## 🔧 Utilities API

### TextProcessor Class

```python
class TextProcessor:
    """
    Utility class for text processing operations.
    """
    
    def __init__(self, config=None):
        """
        Initialize text processor.
        
        Args:
            config (dict): Configuration dictionary
        """
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        pass
    
    def extract_entities(self, text: str, entity_types: list = None) -> list:
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            entity_types (list): Types of entities to extract (optional)
            
        Returns:
            list: Extracted entities
        """
        pass
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """
        Summarize text.
        
        Args:
            text (str): Input text
            max_length (int): Maximum summary length (optional)
            
        Returns:
            str: Summary
        """
        pass
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> list:
        """
        Extract keywords from text.
        
        Args:
            text (str): Input text
            max_keywords (int): Maximum number of keywords (optional)
            
        Returns:
            list: Keywords
        """
        pass
```

### Usage Example

```python
from bharat_fm.utils.text_processor import TextProcessor

# Initialize text processor
processor = TextProcessor()

# Clean text
cleaned = processor.clean_text("  Hello   World!  ")
print(f"Cleaned: {cleaned}")

# Extract entities
entities = processor.extract_entities("Rahul lives in Mumbai and works at TCS.")
print(f"Entities: {entities}")

# Summarize text
summary = processor.summarize_text("Long text content here...", max_length=50)
print(f"Summary: {summary}")
```

## ⚙️ Configuration API

### ConfigManager Class

```python
class ConfigManager:
    """
    Configuration management for BFMF.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file (str): Configuration file path (optional)
        """
        pass
    
    def get(self, key: str, default=None):
        """
        Get configuration value.
        
        Args:
            key (str): Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        pass
    
    def set(self, key: str, value):
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key
            value: Configuration value
        """
        pass
    
    def save(self, filepath: str = None):
        """
        Save configuration to file.
        
        Args:
            filepath (str): File path (optional)
        """
        pass
    
    def load(self, filepath: str):
        """
        Load configuration from file.
        
        Args:
            filepath (str): File path
        """
        pass
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        pass
```

### Usage Example

```python
from bharat_fm.config import ConfigManager

# Initialize configuration manager
config = ConfigManager()

# Set configuration values
config.set("log_level", "INFO")
config.set("cache_size", "1GB")

# Get configuration values
log_level = config.get("log_level")
cache_size = config.get("cache_size")
print(f"Log Level: {log_level}, Cache Size: {cache_size}")

# Save configuration
config.save("config.json")

# Load configuration
config.load("config.json")
```

## 🚨 Error Handling

### Exception Classes

```python
class BFMFError(Exception):
    """Base BFMF exception."""
    pass

class ConfigurationError(BFMFError):
    """Configuration-related error."""
    pass

class ModuleError(BFMFError):
    """Module-related error."""
    pass

class ProcessingError(BFMFError):
    """Processing-related error."""
    pass

class MemoryError(BFMFError):
    """Memory-related error."""
    pass
```

### Error Handling Example

```python
from bharat_fm import BharatFM
from bharat_fm.exceptions import BFMFError, ConfigurationError

try:
    # Initialize BFMF
    bfmf = BharatFM()
    
    # Process request
    result = bfmf.chat("Hello")
    print(result)
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ModuleError as e:
    print(f"Module error: {e}")
except ProcessingError as e:
    print(f"Processing error: {e}")
except BFMFError as e:
    print(f"BFMF error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📝 Response Formats

### Standard Response Format

```python
{
    "success": bool,
    "data": any,
    "message": str,
    "timestamp": str,
    "request_id": str
}
```

### Error Response Format

```python
{
    "success": False,
    "error": {
        "code": str,
        "message": str,
        "details": dict
    },
    "timestamp": str,
    "request_id": str
}
```

### Chat Response Format

```python
{
    "response": str,
    "language": str,
    "confidence": float,
    "modules_used": list,
    "processing_time": float
}
```

This API reference covers the main components of BFMF. For more detailed information about specific modules or advanced usage, please refer to the respective module documentation or the [examples](../examples/) directory.