# Getting Started with BFMF

This guide will help you get up and running with the Bharat Foundation Model Framework (BFMF) quickly.

## 📋 Prerequisites

Before installing BFMF, ensure you have the following:

### System Requirements
- **OS**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: Minimum 10GB free space
- **GPU**: Optional but recommended for better performance

### Software Dependencies
```bash
# Python 3.8+
python --version

# pip (Python package manager)
pip --version

# git (for cloning repository)
git --version
```

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv bharat_fm_env

# Activate virtual environment
# On Linux/macOS:
source bharat_fm_env/bin/activate

# On Windows:
bharat_fm_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 4: Initialize BFMF

```bash
# Run setup script
python setup.py install

# Initialize configuration
python -m bharat_fm init
```

## ⚙️ Configuration

### Environment Setup

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# BFMF Configuration
BFMF_DATA_DIR=./data
BFMF_CACHE_DIR=./cache
BFMF_LOG_LEVEL=INFO

# Database Configuration (optional)
DATABASE_URL=sqlite:///./bharat_fm.db

# API Keys (for external services)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```

### First Run

```python
from bharat_fm import BharatFM

# Initialize BFMF
bfmf = BharatFM()

# Test the installation
print("BFMF Version:", bfmf.version)
print("Available Languages:", bfmf.get_supported_languages())
```

## 🎯 Basic Usage

### 1. Chat Interface

```python
from bharat_fm import BharatFM

# Initialize
bfmf = BharatFM()

# Basic chat
response = bfmf.chat("Hello! How are you?")
print(response)

# Chat in different languages
response_hi = bfmf.chat("नमस्ते, आप कैसे हैं?")
response_bn = bfmf.chat("হ্যালো, আপনি কেমন আছেন?")
```

### 2. Text Processing

```python
from bharat_fm.core.text_processing import TextProcessor

# Initialize text processor
processor = TextProcessor()

# Tokenization
tokens = processor.tokenize("भारत एक महान देश है")
print("Tokens:", tokens)

# Language detection
language = processor.detect_language("Hello world")
print("Language:", language)

# Translation
translated = processor.translate("Hello", target_language="hi")
print("Translated:", translated)
```

### 3. Memory Management

```python
from bharat_fm.memory import ConversationMemory

# Initialize memory
memory = ConversationMemory()

# Store conversation
memory.add_message("user", "What is BFMF?")
memory.add_message("assistant", "BFMF is India's sovereign AI framework")

# Retrieve conversation history
history = memory.get_conversation_history()
print("History:", history)
```

## 🏗️ Project Structure

```
bharat-fm/
├── bharat_fm/                 # Main BFMF package
│   ├── core/                 # Core components
│   │   ├── inference_engine.py
│   │   ├── memory_system.py
│   │   └── security_layer.py
│   ├── domain_modules/       # Domain-specific modules
│   │   ├── language_ai/
│   │   ├── governance_ai/
│   │   └── education_ai/
│   ├── data/                 # Data processing
│   │   ├── tokenization.py
│   │   └── preprocessing.py
│   └── utils/                # Utility functions
├── docs/                     # Documentation
├── examples/                 # Example scripts
├── tests/                    # Test suite
├── requirements.txt          # Dependencies
├── setup.py                  # Installation script
└── README.md                # Project overview
```

## 🔧 CLI Usage

BFMF provides a comprehensive command-line interface:

### Basic Commands

```bash
# Start interactive chat
bharat-fm chat

# Process text file
bharat-fm process --input file.txt --language hi

# Train a model
bharat-fm train --data ./data --model my_model

# Deploy model
bharat-fm deploy --model my_model --port 8000
```

### Configuration Commands

```bash
# Show configuration
bharat-fm config show

# Update configuration
bharat-fm config set --key log_level --value DEBUG

# Initialize new project
bharat-fm init --project my_project
```

## 🧪 Testing Your Installation

Run the test suite to verify everything is working:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_core.py
python -m pytest tests/test_memory.py
python -m pytest tests/test_domain_modules.py

# Run with coverage
python -m pytest --cov=bharat_fm tests/
```

### Sample Test Script

```python
# test_installation.py
from bharat_fm import BharatFM
from bharat_fm.core.text_processing import TextProcessor
from bharat_fm.memory import ConversationMemory

def test_basic_functionality():
    # Test BFMF initialization
    bfmf = BharatFM()
    assert bfmf.version is not None
    
    # Test text processing
    processor = TextProcessor()
    tokens = processor.tokenize("test")
    assert len(tokens) > 0
    
    # Test memory system
    memory = ConversationMemory()
    memory.add_message("user", "test")
    history = memory.get_conversation_history()
    assert len(history) > 0
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, try:
pip install -e .
```

#### 2. Memory Issues
```bash
# If you run into memory issues:
export BFMF_CACHE_SIZE=1GB
```

#### 3. GPU Not Detected
```bash
# Check GPU availability:
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Language Support Issues
```bash
# Verify language support:
python -c "from bharat_fm import BharatFM; print(BharatFM().get_supported_languages())"
```

### Getting Help

- **Documentation**: Check the [FAQ](./faq.md)
- **GitHub Issues**: [Report issues](https://github.com/bharat-ai/bharat-fm/issues)
- **Community**: Join our [Discord server](https://discord.gg/bharat-ai)
- **Email**: support@bharat-ai.org

## 🎉 Next Steps

Now that you have BFMF installed and running, you can:

1. **Explore Domain Modules**: Learn about [specialized AI modules](./domain-modules.md)
2. **Build Your First App**: Follow the [tutorial](./tutorials/first-app.md)
3. **Deploy to Production**: Read the [deployment guide](./deployment.md)
4. **Contribute**: Check the [contributing guide](./contributing.md)

Happy coding! 🚀