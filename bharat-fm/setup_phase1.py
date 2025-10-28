"""
Phase 1 Setup Script for Bharat-FM Enhanced Capabilities
Sets up the environment and installs dependencies
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    dependencies = [
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "asyncio-mqtt>=0.11.0",
        "aiofiles>=0.8.0",
        "psutil>=5.8.0",
        "pydantic>=1.10.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {dep}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "bharat-fm/src/bharat_fm",
        "bharat-fm/src/bharat_fm/core",
        "bharat-fm/src/bharat_fm/memory",
        "bharat-fm/src/bharat_fm/optimization",
        "bharat-fm/src/bharat_fm/api",
        "bharat-fm/src/bharat_fm/shared",
        "bharat-fm/examples",
        "bharat-fm/cache",
        "bharat-fm/conversation_memory",
        "bharat-fm/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {directory}")

def create_init_files():
    """Create __init__.py files"""
    print("🔧 Creating __init__.py files...")
    
    init_files = [
        "bharat-fm/src/__init__.py",
        "bharat-fm/src/bharat_fm/__init__.py",
        "bharat-fm/src/bharat_fm/core/__init__.py",
        "bharat-fm/src/bharat_fm/memory/__init__.py",
        "bharat-fm/src/bharat_fm/optimization/__init__.py",
        "bharat-fm/src/bharat_fm/api/__init__.py",
        "bharat-fm/src/bharat_fm/shared/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"   ✅ Created {init_file}")

def create_config_file():
    """Create configuration file"""
    print("⚙️ Creating configuration file...")
    
    config_content = '''"""
Bharat-FM Phase 1 Configuration
"""

# Inference Optimization Configuration
INFERENCE_CONFIG = {
    "optimization_enabled": True,
    "caching_enabled": True,
    "batching_enabled": True,
    "cost_monitoring_enabled": True,
    "max_cache_entries": 10000,
    "max_batch_size": 8,
    "max_wait_time": 0.1,
    "cache_dir": "./cache",
    "similarity_threshold": 0.95
}

# Conversation Memory Configuration
MEMORY_CONFIG = {
    "memory_dir": "./conversation_memory",
    "max_history_length": 1000,
    "max_sessions_per_user": 10,
    "context_retention_days": 30
}

# Chat Engine Configuration
CHAT_CONFIG = {
    "inference": INFERENCE_CONFIG,
    "memory": MEMORY_CONFIG
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "logs/bharat_fm.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}
'''
    
    with open("bharat-fm/src/bharat_fm/config.py", "w") as f:
        f.write(config_content)
    
    print("   ✅ Created config.py")

def create_requirements_file():
    """Create requirements.txt file"""
    print("📋 Creating requirements.txt...")
    
    requirements = '''numpy>=1.21.0
scikit-learn>=1.0.0
asyncio-mqtt>=0.11.0
aiofiles>=0.8.0
psutil>=5.8.0
pydantic>=1.10.0
fastapi>=0.95.0
uvicorn>=0.20.0
httpx>=0.24.0
pillow>=9.0.0
soundfile>=0.12.0
opencv-python>=4.5.0
'''
    
    with open("bharat-fm/requirements.txt", "w") as f:
        f.write(requirements)
    
    print("   ✅ Created requirements.txt")

def create_gitignore():
    """Create .gitignore file"""
    print("🚫 Creating .gitignore...")
    
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
cache/
conversation_memory/
logs/
*.log
demo_memory/
demo_chat_memory/
.pytest_cache/
.coverage
htmlcov/

# Temporary files
*.tmp
*.temp
'''
    
    with open("bharat-fm/.gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("   ✅ Created .gitignore")

async def test_setup():
    """Test the setup by running the demo"""
    print("🧪 Testing setup...")
    
    try:
        # Change to bharat-fm directory
        os.chdir("bharat-fm")
        
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), "src"))
        
        # Import and test basic functionality
        from bharat_fm.core.inference_engine import InferenceEngine
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        from bharat_fm.core.chat_engine import ChatEngine
        
        print("   ✅ All modules imported successfully")
        
        # Test basic initialization
        print("   🔄 Testing basic initialization...")
        
        # Test inference engine
        engine = InferenceEngine()
        print("   ✅ Inference engine created")
        
        # Test memory manager
        memory = ConversationMemoryManager()
        print("   ✅ Memory manager created")
        
        # Test chat engine
        chat = ChatEngine()
        print("   ✅ Chat engine created")
        
        print("   🎉 Setup test passed!")
        
    except Exception as e:
        print(f"   ❌ Setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main setup function"""
    print("🇮🇳 Bharat Foundation Model Framework - Phase 1 Setup")
    print("=" * 60)
    print("Setting up enhanced capabilities:")
    print("1. Real-time Inference Optimization")
    print("2. Conversational Memory & Context Management")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Create init files
    create_init_files()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Create configuration and other files
    create_config_file()
    create_requirements_file()
    create_gitignore()
    
    # Test setup
    print("\n🧪 Testing setup...")
    if not asyncio.run(test_setup()):
        print("❌ Setup test failed")
        return False
    
    print("\n🎉 Phase 1 Setup Completed Successfully!")
    print("\n📋 Next Steps:")
    print("1. Navigate to bharat-fm directory: cd bharat-fm")
    print("2. Run the demo: python examples/phase1_demo.py")
    print("3. Explore the enhanced capabilities")
    print("\n📚 Documentation:")
    print("- Check the src/bharat_fm/ directory for implementation")
    print("- Review examples/phase1_demo.py for usage examples")
    print("- Configuration is in src/bharat_fm/config.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)