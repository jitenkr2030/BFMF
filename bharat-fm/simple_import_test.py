"""
Simple import test for BFMF
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def simple_import_test():
    print("🇮🇳 Simple Import Test for Bharat-FM")
    print("=" * 40)
    
    try:
        print("Testing basic imports...")
        
        # Test basic Python imports
        import asyncio
        import time
        import json
        import pickle
        import hashlib
        from typing import Dict, List, Any, Optional, Tuple
        from dataclasses import dataclass, field
        from datetime import datetime, timedelta
        from collections import defaultdict, deque
        import logging
        print("✅ Basic Python imports successful")
        
        # Test NumPy
        import numpy as np
        print("✅ NumPy import successful")
        print(f"   NumPy version: {np.__version__}")
        
        # Test scikit-learn
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ scikit-learn import successful")
        
        # Test psutil
        import psutil
        print("✅ psutil import successful")
        print(f"   psutil version: {psutil.__version__}")
        
        print("\n🎉 Simple import test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_import_test()
    sys.exit(0 if success else 1)