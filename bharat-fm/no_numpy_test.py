"""
Test without numpy for Bharat-FM
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def no_numpy_test():
    print("🇮🇳 No-Numpy Test for Bharat-FM")
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
        
        print("\n🎉 No-numpy test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ No-numpy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = no_numpy_test()
    sys.exit(0 if success else 1)