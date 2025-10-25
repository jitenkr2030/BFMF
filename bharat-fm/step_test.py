"""
Step-by-step import test for Bharat-FM
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def step_test():
    print("🇮🇳 Step-by-Step Import Test")
    print("=" * 40)
    
    try:
        print("Step 1: Testing sys import...")
        import sys
        print("✅ sys imported")
        
        print("Step 2: Testing os import...")
        import os
        print("✅ os imported")
        
        print("Step 3: Testing asyncio import...")
        import asyncio
        print("✅ asyncio imported")
        
        print("Step 4: Testing time import...")
        import time
        print("✅ time imported")
        
        print("Step 5: Testing typing import...")
        from typing import Dict, List, Any, Optional, Tuple
        print("✅ typing imported")
        
        print("Step 6: Testing dataclasses import...")
        from dataclasses import dataclass, field
        print("✅ dataclasses imported")
        
        print("Step 7: Testing datetime import...")
        from datetime import datetime, timedelta
        print("✅ datetime imported")
        
        print("Step 8: Testing logging import...")
        import logging
        print("✅ logging imported")
        
        print("Step 9: Testing collections import...")
        from collections import defaultdict, deque
        print("✅ collections imported")
        
        print("Step 10: Testing hashlib import...")
        import hashlib
        print("✅ hashlib imported")
        
        print("Step 11: Testing json import...")
        import json
        print("✅ json imported")
        
        print("Step 12: Testing pickle import...")
        import pickle
        print("✅ pickle imported")
        
        print("Step 13: Testing numpy import...")
        import numpy as np
        print("✅ numpy imported")
        
        print("Step 14: Testing sklearn import...")
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ sklearn imported")
        
        print("\n🎉 All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = step_test()
    sys.exit(0 if success else 1)