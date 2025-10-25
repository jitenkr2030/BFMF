"""
Test full BFMF module import
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_import():
    print("🇮🇳 Testing Full BFMF Module Import")
    print("=" * 40)
    
    try:
        print("Testing full BFMF import...")
        
        # Try to import the full BFMF module
        import bharat_fm
        print("✅ Full BFMF module import successful")
        
        # Test accessing package info
        print("\nTesting package info...")
        package_info = bharat_fm.get_package_info()
        print(f"✅ Package name: {package_info['name']}")
        print(f"✅ Package version: {package_info['version']}")
        
        # Test accessing capabilities
        print("\nTesting capabilities...")
        capabilities = bharat_fm.get_capabilities()
        print(f"✅ Number of capability categories: {len(capabilities)}")
        
        # Test accessing supported features
        print("\nTesting supported features...")
        features = bharat_fm.get_supported_features()
        print(f"✅ Number of supported features: {len(features)}")
        
        print("\n🎉 Full BFMF import test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Full BFMF import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_import()
    sys.exit(0 if success else 1)