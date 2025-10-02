#!/usr/bin/env python3
"""
Test script for OSNet integration.
Tests the hybrid identification system with sample images.
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from osnet_extractor import OSNetExtractor, get_osnet_extractor
    from hybrid_identifier import HybridIdentifier, get_hybrid_identifier
    from database import MatchesDatabase
    print("✓ Successfully imported OSNet and hybrid modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_osnet_extractor():
    """Test OSNet feature extraction."""
    print("\n=== Testing OSNet Feature Extractor ===")

    try:
        # Initialize extractor
        extractor = get_osnet_extractor()
        print(f"✓ OSNet extractor initialized: {extractor.model_name}")

        # Create a dummy test image (256x128x3)
        test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        print(f"✓ Created test image: {test_image.shape}")

        # Extract features
        features = extractor.extract_features(test_image)
        print(f"✓ Extracted features: shape={features.shape}, dtype={features.dtype}")

        # Test similarity computation
        features2 = extractor.extract_features(test_image)
        similarity = extractor.compute_similarity(features, features2)
        print(f"✓ Self-similarity: {similarity:.4f} (should be ~1.0)")

        # Test with different image
        test_image2 = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        features3 = extractor.extract_features(test_image2)
        similarity2 = extractor.compute_similarity(features, features3)
        print(f"✓ Different image similarity: {similarity2:.4f} (should be lower)")

        return True

    except Exception as e:
        print(f"✗ OSNet test failed: {e}")
        return False

def test_hybrid_identifier():
    """Test hybrid identification system."""
    print("\n=== Testing Hybrid Identifier ===")

    try:
        # Initialize hybrid identifier
        identifier = get_hybrid_identifier()
        print("✓ Hybrid identifier initialized")

        # Get model info
        stats = identifier.get_identification_stats()
        print(f"✓ Model info: {stats}")

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Extract all features
        features = identifier.extract_all_features(test_image)
        print(f"✓ Extracted features: {list(features.keys())}")

        # Test identification (should return None for random image)
        result = identifier.identify_person(test_image)
        if result:
            print(f"✓ Identification result: {result.person_id} ({result.confidence:.3f})")
        else:
            print("✓ No identification (expected for random image)")

        return True

    except Exception as e:
        print(f"✗ Hybrid identifier test failed: {e}")
        return False

def test_database_integration():
    """Test database integration with hybrid features."""
    print("\n=== Testing Database Integration ===")

    try:
        # Initialize database
        db = MatchesDatabase('test_matches.db')
        print("✓ Database initialized")

        # Test hybrid match storage
        test_match_id = db.add_hybrid_match(
            subject="test_person",
            confidence=85.5,
            camera="test_camera",
            identification_method="hybrid",
            method_scores={"osnet": 0.8, "face": 0.9, "color": 0.7},
            bbox=[100, 100, 200, 300]
        )
        print(f"✓ Added hybrid match: ID={test_match_id}")

        # Test feature storage
        dummy_features = np.random.rand(512).astype(np.float32)
        feature_id = db.add_person_features("test_person", dummy_features.tobytes())
        print(f"✓ Added person features: ID={feature_id}")

        # Test color features
        dummy_colors = np.random.rand(96).astype(np.float32)
        color_id = db.add_color_features("test_person", dummy_colors.tobytes())
        print(f"✓ Added color features: ID={color_id}")

        # Test retrieval
        hybrid_matches = db.get_hybrid_matches(limit=10)
        print(f"✓ Retrieved {len(hybrid_matches)} hybrid matches")

        stats = db.get_hybrid_stats()
        print(f"✓ Hybrid stats: {stats}")

        # Clean up test data
        db.clear_all_hybrid_data()
        print("✓ Cleaned up test data")

        return True

    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_pytorch_compatibility():
    """Test PyTorch installation and compatibility."""
    print("\n=== Testing PyTorch Compatibility ===")

    try:
        import torch
        import torchvision
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ TorchVision version: {torchvision.__version__}")

        # Test CUDA availability
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")

        # Test tensor operations
        x = torch.randn(1, 3, 256, 128)
        y = torch.nn.functional.normalize(x, p=2, dim=1)
        print(f"✓ Tensor operations working: {y.shape}")

        return True

    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("OSNet Integration Test Suite")
    print("=" * 50)

    tests = [
        test_pytorch_compatibility,
        test_osnet_extractor,
        test_hybrid_identifier,
        test_database_integration
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Results:")
    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! OSNet integration is working.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())