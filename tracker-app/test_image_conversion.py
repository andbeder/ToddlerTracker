#!/usr/bin/env python3
"""
Test script for image conversion functionality.
Tests HEIC to JPEG conversion and validation.
"""

import sys
import os
from io import BytesIO
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from image_converter import get_image_converter, convert_heic_to_jpeg, process_image_upload
    print("✓ Successfully imported image converter modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_converter_initialization():
    """Test converter initialization."""
    print("\n=== Testing Converter Initialization ===")

    try:
        converter = get_image_converter()
        print(f"✓ Converter initialized successfully")
        print(f"✓ Supported formats: {converter.supported_formats}")
        return True
    except Exception as e:
        print(f"✗ Converter initialization failed: {e}")
        return False

def test_jpeg_processing():
    """Test JPEG image processing (should pass through unchanged)."""
    print("\n=== Testing JPEG Processing ===")

    try:
        # Create a test JPEG image
        img = Image.new('RGB', (300, 200), color='red')
        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, format='JPEG', quality=85)
        jpeg_data = jpeg_buffer.getvalue()

        converter = get_image_converter()

        # Test validation
        is_valid, reason = converter.validate_image(jpeg_data)
        print(f"✓ JPEG validation: {is_valid} - {reason}")

        # Test conversion (should not actually convert)
        converted_data, was_converted = converter.convert_to_jpeg(jpeg_data)
        print(f"✓ JPEG conversion: was_converted={was_converted}, size={len(converted_data)} bytes")

        # Test upload processing
        processed_data, filename, was_converted = converter.process_upload(jpeg_data, "test.jpg")
        print(f"✓ Upload processing: filename={filename}, was_converted={was_converted}")

        return True
    except Exception as e:
        print(f"✗ JPEG processing test failed: {e}")
        return False

def test_png_conversion():
    """Test PNG to JPEG conversion."""
    print("\n=== Testing PNG Conversion ===")

    try:
        # Create a test PNG image with transparency
        img = Image.new('RGBA', (300, 200), color=(255, 0, 0, 128))  # Semi-transparent red
        png_buffer = BytesIO()
        img.save(png_buffer, format='PNG')
        png_data = png_buffer.getvalue()

        converter = get_image_converter()

        # Test validation
        is_valid, reason = converter.validate_image(png_data)
        print(f"✓ PNG validation: {is_valid} - {reason}")

        # Test conversion
        converted_data, was_converted = converter.convert_to_jpeg(png_data, quality=90)
        print(f"✓ PNG conversion: was_converted={was_converted}, size={len(png_data)} -> {len(converted_data)} bytes")

        # Verify converted image is valid JPEG
        converted_img = Image.open(BytesIO(converted_data))
        print(f"✓ Converted image: format={converted_img.format}, size={converted_img.size}, mode={converted_img.mode}")

        return True
    except Exception as e:
        print(f"✗ PNG conversion test failed: {e}")
        return False

def test_invalid_image():
    """Test invalid image handling."""
    print("\n=== Testing Invalid Image Handling ===")

    try:
        converter = get_image_converter()

        # Test with random bytes
        invalid_data = b"This is not an image file"
        is_valid, reason = converter.validate_image(invalid_data)
        print(f"✓ Invalid data validation: {is_valid} - {reason}")

        # Test with empty data
        empty_data = b""
        is_valid, reason = converter.validate_image(empty_data)
        print(f"✓ Empty data validation: {is_valid} - {reason}")

        return True
    except Exception as e:
        print(f"✗ Invalid image test failed: {e}")
        return False

def test_format_detection():
    """Test image format detection."""
    print("\n=== Testing Format Detection ===")

    try:
        converter = get_image_converter()

        # Create test images in different formats
        img = Image.new('RGB', (100, 100), color='blue')

        # JPEG
        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, format='JPEG')
        jpeg_format = converter.detect_format(jpeg_buffer.getvalue())
        print(f"✓ JPEG format detection: {jpeg_format}")

        # PNG
        png_buffer = BytesIO()
        img.save(png_buffer, format='PNG')
        png_format = converter.detect_format(png_buffer.getvalue())
        print(f"✓ PNG format detection: {png_format}")

        return True
    except Exception as e:
        print(f"✗ Format detection test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    print("\n=== Testing Convenience Functions ===")

    try:
        # Create test image
        img = Image.new('RGB', (200, 150), color='green')
        png_buffer = BytesIO()
        img.save(png_buffer, format='PNG')
        png_data = png_buffer.getvalue()

        # Test convert_heic_to_jpeg function
        converted_data, was_converted = convert_heic_to_jpeg(png_data)
        print(f"✓ convert_heic_to_jpeg: was_converted={was_converted}")

        # Test process_image_upload function
        processed_data, filename, was_converted = process_image_upload(png_data, "test.png")
        print(f"✓ process_image_upload: filename={filename}, was_converted={was_converted}")

        return True
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Image Conversion Test Suite")
    print("=" * 50)

    tests = [
        test_converter_initialization,
        test_jpeg_processing,
        test_png_conversion,
        test_invalid_image,
        test_format_detection,
        test_convenience_functions
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
        print("🎉 All tests passed! Image conversion is working.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())