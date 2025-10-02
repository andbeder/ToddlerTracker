#!/usr/bin/env python3
"""
Test script for Toddler Position Tracking System (TPTS)
Validates the end-to-end position tracking pipeline.
"""

import sys
from position_tracker import PositionTracker
import numpy as np


def test_bbox_conversion():
    """Test bounding box conversion from Frigate format to pixels."""
    print("\n=== Testing Bounding Box Conversion ===")

    tracker = PositionTracker()

    # Test case from TPTS design doc
    frigate_bbox = [0.654, 0.507, 0.054, 0.161]  # [x_center, y_center, width, height] normalized
    camera_width = 2560
    camera_height = 1920

    bbox_pixels = tracker.convert_bbox_to_pixels(frigate_bbox, camera_width, camera_height)
    print(f"Frigate bbox (normalized): {frigate_bbox}")
    print(f"Pixel bbox [x, y, w, h]: {bbox_pixels}")

    # Verify conversion
    expected_x = int((0.654 - 0.054/2) * 2560)
    expected_y = int((0.507 - 0.161/2) * 1920)
    expected_w = int(0.054 * 2560)
    expected_h = int(0.161 * 1920)

    assert bbox_pixels[0] == expected_x, f"X mismatch: {bbox_pixels[0]} != {expected_x}"
    assert bbox_pixels[1] == expected_y, f"Y mismatch: {bbox_pixels[1]} != {expected_y}"
    assert bbox_pixels[2] == expected_w, f"W mismatch: {bbox_pixels[2]} != {expected_w}"
    assert bbox_pixels[3] == expected_h, f"H mismatch: {bbox_pixels[3]} != {expected_h}"

    print("✓ Bbox conversion test passed!")


def test_feet_position():
    """Test feet position extraction."""
    print("\n=== Testing Feet Position Extraction ===")

    tracker = PositionTracker()

    frigate_bbox = [0.654, 0.507, 0.054, 0.161]
    camera_width = 2560
    camera_height = 1920

    cam_x, cam_y = tracker.get_feet_position(frigate_bbox, camera_width, camera_height)
    print(f"Feet position (cam_x, cam_y): ({cam_x}, {cam_y})")

    # Verify feet position (bottom-center of bbox)
    expected_x = int(0.654 * 2560)
    expected_y = int((0.507 + 0.161/2) * 1920)

    assert cam_x == expected_x, f"Feet X mismatch: {cam_x} != {expected_x}"
    assert cam_y == expected_y, f"Feet Y mismatch: {cam_y} != {expected_y}"

    print(f"✓ Feet position test passed! Position: ({cam_x}, {cam_y})")


def test_projection_lookup():
    """Test map position lookup (requires pre-computed projection)."""
    print("\n=== Testing Projection Lookup ===")

    tracker = PositionTracker()

    # Test with example camera and map
    camera_name = "side_yard"
    map_id = 1

    # Example feet position
    cam_x, cam_y = 1674, 1128

    try:
        map_position = tracker.lookup_map_position(camera_name, map_id, cam_x, cam_y)

        if map_position:
            map_x, map_y = map_position
            print(f"✓ Projection lookup successful!")
            print(f"  Camera position: ({cam_x}, {cam_y})")
            print(f"  Map position: ({map_x}, {map_y})")
        else:
            print(f"⚠ No projection found for camera '{camera_name}', map {map_id}")
            print(f"  Make sure you've generated a projection for this camera first.")

    except Exception as e:
        print(f"⚠ Projection lookup error: {e}")
        print(f"  This is expected if projection hasn't been generated yet.")


def test_position_storage():
    """Test position storage in database."""
    print("\n=== Testing Position Storage ===")

    tracker = PositionTracker()

    # Store a test position
    result = tracker.store_toddler_position(
        subject="TestToddler",
        camera="test_camera",
        map_x=640,
        map_y=360,
        confidence=0.85
    )

    if result:
        print("✓ Position storage test passed!")
        print("  Stored position: (640, 360) with confidence 0.85")
    else:
        print("✗ Position storage failed!")


def test_complete_pipeline():
    """Test the complete detection pipeline."""
    print("\n=== Testing Complete Pipeline ===")

    tracker = PositionTracker()

    # Simulate a Frigate detection
    camera_name = "side_yard"
    map_id = 1
    frigate_bbox = [0.654, 0.507, 0.054, 0.161]
    camera_width = 2560
    camera_height = 1920
    subject = "Toddler"
    confidence = 0.85

    try:
        result = tracker.process_detection(
            camera_name=camera_name,
            map_id=map_id,
            frigate_bbox=frigate_bbox,
            camera_width=camera_width,
            camera_height=camera_height,
            subject=subject,
            confidence=confidence
        )

        if result:
            map_x, map_y = result
            print(f"✓ Complete pipeline test passed!")
            print(f"  Input: Frigate bbox {frigate_bbox} from {camera_name}")
            print(f"  Output: Map position ({map_x}, {map_y})")
            print(f"  Confidence: {confidence}")
        else:
            print(f"⚠ Pipeline test incomplete - projection not found")
            print(f"  Generate a projection for '{camera_name}' first")

    except Exception as e:
        print(f"⚠ Pipeline test error: {e}")


def test_frigate_events():
    """Test Frigate events API."""
    print("\n=== Testing Frigate Events API ===")

    tracker = PositionTracker()

    try:
        events = tracker.get_frigate_events(camera="side_yard", limit=3)

        if events:
            print(f"✓ Retrieved {len(events)} events from Frigate")
            for i, event in enumerate(events[:2], 1):
                print(f"\n  Event {i}:")
                print(f"    ID: {event.get('id')}")
                print(f"    Camera: {event.get('camera')}")
                print(f"    Label: {event.get('label')}")
                bbox = event.get('data', {}).get('box')
                if bbox:
                    print(f"    Bbox: {bbox}")
        else:
            print("⚠ No events retrieved from Frigate")
            print("  Make sure Frigate is running and has recent person detections")

    except Exception as e:
        print(f"⚠ Frigate events test error: {e}")
        print("  Make sure Frigate is running at http://localhost:5000")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Toddler Position Tracking System (TPTS) - Test Suite")
    print("=" * 60)

    # Basic tests (always work)
    test_bbox_conversion()
    test_feet_position()
    test_position_storage()

    # Tests requiring setup
    test_projection_lookup()
    test_complete_pipeline()
    test_frigate_events()

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Generate camera projections in the Projection tab")
    print("2. Set active map ID: POST /set_active_map with {\"map_id\": 1}")
    print("3. Enable detection service to start tracking positions")
    print("4. View live positions on the Map tab")


if __name__ == "__main__":
    main()
