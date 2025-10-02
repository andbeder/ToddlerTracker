#!/usr/bin/env python3
"""
Performance test script for optimized CUDA boundary detection and rasterization.
Tests the new optimized implementations against the 15M+ point fused.ply file.
"""

import time
import sys
import json
from yard_manager import YardManager

def test_boundary_detection():
    """Test optimized boundary detection performance."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZED BOUNDARY DETECTION")
    print("="*60)

    yard = YardManager('yard.db')

    # Test with stored PLY from database
    print("\nProcessing stored fused.ply from database...")
    print("File size: 417 MB")

    start_time = time.time()
    result = yard.scan_boundaries_stored(ply_id=1)  # Using the stored fused.ply
    end_time = time.time()

    elapsed = end_time - start_time

    if result['status'] == 'success':
        boundaries = result['boundaries']
        print(f"\n✅ Boundary detection successful!")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"📊 Total points: {boundaries.get('total_points', 'N/A'):,}")
        print(f"📊 Filtered points: {boundaries.get('filtered_points', 'N/A'):,}")
        print(f"📊 Filter percentage: {boundaries.get('filter_percentage', 'N/A'):.1f}%")
        print(f"🚀 CUDA accelerated: {boundaries.get('cuda_accelerated', False)}")
        print(f"🎯 Method used: {boundaries.get('method', 'unknown')}")
        print(f"\nBoundaries:")
        print(f"  X: [{boundaries['x_min']:.2f}, {boundaries['x_max']:.2f}] meters")
        print(f"  Z: [{boundaries['z_min']:.2f}, {boundaries['z_max']:.2f}] meters")
        print(f"  Center: ({boundaries['center_x']:.2f}, {boundaries['center_z']:.2f})")
        print(f"  Dimensions: {boundaries['width']:.2f}m x {boundaries['height']:.2f}m")

        return elapsed, boundaries
    else:
        print(f"❌ Error: {result['message']}")
        return None, None

def test_rasterization(boundaries):
    """Test optimized rasterization performance."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZED RASTERIZATION")
    print("="*60)

    if not boundaries:
        print("❌ Cannot test rasterization without boundaries")
        return None

    yard = YardManager('yard.db')

    # Test interactive projection with optimized rasterizer
    print("\nGenerating yard map with optimized spatial hash grid...")
    print("Resolution: 800x600 pixels")
    print("Resolution per pixel: 0.01 meters")

    start_time = time.time()
    result = yard.project_yard_interactive(
        file_path=None,  # Use stored PLY
        center_x=boundaries['center_x'],
        center_y=boundaries['center_z'],  # Note: Z becomes Y in 2D projection
        rotation=0.0,
        resolution=0.01,
        output_size=(800, 600)
    )
    end_time = time.time()

    elapsed = end_time - start_time

    if result['status'] == 'success':
        print(f"\n✅ Rasterization successful!")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"📊 Points processed: {result.get('point_count', 'N/A'):,}")
        print(f"🚀 CUDA accelerated: {result.get('cuda_accelerated', False)}")
        print(f"📐 Output size: {result['width']}x{result['height']}")
        print(f"📍 Center: ({result['center_x']:.2f}, {result['center_y']:.2f})")

        # Calculate pixels per second
        total_pixels = result['width'] * result['height']
        pixels_per_second = total_pixels / elapsed if elapsed > 0 else 0
        print(f"⚡ Performance: {pixels_per_second:,.0f} pixels/second")

        return elapsed
    else:
        print(f"❌ Error: {result['message']}")
        return None

def main():
    """Run performance tests."""
    print("\n" + "="*60)
    print("OPTIMIZED CUDA PERFORMANCE TEST")
    print("Testing with 15M+ point fused.ply file")
    print("="*60)

    # Test boundary detection
    boundary_time, boundaries = test_boundary_detection()

    # Test rasterization
    raster_time = test_rasterization(boundaries)

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    if boundary_time:
        print(f"\n🎯 Boundary Detection:")
        print(f"   Time: {boundary_time:.2f} seconds")

        # Compare with expected non-optimized time (based on previous observations)
        # Previous implementation was taking ~47 seconds
        expected_old_time = 47.0
        speedup = expected_old_time / boundary_time if boundary_time > 0 else 0
        print(f"   Expected speedup: {speedup:.1f}x faster")

    if raster_time:
        print(f"\n🎯 Rasterization:")
        print(f"   Time: {raster_time:.2f} seconds")

        # Previous rasterization was taking ~10+ seconds
        expected_old_time = 10.0
        speedup = expected_old_time / raster_time if raster_time > 0 else 0
        print(f"   Expected speedup: {speedup:.1f}x faster")

    if boundary_time and raster_time:
        total_time = boundary_time + raster_time
        print(f"\n⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"🚀 Ready for real-time yard mapping!")

if __name__ == '__main__':
    main()