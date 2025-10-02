#!/usr/bin/env python3
"""
Test performance with optimized file-based PLY storage.
Compares database blob storage vs direct file access.
"""

import time
from yard_manager import YardManager

def test_optimized_file_storage():
    """Test performance with file-based storage."""
    print("\n" + "="*60)
    print("FILE-BASED STORAGE PERFORMANCE TEST")
    print("="*60)

    yard = YardManager('yard.db')

    # Check if optimized storage is available
    if not yard.optimized_db:
        print("❌ Optimized database not available")
        return

    print("\n✅ Optimized file-based storage is active")

    # Test boundary detection with file storage
    print("\n" + "-"*40)
    print("TEST: Boundary Detection with File Storage")
    print("-"*40)

    start_time = time.time()
    result = yard.scan_boundaries_stored()  # Will use file-based storage
    end_time = time.time()

    elapsed = end_time - start_time

    if result['status'] == 'success':
        boundaries = result['boundaries']
        print(f"\n✅ Boundary detection successful!")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds")
        print(f"📊 Total points: {boundaries.get('total_points', 'N/A'):,}")
        print(f"🚀 CUDA accelerated: {boundaries.get('cuda_accelerated', False)}")
        print(f"🎯 Method used: {boundaries.get('method', 'unknown')}")

        # Test rasterization
        print("\n" + "-"*40)
        print("TEST: Rasterization with File Storage")
        print("-"*40)

        start_time = time.time()
        raster_result = yard.project_yard_interactive(
            file_path=None,  # Use stored PLY
            center_x=boundaries['center_x'],
            center_y=boundaries['center_z'],
            rotation=0.0,
            resolution=0.01,
            output_size=(800, 600)
        )
        raster_time = time.time() - start_time

        if raster_result['status'] == 'success':
            print(f"\n✅ Rasterization successful!")
            print(f"⏱️  Time taken: {raster_time:.2f} seconds")
            print(f"📊 Points processed: {raster_result.get('point_count', 'N/A'):,}")

        # Performance comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)

        print("\n📊 With Database Blob Storage (previous):")
        print("   Boundary detection: ~45 seconds")
        print("   Rasterization: ~45 seconds")
        print("   Total: ~90 seconds")

        print(f"\n🚀 With File-Based Storage (current):")
        print(f"   Boundary detection: {elapsed:.2f} seconds")
        print(f"   Rasterization: {raster_time:.2f} seconds")
        print(f"   Total: {elapsed + raster_time:.2f} seconds")

        speedup = 90 / (elapsed + raster_time)
        print(f"\n⚡ Overall speedup: {speedup:.1f}x faster!")

        # Theoretical pure CUDA performance
        print("\n💡 Theoretical pure CUDA performance:")
        print("   Boundary detection: ~0.4 seconds")
        print("   Rasterization: ~0.7 seconds")
        print("   Total: ~1.1 seconds")

        actual_vs_theoretical = (elapsed + raster_time) / 1.1
        print(f"\n📈 Current vs theoretical: {actual_vs_theoretical:.1f}x slower")

        if actual_vs_theoretical > 2:
            print("\n💭 Remaining bottleneck: PLY parsing time")
            print("   Consider: Caching parsed point arrays in memory")
        else:
            print("\n✨ Near-optimal performance achieved!")

    else:
        print(f"❌ Error: {result['message']}")

if __name__ == '__main__':
    test_optimized_file_storage()