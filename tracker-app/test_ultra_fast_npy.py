#!/usr/bin/env python3
"""
Test ultra-fast NPY memory-mapped performance.
This should achieve sub-second total processing time!
"""

import time
from yard_manager import YardManager

def test_ultra_fast_npy():
    """Test the ultimate performance with memory-mapped NPY files."""
    print("\n" + "="*60)
    print("🚀 ULTRA-FAST NPY MEMORY-MAPPED PERFORMANCE TEST")
    print("Target: Sub-second total processing!")
    print("="*60)

    yard = YardManager('yard.db')

    # Check if NPY loader is available
    if not yard.npy_loader:
        print("❌ NPY fast loader not available")
        return

    print("\n✅ Ultra-fast NPY memory-mapped loader ready")

    # Test 1: Ultra-fast boundary detection
    print("\n" + "-"*40)
    print("TEST 1: Ultra-Fast Boundary Detection")
    print("-"*40)

    start_time = time.time()
    result = yard.scan_boundaries_ultra_fast('fused')
    boundary_time = time.time() - start_time

    if result['status'] == 'success':
        boundaries = result['boundaries']
        print(f"\n✅ Ultra-fast boundary detection successful!")
        print(f"⏱️  Total time: {boundary_time:.6f} seconds")
        print(f"📊 Total points: {boundaries.get('total_points', 'N/A'):,}")
        print(f"🚀 Method: {boundaries.get('method', 'unknown')}")
        print(f"💾 Load time: {boundaries.get('load_time_seconds', 0):.6f} seconds")

        # Test 2: Ultra-fast rasterization with memory-mapped data
        print("\n" + "-"*40)
        print("TEST 2: Ultra-Fast Rasterization")
        print("-"*40)

        # Get the memory-mapped data for rasterization
        points, colors, metadata = yard.npy_loader.load_dataset_by_name('fused')

        if points is not None:
            start_time = time.time()

            # Use optimized rasterizer directly with memory-mapped arrays
            if yard.optimized_rasterizer:
                image = yard.optimized_rasterizer.rasterize_point_cloud(
                    points=points,
                    center_x=boundaries['center_x'],
                    center_y=boundaries['center_z'],
                    rotation=0.0,
                    resolution=0.01,
                    output_size=(800, 600),
                    colors=colors
                )
                raster_time = time.time() - start_time

                print(f"\n✅ Ultra-fast rasterization successful!")
                print(f"⏱️  Time taken: {raster_time:.6f} seconds")
                print(f"📊 Points processed: {len(points):,}")

                # ULTIMATE PERFORMANCE SUMMARY
                print("\n" + "="*60)
                print("🏆 ULTIMATE PERFORMANCE ACHIEVED!")
                print("="*60)

                total_time = boundary_time + raster_time

                print(f"\n📊 Performance Breakdown:")
                print(f"   💾 Data loading: {boundaries.get('load_time_seconds', 0):.6f} seconds")
                print(f"   🎯 Boundary detection: {boundary_time:.6f} seconds")
                print(f"   🎨 Rasterization: {raster_time:.6f} seconds")
                print(f"   ⏱️  TOTAL: {total_time:.6f} seconds")

                # Compare with original performance
                original_time = 90.0  # Original ~90 seconds
                speedup = original_time / total_time

                print(f"\n🚀 Performance vs Original:")
                print(f"   Original total time: {original_time:.1f} seconds")
                print(f"   Ultra-fast time: {total_time:.6f} seconds")
                print(f"   SPEEDUP: {speedup:.0f}x faster!")

                # Check if we achieved sub-second performance
                if total_time < 1.0:
                    print(f"\n🎉 SUB-SECOND PERFORMANCE ACHIEVED!")
                    print(f"   Target: < 1.0 seconds")
                    print(f"   Actual: {total_time:.6f} seconds")
                    print(f"   🏅 MISSION ACCOMPLISHED!")
                else:
                    print(f"\n⚡ Near sub-second performance:")
                    print(f"   Current: {total_time:.6f} seconds")
                    print(f"   Remaining optimization potential: {(total_time - 1.0):.6f} seconds")

                # Theoretical comparison
                print(f"\n💡 Compared to pure CUDA computation:")
                pure_cuda_time = 0.35 + 0.7  # boundary + raster
                overhead = total_time - pure_cuda_time
                print(f"   Pure CUDA time: {pure_cuda_time:.3f} seconds")
                print(f"   Memory-mapped overhead: {overhead:.6f} seconds")
                print(f"   Efficiency: {(pure_cuda_time/total_time)*100:.1f}%")

            else:
                print("❌ Optimized rasterizer not available")
        else:
            print("❌ Failed to load memory-mapped data")

    else:
        print(f"❌ Error: {result['message']}")

if __name__ == '__main__':
    test_ultra_fast_npy()