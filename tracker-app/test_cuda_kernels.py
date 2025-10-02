#!/usr/bin/env python3
"""
Direct test of CUDA kernel performance without database overhead.
"""

import numpy as np
import time
import tempfile
import os
from yard_manager import YardManager

def extract_ply_to_file():
    """Extract PLY from database to file for direct testing."""
    print("Extracting PLY from database...")
    import sqlite3
    conn = sqlite3.connect('yard.db')
    cursor = conn.cursor()
    cursor.execute("SELECT file_data FROM ply_files WHERE id = 1")
    result = cursor.fetchone()
    conn.close()

    if result:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp:
            tmp.write(result[0])
            return tmp.name
    return None

def test_direct_cuda_performance():
    """Test CUDA performance without database overhead."""
    print("\n" + "="*60)
    print("DIRECT CUDA KERNEL PERFORMANCE TEST")
    print("="*60)

    # Extract PLY once
    ply_path = extract_ply_to_file()
    if not ply_path:
        print("❌ Failed to extract PLY")
        return

    try:
        yard = YardManager('yard.db')

        # Parse PLY file
        print("\nParsing PLY file...")
        parse_start = time.time()
        ply_data = yard.parser.parse_ply_file(ply_path)
        parse_time = time.time() - parse_start
        points = ply_data['points']
        colors = ply_data.get('colors')
        print(f"✅ PLY parsed in {parse_time:.2f} seconds")
        print(f"📊 Points loaded: {len(points):,}")

        # Test 1: Optimized Boundary Detection (CUDA only)
        print("\n" + "-"*40)
        print("TEST 1: CUDA Boundary Detection")
        print("-"*40)

        if yard.optimized_detector and yard.optimized_detector.cuda_available:
            cuda_start = time.time()
            boundaries = yard.optimized_detector.detect_boundaries(
                points, percentile_min=2.0, percentile_max=98.0, method='statistical'
            )
            cuda_time = time.time() - cuda_start

            print(f"✅ CUDA boundary detection: {cuda_time:.2f} seconds")
            print(f"📊 Method: {boundaries.get('method', 'unknown')}")
            print(f"🚀 Points processed: {boundaries.get('total_points', 0):,}")
            print(f"⚡ Points per second: {len(points)/cuda_time:,.0f}")
        else:
            print("❌ Optimized detector not available")

        # Test 2: Optimized Rasterization (CUDA only)
        print("\n" + "-"*40)
        print("TEST 2: CUDA Rasterization with Spatial Grid")
        print("-"*40)

        if yard.optimized_rasterizer and yard.optimized_rasterizer.cuda_available:
            # Use the boundaries from previous step
            cuda_start = time.time()
            image = yard.optimized_rasterizer.rasterize_point_cloud(
                points=points,
                center_x=boundaries['center_x'],
                center_y=boundaries['center_z'],
                rotation=0.0,
                resolution=0.01,
                output_size=(800, 600),
                colors=colors
            )
            cuda_time = time.time() - cuda_start

            print(f"✅ CUDA rasterization: {cuda_time:.2f} seconds")
            print(f"📐 Output size: 800x600 pixels")
            print(f"⚡ Pixels per second: {(800*600)/cuda_time:,.0f}")
        else:
            print("❌ Optimized rasterizer not available")

        # Test 3: Database overhead measurement
        print("\n" + "-"*40)
        print("TEST 3: Database Overhead Measurement")
        print("-"*40)

        db_start = time.time()
        result = yard.scan_boundaries_stored(ply_id=1)
        db_total_time = time.time() - db_start

        print(f"⏱️  Total time with database: {db_total_time:.2f} seconds")
        print(f"📊 Database overhead: {db_total_time - cuda_time:.2f} seconds")

        # Summary
        print("\n" + "="*60)
        print("PERFORMANCE BREAKDOWN")
        print("="*60)
        print(f"\n📁 File parsing: {parse_time:.2f} seconds")
        print(f"🚀 Pure CUDA computation: ~{cuda_time:.2f} seconds")
        print(f"💾 Database operations: ~{db_total_time - cuda_time:.2f} seconds")
        print(f"⏱️  Total with database: {db_total_time:.2f} seconds")

        percentage_overhead = ((db_total_time - cuda_time) / db_total_time) * 100
        print(f"\n⚠️  Database overhead: {percentage_overhead:.1f}% of total time")
        print(f"💡 Recommendation: Direct file access would be {db_total_time/cuda_time:.1f}x faster")

    finally:
        # Cleanup
        if os.path.exists(ply_path):
            os.unlink(ply_path)
            print("\n✅ Temporary file cleaned up")

if __name__ == '__main__':
    test_direct_cuda_performance()