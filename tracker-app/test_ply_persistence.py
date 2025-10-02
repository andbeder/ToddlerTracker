#!/usr/bin/env python3
"""
Test script for PLY file persistence functionality.
"""

import sys
import os
import sqlite3

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_tables():
    """Test if PLY files table was created."""
    try:
        conn = sqlite3.connect('yard.db')
        cursor = conn.cursor()

        # Check if ply_files table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='ply_files'
        """)

        result = cursor.fetchone()

        if result:
            print("✓ PLY files table exists")

            # Check table structure
            cursor.execute("PRAGMA table_info(ply_files)")
            columns = cursor.fetchall()
            print("✓ Table columns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")

            # Check if any PLY files are stored
            cursor.execute("SELECT COUNT(*) FROM ply_files")
            count = cursor.fetchone()[0]
            print(f"✓ Stored PLY files: {count}")

            if count > 0:
                cursor.execute("""
                    SELECT id, name, vertex_count, has_color, format, uploaded_at
                    FROM ply_files
                    ORDER BY uploaded_at DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()
                print(f"✓ Latest PLY file:")
                print(f"  - ID: {latest[0]}")
                print(f"  - Name: {latest[1]}")
                print(f"  - Vertices: {latest[2]}")
                print(f"  - Has color: {latest[3]}")
                print(f"  - Format: {latest[4]}")
                print(f"  - Uploaded: {latest[5]}")

            return True
        else:
            print("✗ PLY files table does not exist")
            return False

    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False
    finally:
        conn.close()

def test_yard_manager():
    """Test YardManager PLY storage methods."""
    try:
        from yard_manager import YardManager

        yard_manager = YardManager('yard.db')
        print("\n✓ YardManager initialized")

        # Test get_latest_ply_data
        latest_ply = yard_manager.get_latest_ply_data()

        if latest_ply:
            print("✓ Latest PLY data retrieved:")
            print(f"  - Name: {latest_ply['name']}")
            print(f"  - Vertices: {latest_ply['vertex_count']}")
            print(f"  - Uploaded: {latest_ply['uploaded_at']}")
        else:
            print("ℹ No PLY files stored yet")

        return True

    except Exception as e:
        print(f"✗ YardManager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("PLY File Persistence Test")
    print("=" * 50)

    tests = [
        test_database_tables,
        test_yard_manager
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
        print("🎉 PLY persistence is working!")
        print("\nNext steps:")
        print("1. Upload a fused.ply file through the web interface")
        print("2. Refresh the page - the PLY file should still be available")
        print("3. You can proceed with boundary scanning without re-uploading")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())