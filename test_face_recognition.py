#!/usr/bin/env python3
"""
Test Face Recognition with Frigate Camera
"""

import sys
import os
import time
import requests
from compreface_integration import CompreFaceClient, FrigateIntegration

def test_camera_snapshot():
    """Test getting snapshot from Frigate"""
    print("\n1. Testing Frigate camera snapshot...")

    frigate_url = "http://localhost:5000"
    camera_name = "garage_front"

    try:
        url = f"{frigate_url}/api/{camera_name}/latest.jpg"
        response = requests.get(url)

        if response.status_code == 200:
            # Save snapshot
            snapshot_path = "/tmp/test_snapshot.jpg"
            with open(snapshot_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Successfully captured snapshot from {camera_name}")
            print(f"   Saved to: {snapshot_path}")
            return snapshot_path
        else:
            print(f"❌ Failed to get snapshot: HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error getting snapshot: {e}")
        return None

def test_face_detection(client, image_path):
    """Test face detection on image"""
    print("\n2. Testing face detection...")

    if not image_path or not os.path.exists(image_path):
        print("❌ No image available for detection")
        return

    try:
        result = client.detect_faces(image_path)

        if 'result' in result:
            face_count = len(result['result'])
            print(f"✅ Detection successful: Found {face_count} face(s)")

            for i, face in enumerate(result['result']):
                box = face.get('box', {})
                prob = box.get('probability', 0)
                print(f"   Face {i+1}: Confidence {prob:.2%}")
        else:
            print("⚠️  No faces detected in the image")

    except Exception as e:
        print(f"❌ Detection error: {e}")

def add_sample_subject(client, name, image_path):
    """Add a sample subject for testing"""
    print(f"\n3. Adding subject '{name}' to database...")

    if not image_path or not os.path.exists(image_path):
        print("❌ No image available for adding subject")
        return False

    try:
        result = client.add_subject(name, image_path)

        if result.get('success'):
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ {result.get('error', 'Failed to add subject')}")
            return False

    except Exception as e:
        print(f"❌ Error adding subject: {e}")
        return False

def test_recognition(client, image_path):
    """Test face recognition"""
    print("\n4. Testing face recognition...")

    if not image_path or not os.path.exists(image_path):
        print("❌ No image available for recognition")
        return

    try:
        result = client.recognize_faces(image_path)

        if 'result' in result and result['result']:
            print(f"✅ Recognition successful")

            for face in result['result']:
                subjects = face.get('subjects', [])
                if subjects:
                    for subj in subjects:
                        name = subj.get('subject', 'Unknown')
                        similarity = subj.get('similarity', 0)
                        print(f"   Recognized: {name} (similarity: {similarity:.2%})")
                else:
                    print("   Unknown person detected")
        else:
            print("⚠️  No faces found for recognition")

    except Exception as e:
        print(f"❌ Recognition error: {e}")

def main():
    print("="*60)
    print("TODDLER TRACKER - Face Recognition Test")
    print("="*60)

    # Initialize CompreFace client
    api_key = "9af55064-53f5-4ccd-a43e-b864ea401de2"
    client = CompreFaceClient(api_key=api_key)

    # Test connection
    print("\nChecking CompreFace connection...")
    subjects = client.list_subjects()
    if isinstance(subjects, list):
        print(f"✅ Connected to CompreFace")
        print(f"   Current subjects in database: {len(subjects)}")
        if subjects:
            print(f"   Subjects: {', '.join(subjects)}")
    else:
        print("❌ Failed to connect to CompreFace")
        return

    # Test camera snapshot
    snapshot_path = test_camera_snapshot()

    if snapshot_path:
        # Test face detection
        test_face_detection(client, snapshot_path)

        # Optional: Add a test subject (uncomment to use)
        # print("\nWould you like to add this face as a known person? (y/n)")
        # if input().lower() == 'y':
        #     name = input("Enter person's name: ")
        #     add_sample_subject(client, name, snapshot_path)

        # Test recognition
        if subjects:
            test_recognition(client, snapshot_path)

    print("\n" + "="*60)
    print("Test Complete!")
    print("\nNext steps:")
    print("1. Add known people to the database using add_subject()")
    print("2. Use monitor_camera() for continuous monitoring")
    print("3. Integrate with Home Assistant for automation")
    print("="*60)

if __name__ == "__main__":
    main()