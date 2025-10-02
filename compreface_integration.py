#!/usr/bin/env python3
"""
Direct CompreFace Integration Example
Provides face recognition capabilities without double-take
"""

import requests
import json
import time
from typing import Dict, List, Optional

class CompreFaceClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize CompreFace client

        Args:
            base_url: CompreFace server URL
            api_key: Recognition service API key (create via web UI)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}

    def detect_faces(self, image_path: str) -> Dict:
        """
        Detect faces in an image

        Args:
            image_path: Path to image file

        Returns:
            Detection results with face locations and confidence
        """
        url = f"{self.base_url}/api/v1/detection/detect"

        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(url, files=files, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Detection failed: {response.status_code}", "message": response.text}

    def recognize_faces(self, image_path: str, limit: int = 0, det_prob_threshold: float = 0.8) -> Dict:
        """
        Recognize faces in an image against known subjects

        Args:
            image_path: Path to image file
            limit: Maximum number of faces to recognize (0 = unlimited)
            det_prob_threshold: Detection confidence threshold

        Returns:
            Recognition results with subject matches
        """
        url = f"{self.base_url}/api/v1/recognition/recognize"
        params = {
            'limit': limit,
            'det_prob_threshold': det_prob_threshold
        }

        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(url, files=files, params=params, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Recognition failed: {response.status_code}", "message": response.text}

    def add_subject(self, subject_name: str, image_path: str) -> Dict:
        """
        Add a new subject to the recognition database

        Args:
            subject_name: Name/identifier for the subject
            image_path: Path to training image

        Returns:
            Result of adding the subject
        """
        # First create the subject
        url = f"{self.base_url}/api/v1/recognition/subjects"
        data = {'subject': subject_name}

        response = requests.post(url, data=data, headers=self.headers)

        if response.status_code not in [200, 201, 409]:  # 409 = already exists
            return {"error": f"Failed to create subject: {response.status_code}", "message": response.text}

        # Then add example image
        url = f"{self.base_url}/api/v1/recognition/subjects/{subject_name}"

        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(url, files=files, headers=self.headers)

        if response.status_code == 201:
            return {"success": True, "message": f"Subject '{subject_name}' added successfully"}
        else:
            return {"error": f"Failed to add image: {response.status_code}", "message": response.text}

    def list_subjects(self) -> List[str]:
        """
        Get list of all subjects in the database

        Returns:
            List of subject names
        """
        url = f"{self.base_url}/api/v1/recognition/subjects"

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json().get('subjects', [])
        else:
            return []

class FrigateIntegration:
    """
    Integration with Frigate NVR for real-time face recognition
    """

    def __init__(self, frigate_url: str = "http://localhost:5000",
                 compreface_client: CompreFaceClient = None):
        self.frigate_url = frigate_url.rstrip('/')
        self.compreface = compreface_client

    def get_latest_snapshot(self, camera_name: str, save_path: str = "/tmp/snapshot.jpg") -> str:
        """
        Get latest snapshot from Frigate camera

        Args:
            camera_name: Name of the camera (e.g., 'garage_front')
            save_path: Where to save the snapshot

        Returns:
            Path to saved snapshot file
        """
        url = f"{self.frigate_url}/api/{camera_name}/latest.jpg"

        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return save_path
        else:
            raise Exception(f"Failed to get snapshot: {response.status_code}")

    def monitor_camera(self, camera_name: str, interval: int = 10):
        """
        Continuously monitor camera for face recognition

        Args:
            camera_name: Camera to monitor
            interval: Check interval in seconds
        """
        if not self.compreface or not self.compreface.api_key:
            print("Error: CompreFace client not configured with API key")
            return

        print(f"Starting face recognition monitoring for camera: {camera_name}")
        print(f"Check interval: {interval} seconds")
        print("Press Ctrl+C to stop")

        try:
            while True:
                try:
                    # Get latest snapshot
                    snapshot_path = self.get_latest_snapshot(camera_name)

                    # Recognize faces
                    result = self.compreface.recognize_faces(snapshot_path)

                    if 'result' in result and result['result']:
                        for face in result['result']:
                            subjects = face.get('subjects', [])
                            if subjects:
                                best_match = subjects[0]
                                confidence = best_match.get('similarity', 0)
                                subject_name = best_match.get('subject', 'Unknown')
                                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                      f"Recognized: {subject_name} (confidence: {confidence:.2f})")
                            else:
                                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                      f"Unknown person detected")

                except Exception as e:
                    print(f"Error during monitoring: {e}")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopping camera monitoring")

def main():
    """
    Example usage of CompreFace integration
    """
    print("CompreFace Direct Integration Example")
    print("="*50)

    # Initialize client with your API key
    api_key = "9af55064-53f5-4ccd-a43e-b864ea401de2"

    client = CompreFaceClient(api_key=api_key)

    # Test connection
    subjects = client.list_subjects()
    if isinstance(subjects, list):
        print(f"Connected to CompreFace. Found {len(subjects)} subjects.")
        if subjects:
            print(f"Subjects: {', '.join(subjects)}")
    else:
        print("Failed to connect to CompreFace. Please check:")
        print("1. CompreFace is running (http://localhost:8000)")
        print("2. You have created a Recognition service via web UI")
        print("3. API key is correct")
        return

    # Example: Add a subject (uncomment and modify)
    # result = client.add_subject("John Doe", "/path/to/johns_photo.jpg")
    # print(f"Add subject result: {result}")

    # Example: Recognize faces in an image
    # result = client.recognize_faces("/path/to/test_image.jpg")
    # print(f"Recognition result: {json.dumps(result, indent=2)}")

    # Example: Monitor Frigate camera
    frigate = FrigateIntegration(compreface_client=client)
    # frigate.monitor_camera("garage_front", interval=30)

if __name__ == "__main__":
    main()