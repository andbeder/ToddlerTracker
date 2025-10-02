"""
CompreFace API client for face recognition operations.
Handles all interactions with the CompreFace service.
"""

import requests
from typing import Dict, List, Optional, Tuple


class CompreFaceClient:
    """Client for interacting with CompreFace API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}

    def test_connection(self) -> Tuple[bool, List[str]]:
        """Test connection and return (success, subjects_list)."""
        try:
            url = f"{self.base_url}/api/v1/recognition/subjects"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                subjects = response.json().get('subjects', [])
                return True, subjects
            return False, []
        except Exception as e:
            return False, []

    def add_subject(self, subject_name: str, image_data: bytes) -> Dict:
        """Add a new subject with training image."""
        try:
            # Check if subject exists, if not create it
            subjects = self.list_subjects()
            if subject_name not in subjects:
                url = f"{self.base_url}/api/v1/recognition/subjects"
                response = requests.post(url, data={'subject': subject_name}, headers=self.headers)
                if response.status_code not in [200, 201]:
                    return {"error": f"Failed to create subject: {response.text}"}

            # Add the training image
            url = f"{self.base_url}/api/v1/recognition/faces"
            files = {'file': ('image.jpg', image_data, 'image/jpeg')}
            data = {'subject': subject_name}
            response = requests.post(url, files=files, data=data, headers=self.headers)

            if response.status_code in [200, 201]:
                result = {"success": True}
                response_data = response.json()
                if 'image_id' in response_data:
                    result["face_id"] = response_data['image_id']
                print(f"Successfully added face for {subject_name}: {result.get('face_id', 'unknown')}")
                return result
            else:
                error_msg = f"Failed to add face: {response.text}"
                print(f"Error adding face for {subject_name}: {error_msg}")
                return {"error": error_msg}
        except Exception as e:
            return {"error": str(e)}

    def delete_subject(self, subject_name: str) -> Dict:
        """Delete a subject from the database."""
        try:
            url = f"{self.base_url}/api/v1/recognition/subjects/{subject_name}"
            response = requests.delete(url, headers=self.headers)

            if response.status_code in [200, 204]:
                return {"success": True}
            else:
                return {"error": f"Failed to delete subject: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def list_subjects(self) -> List[str]:
        """Get list of all subjects."""
        try:
            url = f"{self.base_url}/api/v1/recognition/subjects"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json().get('subjects', [])
            return []
        except Exception:
            return []

    def get_subject_faces(self, subject_name: str) -> List[Dict]:
        """Get all faces for a specific subject."""
        try:
            url = f"{self.base_url}/api/v1/recognition/faces"
            params = {'subject': subject_name}
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json().get('faces', [])
            return []
        except Exception:
            return []

    def get_face_image(self, face_id: str) -> Optional[bytes]:
        """Get the actual image data for a face ID."""
        try:
            # CompreFace API to download face image
            url = f"{self.base_url}/api/v1/recognition/faces/{face_id}/img"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.content
            return None
        except Exception:
            return None

    def delete_face(self, face_id: str) -> Dict:
        """Delete a specific face by ID."""
        try:
            url = f"{self.base_url}/api/v1/recognition/faces/{face_id}"
            response = requests.delete(url, headers=self.headers)
            if response.status_code in [200, 204]:
                return {"success": True}
            else:
                return {"error": f"Failed to delete face: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def recognize_face(self, image_data: bytes) -> List[Dict]:
        """Recognize faces in an image and return matches with confidence scores."""
        try:
            url = f"{self.base_url}/api/v1/recognition/recognize"
            files = {'file': ('image.jpg', image_data, 'image/jpeg')}
            response = requests.post(url, files=files, headers=self.headers)

            if response.status_code == 200:
                result = response.json()
                matches = []
                for face in result.get('result', []):
                    subjects = face.get('subjects', [])
                    if subjects:
                        # Get the best match
                        best_match = max(subjects, key=lambda x: x.get('similarity', 0))
                        matches.append({
                            'subject': best_match.get('subject'),
                            'confidence': best_match.get('similarity', 0) * 100,  # Convert to percentage
                            'box': face.get('box', {})  # Face bounding box
                        })
                return matches
            else:
                print(f"Recognition error: {response.text}")
                return []
        except Exception as e:
            print(f"Recognition exception: {str(e)}")
            return []