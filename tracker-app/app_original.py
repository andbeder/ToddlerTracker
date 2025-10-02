from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import yaml
import os
import subprocess
import tempfile
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
import requests
import json
import sqlite3
from datetime import datetime, timedelta
import threading
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def parse_rtsp_url(rtsp_url: str) -> Dict[str, str]:
    """Parse RTSP URL to extract components."""
    if not rtsp_url.startswith('rtsp://'):
        return {}

    try:
        # Remove rtsp:// prefix
        url_part = rtsp_url[7:]

        # Split credentials and rest
        if '@' in url_part:
            creds, rest = url_part.split('@', 1)
            if ':' in creds:
                username, password = creds.split(':', 1)
            else:
                username, password = creds, ''
        else:
            username, password = '', ''
            rest = url_part

        # Split IP/port and path
        if '/' in rest:
            ip_port, path = rest.split('/', 1)
            path = '/' + path
        else:
            ip_port, path = rest, '/stream'

        # Split IP and port
        if ':' in ip_port:
            ip_address, port = ip_port.rsplit(':', 1)
        else:
            ip_address, port = ip_port, '554'

        return {
            'username': username,
            'password': password,
            'ip_address': ip_address,
            'port': port,
            'stream_path': path
        }
    except Exception:
        return {}

# Make parse_rtsp_url available in templates
app.jinja_env.globals['parse_rtsp_url'] = parse_rtsp_url

FRIGATE_CONFIG_PATH = '../frigate/config/config.yaml'
CONFIG_FILE_PATH = 'config.yaml'  # Local config for CompreFace settings
MATCHES_DB_PATH = 'matches.db'  # SQLite database for storing matches
THRESHOLDS_CONFIG_PATH = 'thresholds.json'  # JSON config for subject thresholds
DETECTION_CONFIG_PATH = 'detection_settings.json'  # JSON config for detection settings

# Initialize matches database
def init_matches_db():
    """Initialize the matches database"""
    conn = sqlite3.connect(MATCHES_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            camera TEXT,
            face_id TEXT,
            image_data BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_match_to_db(subject: str, confidence: float, camera: str = None, face_id: str = None, image_data: bytes = None):
    """Add a face recognition match to the database"""
    conn = sqlite3.connect(MATCHES_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO matches (subject, confidence, timestamp, camera, face_id, image_data)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (subject, confidence, datetime.now().isoformat(), camera, face_id, image_data))
    conn.commit()
    conn.close()

def get_matches_from_db(limit: int = 100) -> List[Dict]:
    """Get recent matches from the database"""
    conn = sqlite3.connect(MATCHES_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, subject, confidence, timestamp, camera, face_id, created_at
        FROM matches
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))

    matches = []
    for row in cursor.fetchall():
        matches.append({
            'id': row[0],
            'subject': row[1],
            'confidence': row[2],
            'timestamp': row[3],
            'camera': row[4],
            'face_id': row[5],
            'created_at': row[6]
        })

    conn.close()
    return matches

def clear_matches_db():
    """Clear all matches from the database"""
    conn = sqlite3.connect(MATCHES_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM matches')
    conn.commit()
    conn.close()

def create_thumbnail(image_data: bytes, size: tuple = (150, 150)) -> bytes:
    """Create a thumbnail from image data."""
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_data))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create thumbnail while maintaining aspect ratio
        image.thumbnail(size, Image.Resampling.LANCZOS)

        # Save as JPEG to bytes
        output = BytesIO()
        image.save(output, format='JPEG', quality=85, optimize=True)
        output.seek(0)

        return output.getvalue()
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return None

# Initialize the database on startup
init_matches_db()

# Global variable for detection thread control
detection_thread = None
detection_thread_running = False

def run_background_detection():
    """Background thread for continuous face detection"""
    global detection_thread_running

    print("Background detection thread started")

    while detection_thread_running:
        interval = 10  # Default interval

        try:
            settings = load_detection_settings()

            if not settings.get("enabled", False):
                # Sleep longer when disabled to reduce CPU usage
                threading.Event().wait(5)
                continue

            # Get interval from settings
            interval = settings.get("scan_interval", 10)

            # Perform detection
            new_matches = perform_detection_scan()

            if new_matches > 0:
                print(f"Background detection found {new_matches} new matches")

        except Exception as e:
            print(f"Error in background detection: {e}")

        # Wait for the configured interval
        threading.Event().wait(interval)

    print("Background detection thread stopped")

def perform_detection_scan() -> int:
    """Perform a single detection scan on all cameras"""
    try:
        settings = load_detection_settings()
        new_matches = 0

        # Get CompreFace client
        client = get_compreface_client()
        if not client:
            return 0

        # Get list of cameras from Frigate config
        config = load_config()
        cameras = config.get('cameras', {})

        for camera_name in cameras.keys():
            try:
                # Get latest snapshot from Frigate
                frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"
                response = requests.get(frigate_url, timeout=5)

                if response.status_code == 200:
                    image_data = response.content

                    # Run face recognition
                    matches = client.recognize_face(image_data)

                    for match in matches:
                        subject = match['subject']
                        confidence = match['confidence']

                        # Check thresholds
                        global_min = settings.get("global_min_confidence", 50)
                        subject_threshold = get_subject_threshold(subject)

                        if confidence >= global_min and confidence >= subject_threshold:
                            # Create thumbnail from the camera snapshot
                            thumbnail_data = create_thumbnail(image_data)

                            add_match_to_db(
                                subject=subject,
                                confidence=confidence,
                                camera=camera_name,
                                image_data=thumbnail_data  # Store the thumbnail instead
                            )
                            new_matches += 1

            except Exception as e:
                print(f"Error processing camera {camera_name} in background: {str(e)}")
                continue

        return new_matches
    except Exception as e:
        print(f"Error in detection scan: {e}")
        return 0

def start_background_detection():
    """Start the background detection thread"""
    global detection_thread, detection_thread_running

    if detection_thread is not None and detection_thread.is_alive():
        return False  # Already running

    detection_thread_running = True
    detection_thread = threading.Thread(target=run_background_detection, daemon=True)
    detection_thread.start()
    return True

def stop_background_detection():
    """Stop the background detection thread"""
    global detection_thread_running
    detection_thread_running = False

# Background detection will be started in main section after all functions are loaded

def load_thresholds() -> Dict[str, float]:
    """Load subject thresholds from JSON config file"""
    try:
        if os.path.exists(THRESHOLDS_CONFIG_PATH):
            with open(THRESHOLDS_CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading thresholds: {e}")
        return {}

def save_thresholds(thresholds: Dict[str, float]) -> bool:
    """Save subject thresholds to JSON config file"""
    try:
        with open(THRESHOLDS_CONFIG_PATH, 'w') as f:
            json.dump(thresholds, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving thresholds: {e}")
        return False

def get_subject_threshold(subject: str) -> float:
    """Get threshold for a specific subject (default 75%)"""
    thresholds = load_thresholds()
    return thresholds.get(subject, 75.0)

def set_subject_threshold(subject: str, threshold: float) -> bool:
    """Set threshold for a specific subject"""
    thresholds = load_thresholds()
    thresholds[subject] = threshold
    return save_thresholds(thresholds)

def get_all_thresholds() -> Dict[str, float]:
    """Get all subject thresholds with defaults for known subjects"""
    thresholds = load_thresholds()

    # Get list of known subjects from CompreFace
    client = get_compreface_client()
    if client:
        subjects = client.list_subjects()
        for subject in subjects:
            if subject not in thresholds:
                thresholds[subject] = 75.0  # Default threshold

    return thresholds

def load_detection_settings() -> Dict[str, Any]:
    """Load detection settings from JSON config file"""
    try:
        if os.path.exists(DETECTION_CONFIG_PATH):
            with open(DETECTION_CONFIG_PATH, 'r') as f:
                return json.load(f)
        return get_default_detection_settings()
    except Exception as e:
        print(f"Error loading detection settings: {e}")
        return get_default_detection_settings()

def get_default_detection_settings() -> Dict[str, Any]:
    """Get default detection settings"""
    return {
        "enabled": False,
        "scan_interval": 10,  # seconds
        "global_min_confidence": 50,  # percentage
        "max_matches_per_hour": 100,  # rate limiting
        "cameras_enabled": True,  # whether to scan cameras
        "last_modified": datetime.now().isoformat()
    }

def save_detection_settings(settings: Dict[str, Any]) -> bool:
    """Save detection settings to JSON config file"""
    try:
        settings["last_modified"] = datetime.now().isoformat()
        with open(DETECTION_CONFIG_PATH, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving detection settings: {e}")
        return False

def update_detection_setting(key: str, value: Any) -> bool:
    """Update a specific detection setting"""
    settings = load_detection_settings()
    settings[key] = value
    return save_detection_settings(settings)

class CompreFaceClient:
    """Client for interacting with CompreFace API"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}

    def test_connection(self) -> tuple[bool, List[str]]:
        """Test connection and return (success, subjects_list)"""
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
        """Add a new subject with training image"""
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
        """Delete a subject from the database"""
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
        """Get list of all subjects"""
        try:
            url = f"{self.base_url}/api/v1/recognition/subjects"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json().get('subjects', [])
            return []
        except Exception:
            return []

    def get_subject_faces(self, subject_name: str) -> List[Dict]:
        """Get all faces for a specific subject"""
        try:
            url = f"{self.base_url}/api/v1/recognition/faces"
            params = {'subject': subject_name}
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json().get('faces', [])
            return []
        except Exception:
            return []

    def get_face_image(self, face_id: str) -> bytes:
        """Get the actual image data for a face ID"""
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
        """Delete a specific face by ID"""
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
        """Recognize faces in an image and return matches with confidence scores"""
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

def load_config() -> Dict[str, Any]:
    """Load the Frigate configuration from YAML file."""
    try:
        with open(FRIGATE_CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        flash('Frigate config file not found!', 'error')
        return {}
    except yaml.YAMLError as e:
        flash(f'Error parsing YAML: {e}', 'error')
        return {}

def load_app_config() -> Dict[str, Any]:
    """Load app configuration including CompreFace settings"""
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r') as file:
                return yaml.safe_load(file) or {}
        return {}
    except Exception:
        return {}

def save_app_config(config: Dict[str, Any]) -> bool:
    """Save app configuration"""
    try:
        with open(CONFIG_FILE_PATH, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        return True
    except Exception:
        return False

def get_compreface_client() -> Optional[CompreFaceClient]:
    """Get configured CompreFace client"""
    config = load_app_config()
    api_url = config.get('compreface_url')
    api_key = config.get('compreface_api_key')

    if api_url and api_key:
        return CompreFaceClient(api_url, api_key)
    return None

def save_config(config: Dict[str, Any]) -> bool:
    """Save the configuration to YAML file."""
    try:
        with open(FRIGATE_CONFIG_PATH, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        flash(f'Error saving config: {e}', 'error')
        return False

@app.route('/')
def index():
    """Main page showing all cameras."""
    config = load_config()
    cameras = config.get('cameras') or {}
    return render_template('index.html', cameras=cameras)

@app.route('/add_camera', methods=['GET', 'POST'])
def add_camera():
    """Add a new camera configuration."""
    if request.method == 'POST':
        camera_name = request.form.get('camera_name')
        ip_address = request.form.get('ip_address')
        username = request.form.get('username')
        password = request.form.get('password')
        port = request.form.get('port', '554')
        stream_path = request.form.get('stream_path', '/stream')

        if not all([camera_name, ip_address, username, password]):
            flash('All fields except port and stream path are required!', 'error')
            return render_template('add_camera.html')

        config = load_config()
        if not config:
            return redirect(url_for('index'))

        # Ensure cameras section exists and is a dict
        if 'cameras' not in config or config['cameras'] is None:
            config['cameras'] = {}

        # Create camera configuration
        rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}{stream_path}"
        camera_config = {
            'ffmpeg': {
                'inputs': [
                    {
                        'path': rtsp_url,
                        'roles': ['detect']
                    }
                ]
            },
            'detect': {
                'enabled': True
            }
        }

        config['cameras'][camera_name] = camera_config

        if save_config(config):
            flash(f'Camera "{camera_name}" added successfully!', 'success')
            return redirect(url_for('index'))

    return render_template('add_camera.html')

@app.route('/edit_camera/<camera_name>', methods=['GET', 'POST'])
def edit_camera(camera_name):
    """Edit an existing camera configuration."""
    config = load_config()
    if not config or 'cameras' not in config or config['cameras'] is None or camera_name not in config['cameras']:
        flash('Camera not found!', 'error')
        return redirect(url_for('index'))

    camera = config['cameras'][camera_name]

    if request.method == 'POST':
        new_name = request.form.get('camera_name')
        ip_address = request.form.get('ip_address')
        username = request.form.get('username')
        password = request.form.get('password')
        port = request.form.get('port', '554')
        stream_path = request.form.get('stream_path', '/stream')

        if not all([new_name, ip_address, username, password]):
            flash('All fields except port and stream path are required!', 'error')
            return render_template('edit_camera.html', camera_name=camera_name, camera=camera)

        # Create new camera configuration
        rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}{stream_path}"
        new_camera_config = {
            'ffmpeg': {
                'inputs': [
                    {
                        'path': rtsp_url,
                        'roles': ['detect']
                    }
                ]
            },
            'detect': {
                'enabled': camera.get('detect', {}).get('enabled', True)
            }
        }

        # Remove old camera if name changed
        if new_name != camera_name:
            del config['cameras'][camera_name]

        config['cameras'][new_name] = new_camera_config

        if save_config(config):
            flash(f'Camera "{new_name}" updated successfully!', 'success')
            return redirect(url_for('index'))

    # Extract current values for the form
    rtsp_path = camera.get('ffmpeg', {}).get('inputs', [{}])[0].get('path', '')
    current_values = parse_rtsp_url(rtsp_path)

    return render_template('edit_camera.html', camera_name=camera_name, camera=camera, current_values=current_values)

@app.route('/delete_camera/<camera_name>', methods=['POST'])
def delete_camera(camera_name):
    """Delete a camera configuration."""
    config = load_config()
    if not config or 'cameras' not in config or config['cameras'] is None or camera_name not in config['cameras']:
        flash('Camera not found!', 'error')
        return redirect(url_for('index'))

    del config['cameras'][camera_name]

    if save_config(config):
        flash(f'Camera "{camera_name}" deleted successfully!', 'success')

    return redirect(url_for('index'))

@app.route('/toggle_camera/<camera_name>', methods=['POST'])
def toggle_camera(camera_name):
    """Toggle camera detection on/off."""
    config = load_config()
    if not config or 'cameras' not in config or config['cameras'] is None or camera_name not in config['cameras']:
        return jsonify({'error': 'Camera not found'}), 404

    camera = config['cameras'][camera_name]
    current_state = camera.get('detect', {}).get('enabled', True)

    if 'detect' not in camera:
        camera['detect'] = {}
    camera['detect']['enabled'] = not current_state

    if save_config(config):
        return jsonify({'enabled': camera['detect']['enabled']})
    else:
        return jsonify({'error': 'Failed to save config'}), 500

@app.route('/save_config', methods=['POST'])
def save_config_route():
    """Save current configuration to Frigate config file."""
    config = load_config()
    if not config:
        return jsonify({'error': 'Failed to load current config'}), 500

    if save_config(config):
        flash('Configuration saved to Frigate successfully!', 'success')
        return jsonify({'success': True, 'message': 'Configuration saved successfully'})
    else:
        return jsonify({'error': 'Failed to save configuration'}), 500

@app.route('/restart_services', methods=['POST'])
def restart_services():
    """Restart Frigate services."""
    try:
        # Try multiple restart methods in order of preference
        restart_commands = [
            # Try sudo with docker compose
            ['sudo', 'docker', 'compose', '-f', '../docker-compose.yml', 'restart', 'frigate'],
            # Try sudo with docker restart
            ['sudo', 'docker', 'restart', 'frigate'],
            # Try without sudo (in case user is in docker group)
            ['docker', 'compose', '-f', '../docker-compose.yml', 'restart', 'frigate'],
            ['docker', 'restart', 'frigate'],
            # Try systemctl if running as service
            ['sudo', 'systemctl', 'restart', 'frigate'],
        ]

        for cmd in restart_commands:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    method = ' '.join(cmd[:3])
                    flash(f'Frigate services restarted successfully using {method}!', 'success')
                    return jsonify({'success': True, 'message': f'Frigate services restarted successfully using {method}'})

            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        # If all methods failed, return the last error
        return jsonify({'error': 'Failed to restart Frigate. Please check Docker permissions or try running: sudo usermod -aG docker $USER'}), 500

    except Exception as e:
        return jsonify({'error': f'Error restarting services: {str(e)}'}), 500

@app.route('/images')
def images():
    """Face recognition configuration page."""
    config = load_app_config()
    return render_template('images.html', config=config)

@app.route('/save_compreface_config', methods=['POST'])
def save_compreface_config():
    """Save CompreFace API configuration."""
    api_url = request.form.get('api_url')
    api_key = request.form.get('api_key')

    if not api_url or not api_key:
        return jsonify({'error': 'API URL and Key are required'}), 400

    config = load_app_config()
    config['compreface_url'] = api_url
    config['compreface_api_key'] = api_key

    if save_app_config(config):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to save configuration'}), 500

@app.route('/test_compreface_connection', methods=['POST'])
def test_compreface_connection():
    """Test connection to CompreFace API."""
    data = request.json
    api_url = data.get('api_url')
    api_key = data.get('api_key')

    if not api_url or not api_key:
        return jsonify({'error': 'API URL and Key are required'}), 400

    client = CompreFaceClient(api_url, api_key)
    success, subjects = client.test_connection()

    if success:
        return jsonify({'success': True, 'subject_count': len(subjects)})
    else:
        return jsonify({'error': 'Failed to connect to CompreFace'}), 500

@app.route('/upload_training_images', methods=['POST'])
def upload_training_images():
    """Upload and train face recognition images."""
    try:
        subject_name = request.form.get('subject_name')
        images = request.files.getlist('images')

        if not subject_name or not images:
            return jsonify({'error': 'Subject name and images are required'}), 400

        client = get_compreface_client()
        if not client:
            return jsonify({'error': 'CompreFace not configured'}), 500

        images_processed = 0
        errors = []

        for image in images:
            if image and image.filename:
                try:
                    image_data = image.read()
                    result = client.add_subject(subject_name, image_data)

                    if result.get('success'):
                        images_processed += 1
                    else:
                        errors.append(result.get('error', 'Unknown error'))
                except Exception as e:
                    errors.append(f"Error processing {image.filename}: {str(e)}")

        if images_processed > 0:
            return jsonify({
                'success': True,
                'images_processed': images_processed,
                'errors': errors if errors else None
            })
        else:
            return jsonify({'error': 'Failed to process any images', 'details': errors}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get_subjects')
def get_subjects():
    """Get list of trained subjects."""
    client = get_compreface_client()
    if not client:
        return jsonify({'error': 'CompreFace not configured. Please configure API settings first.'}), 200

    subjects = client.list_subjects()
    return jsonify({'subjects': subjects})

@app.route('/delete_subject', methods=['POST'])
def delete_subject():
    """Delete a subject from face recognition."""
    data = request.json
    subject_name = data.get('subject_name')

    if not subject_name:
        return jsonify({'error': 'Subject name is required'}), 400

    client = get_compreface_client()
    if not client:
        return jsonify({'error': 'CompreFace not configured'}), 500

    result = client.delete_subject(subject_name)

    if result.get('success'):
        return jsonify({'success': True})
    else:
        return jsonify({'error': result.get('error', 'Failed to delete subject')}), 500

@app.route('/subject/<subject_name>')
def subject_detail(subject_name):
    """Show details for a specific subject."""
    return render_template('subject_detail.html', subject_name=subject_name)

@app.route('/get_subject_faces/<subject_name>')
def get_subject_faces(subject_name):
    """Get all faces for a specific subject."""
    client = get_compreface_client()
    if not client:
        return jsonify({'error': 'CompreFace not configured'}), 500

    faces = client.get_subject_faces(subject_name)
    return jsonify({'faces': faces, 'subject': subject_name})

@app.route('/face_image/<face_id>')
def face_image(face_id):
    """Get the actual image for a face ID."""
    # Handle None or 'None' face_id
    if face_id is None or face_id == 'None' or face_id == 'null':
        placeholder_svg = '''
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="100" height="100" fill="#e9ecef"/>
            <text x="50" y="35" text-anchor="middle" font-family="Arial" font-size="10" fill="#6c757d">
                Live
            </text>
            <text x="50" y="50" text-anchor="middle" font-family="Arial" font-size="10" fill="#6c757d">
                Detection
            </text>
            <text x="50" y="75" text-anchor="middle" font-family="Arial" font-size="8" fill="#adb5bd">
                No Stored Image
            </text>
        </svg>
        '''
        return Response(placeholder_svg, mimetype='image/svg+xml')

    client = get_compreface_client()
    if not client:
        return Response('CompreFace not configured', status=500)

    image_data = client.get_face_image(face_id)
    if image_data:
        return Response(image_data, mimetype='image/jpeg')
    else:
        # Return a placeholder if image not found
        placeholder_svg = '''
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="100" height="100" fill="#f0f0f0"/>
            <text x="50" y="50" text-anchor="middle" font-family="Arial" font-size="12" fill="#999">
                No Image
            </text>
        </svg>
        '''
        return Response(placeholder_svg, mimetype='image/svg+xml')

@app.route('/match_image/<int:match_id>')
def match_image(match_id):
    """Get the camera snapshot image for a specific match."""
    try:
        conn = sqlite3.connect(MATCHES_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT image_data FROM matches WHERE id = ?', (match_id,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return Response(result[0], mimetype='image/jpeg')
        else:
            # Return placeholder if no image data
            placeholder_svg = '''
            <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" fill="#e9ecef"/>
                <text x="50" y="35" text-anchor="middle" font-family="Arial" font-size="10" fill="#6c757d">
                    No
                </text>
                <text x="50" y="50" text-anchor="middle" font-family="Arial" font-size="10" fill="#6c757d">
                    Snapshot
                </text>
                <text x="50" y="75" text-anchor="middle" font-family="Arial" font-size="8" fill="#adb5bd">
                    Available
                </text>
            </svg>
            '''
            return Response(placeholder_svg, mimetype='image/svg+xml')
    except Exception as e:
        return Response(f'Error: {str(e)}', status=500)

@app.route('/delete_face', methods=['POST'])
def delete_face():
    """Delete a specific face image."""
    data = request.json
    face_id = data.get('face_id')

    if not face_id:
        return jsonify({'error': 'Face ID is required'}), 400

    client = get_compreface_client()
    if not client:
        return jsonify({'error': 'CompreFace not configured'}), 500

    result = client.delete_face(face_id)

    if result.get('success'):
        return jsonify({'success': True})
    else:
        return jsonify({'error': result.get('error', 'Failed to delete face')}), 500

@app.route('/add_faces_to_subject', methods=['POST'])
def add_faces_to_subject():
    """Add more faces to an existing subject."""
    try:
        subject_name = request.form.get('subject_name')
        images = request.files.getlist('images')

        if not subject_name or not images:
            return jsonify({'error': 'Subject name and images are required'}), 400

        client = get_compreface_client()
        if not client:
            return jsonify({'error': 'CompreFace not configured'}), 500

        images_processed = 0
        errors = []
        face_ids = []

        for image in images:
            if image and image.filename:
                try:
                    image_data = image.read()
                    result = client.add_subject(subject_name, image_data)

                    if result.get('success'):
                        images_processed += 1
                        if result.get('face_id'):
                            face_ids.append(result.get('face_id'))
                    else:
                        errors.append(result.get('error', 'Unknown error'))
                except Exception as e:
                    errors.append(f"Error processing {image.filename}: {str(e)}")

        if images_processed > 0:
            return jsonify({
                'success': True,
                'images_processed': images_processed,
                'face_ids': face_ids,
                'errors': errors if errors else None
            })
        else:
            return jsonify({'error': 'Failed to process any images', 'details': errors}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/matches')
def matches():
    """Face recognition matches page."""
    return render_template('matches.html')

@app.route('/get_matches')
def get_matches():
    """Get recent face recognition matches."""
    try:
        matches = get_matches_from_db()
        return jsonify({'matches': matches})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_match', methods=['POST'])
def add_match():
    """Add a new face recognition match."""
    try:
        data = request.json
        subject = data.get('subject')
        confidence = data.get('confidence')
        camera = data.get('camera')
        face_id = data.get('face_id')

        if not subject or confidence is None:
            return jsonify({'error': 'Subject and confidence are required'}), 400

        add_match_to_db(subject, confidence, camera, face_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_matches', methods=['POST'])
def clear_matches():
    """Clear all face recognition matches."""
    try:
        clear_matches_db()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trigger_detection', methods=['POST'])
def trigger_detection():
    """Trigger face detection on camera feeds."""
    try:
        # This would integrate with Frigate to get latest snapshots
        # and run them through CompreFace for recognition
        new_matches = 0

        # Get CompreFace client
        client = get_compreface_client()
        if not client:
            return jsonify({'error': 'CompreFace not configured', 'new_matches': 0})

        # Get list of cameras from Frigate config
        config = load_config()
        cameras = config.get('cameras', {})

        for camera_name in cameras.keys():
            try:
                # Get latest snapshot from Frigate
                frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"
                response = requests.get(frigate_url, timeout=5)

                if response.status_code == 200:
                    image_data = response.content

                    # Run face recognition
                    matches = client.recognize_face(image_data)

                    for match in matches:
                        subject = match['subject']
                        confidence = match['confidence']

                        # Check both global minimum and subject-specific threshold
                        subject_threshold = get_subject_threshold(subject)
                        if confidence >= 50 and confidence >= subject_threshold:
                            # Create thumbnail from the camera snapshot
                            thumbnail_data = create_thumbnail(image_data)

                            add_match_to_db(
                                subject=subject,
                                confidence=confidence,
                                camera=camera_name,
                                image_data=thumbnail_data
                            )
                            new_matches += 1

            except Exception as e:
                print(f"Error processing camera {camera_name}: {str(e)}")
                continue

        return jsonify({'success': True, 'new_matches': new_matches})
    except Exception as e:
        return jsonify({'error': str(e), 'new_matches': 0}), 500

@app.route('/get_thresholds')
def get_thresholds():
    """Get all subject thresholds."""
    try:
        thresholds = get_all_thresholds()
        return jsonify({'thresholds': thresholds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Set threshold for a specific subject."""
    try:
        data = request.json
        subject = data.get('subject')
        threshold = data.get('threshold')

        if not subject or threshold is None:
            return jsonify({'error': 'Subject and threshold are required'}), 400

        # Validate threshold range
        if not 0 <= threshold <= 100:
            return jsonify({'error': 'Threshold must be between 0 and 100'}), 400

        success = set_subject_threshold(subject, threshold)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Failed to save threshold'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_threshold/<subject>')
def get_threshold(subject):
    """Get threshold for a specific subject."""
    try:
        threshold = get_subject_threshold(subject)
        return jsonify({'subject': subject, 'threshold': threshold})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_detection_settings')
def get_detection_settings():
    """Get current detection settings."""
    try:
        settings = load_detection_settings()
        return jsonify({'settings': settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_detection_settings', methods=['POST'])
def update_detection_settings():
    """Update detection settings."""
    try:
        data = request.json
        settings = load_detection_settings()

        # Update provided settings
        for key, value in data.items():
            if key in ['enabled', 'scan_interval', 'global_min_confidence', 'max_matches_per_hour', 'cameras_enabled']:
                settings[key] = value

        # Validate settings
        if 'scan_interval' in settings and not (1 <= settings['scan_interval'] <= 300):
            return jsonify({'error': 'Scan interval must be between 1 and 300 seconds'}), 400

        if 'global_min_confidence' in settings and not (0 <= settings['global_min_confidence'] <= 100):
            return jsonify({'error': 'Global min confidence must be between 0 and 100'}), 400

        success = save_detection_settings(settings)
        if success:
            return jsonify({'success': True, 'settings': settings})
        else:
            return jsonify({'error': 'Failed to save settings'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection enabled/disabled."""
    try:
        data = request.json
        enabled = data.get('enabled', False)

        success = update_detection_setting('enabled', enabled)
        if success:
            status = "enabled" if enabled else "disabled"
            return jsonify({'success': True, 'enabled': enabled, 'message': f'Detection {status}'})
        else:
            return jsonify({'error': 'Failed to update detection status'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detection_status')
def detection_status():
    """Get current detection status and stats."""
    try:
        settings = load_detection_settings()

        # Get recent matches count
        matches = get_matches_from_db(limit=50)
        recent_matches = len([m for m in matches if
                            datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(hours=1)])

        status = {
            'enabled': settings.get('enabled', False),
            'scan_interval': settings.get('scan_interval', 10),
            'global_min_confidence': settings.get('global_min_confidence', 50),
            'thread_running': detection_thread_running,
            'recent_matches_last_hour': recent_matches,
            'last_modified': settings.get('last_modified', 'Never')
        }

        return jsonify({'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera_thumbnail/<camera_name>')
def camera_thumbnail(camera_name):
    """Get a thumbnail image from Frigate's latest snapshot."""
    try:
        # First, try to get snapshot from Frigate API
        frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"

        try:
            response = requests.get(frigate_url, timeout=5)
            if response.status_code == 200:
                return Response(response.content, mimetype='image/jpeg')
        except Exception as e:
            print(f"Failed to get snapshot from Frigate for {camera_name}: {str(e)}")

        # If Frigate snapshot fails, return a placeholder
        # Create a simple placeholder image
        placeholder_svg = f'''
        <svg width="320" height="240" xmlns="http://www.w3.org/2000/svg">
            <rect width="320" height="240" fill="#f0f0f0"/>
            <text x="160" y="120" text-anchor="middle" font-family="Arial" font-size="16" fill="#999">
                {camera_name}
            </text>
            <text x="160" y="140" text-anchor="middle" font-family="Arial" font-size="12" fill="#999">
                No snapshot available
            </text>
        </svg>
        '''
        return Response(placeholder_svg, mimetype='image/svg+xml')

    except Exception as e:
        print(f"Camera thumbnail exception for {camera_name}: {str(e)}")
        return Response(f'Error: {str(e)}', status=500)

@app.route('/health_check')
def system_health_check():
    """Get health status of all system components."""
    health_status = {
        'frigate': check_frigate_health(),
        'compreface': check_compreface_health(),
        'mqtt': check_mqtt_health(),
        'homeassistant': check_homeassistant_health(),
        'live_detection': check_live_detection_health()
    }

    # Overall system status
    all_healthy = all(status['status'] == 'healthy' for status in health_status.values())

    return jsonify({
        'overall_status': 'healthy' if all_healthy else 'degraded',
        'components': health_status,
        'timestamp': datetime.now().isoformat()
    })

def check_frigate_health():
    """Check if Frigate is responding."""
    try:
        response = requests.get('http://localhost:5000/api/config', timeout=5)
        if response.status_code == 200:
            return {'status': 'healthy', 'message': 'Frigate API responding'}
        else:
            return {'status': 'unhealthy', 'message': f'Frigate API returned {response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'status': 'unhealthy', 'message': f'Frigate connection error: {str(e)}'}

def check_compreface_health():
    """Check if CompreFace is responding."""
    try:
        client = get_compreface_client()
        if not client:
            return {'status': 'unhealthy', 'message': 'CompreFace not configured'}

        # Try to list subjects as a health check
        subjects = client.list_subjects()
        if subjects is not None:
            return {'status': 'healthy', 'message': f'CompreFace responding - {len(subjects)} subjects'}
        else:
            return {'status': 'unhealthy', 'message': 'CompreFace returned null response'}
    except Exception as e:
        return {'status': 'unhealthy', 'message': f'CompreFace error: {str(e)}'}

def check_mqtt_health():
    """Check MQTT broker status."""
    try:
        import paho.mqtt.client as mqtt

        # Create a test client
        client = mqtt.Client()

        # Try to connect to MQTT broker
        result = client.connect('localhost', 1883, 5)
        if result == 0:
            client.disconnect()
            return {'status': 'healthy', 'message': 'MQTT broker responding'}
        else:
            return {'status': 'unhealthy', 'message': f'MQTT connection failed: {result}'}
    except Exception as e:
        return {'status': 'unhealthy', 'message': f'MQTT error: {str(e)}'}

def check_homeassistant_health():
    """Check Home Assistant status."""
    try:
        # Try to reach Home Assistant API
        response = requests.get('http://localhost:8123/api/', timeout=5)
        if response.status_code == 200:
            return {'status': 'healthy', 'message': 'Home Assistant API responding'}
        elif response.status_code == 401:
            return {'status': 'healthy', 'message': 'Home Assistant responding (auth required)'}
        else:
            return {'status': 'unhealthy', 'message': f'Home Assistant returned {response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'status': 'unhealthy', 'message': f'Home Assistant connection error: {str(e)}'}

def check_live_detection_health():
    """Check live detection system status."""
    try:
        settings = load_detection_settings()
        enabled = settings.get('enabled', False)

        # Check if detection thread is running
        global detection_thread, detection_thread_running
        thread_running = detection_thread is not None and detection_thread.is_alive() and detection_thread_running

        if enabled and thread_running:
            return {'status': 'healthy', 'message': 'Live detection active and running'}
        elif not enabled:
            return {'status': 'disabled', 'message': 'Live detection disabled by user'}
        else:
            return {'status': 'unhealthy', 'message': 'Live detection enabled but thread not running'}
    except Exception as e:
        return {'status': 'unhealthy', 'message': f'Live detection error: {str(e)}'}

if __name__ == '__main__':
    # Start background detection thread after all functions are loaded
    start_background_detection()
    app.run(debug=True, host='0.0.0.0', port=9000)