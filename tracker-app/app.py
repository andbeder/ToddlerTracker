"""
Main Flask application for Toddler Tracker.
Refactored to use modular components for better organization and maintainability.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import subprocess
import os
import json
import requests
import logging
import numpy as np
import time
from datetime import datetime, timedelta

# Import our custom modules
from database import MatchesDatabase
from config_manager import ConfigManager
from detection_service import DetectionService
from hybrid_detection_service import HybridDetectionService
from health_monitor import HealthMonitor
from image_utils import ImageProcessor
from pose_manager import PoseManager
from yard_manager import YardManager
from multi_camera_tracker import TrackingService
from image_converter import get_image_converter

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size for COLMAP files
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Auto-reload templates in development

# Setup logging
logger = logging.getLogger(__name__)

# Configuration constants
FRIGATE_URL = 'http://localhost:5000'

# Initialize components
db = MatchesDatabase('matches.db')
config = ConfigManager()
# Use HybridDetectionService for enhanced identification and position tracking
detection_service = HybridDetectionService(db, config)
health_monitor = HealthMonitor(config, detection_service)
pose_manager = PoseManager('poses.db')
yard_manager = YardManager('yard.db')
tracking_service = TrackingService()

# Make parse_rtsp_url available in templates
app.jinja_env.globals['parse_rtsp_url'] = config.parse_rtsp_url

# Background detection disabled - running in Docker container
# Uncomment to run detection in this process:
detection_service.start_background_detection()


# =============================================================================
# CAMERA MANAGEMENT ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main page - redirect to map view."""
    return redirect(url_for('map_view'))


@app.route('/map')
def map_view():
    """Live map view with toddler tracking."""
    return render_template('map.html')


@app.route('/cameras')
def cameras():
    """Camera configuration page."""
    frigate_config = config.load_frigate_config()
    cameras = frigate_config.get('cameras') or {}
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

        frigate_config = config.load_frigate_config()
        if not frigate_config:
            return redirect(url_for('index'))

        # Ensure cameras section exists and is a dict
        if 'cameras' not in frigate_config or frigate_config['cameras'] is None:
            frigate_config['cameras'] = {}

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

        frigate_config['cameras'][camera_name] = camera_config

        if config.save_frigate_config(frigate_config):
            flash(f'Camera "{camera_name}" added successfully!', 'success')
            return redirect(url_for('index'))

    return render_template('add_camera.html')


@app.route('/edit_camera/<camera_name>', methods=['GET', 'POST'])
def edit_camera(camera_name):
    """Edit an existing camera configuration."""
    frigate_config = config.load_frigate_config()
    if not frigate_config or 'cameras' not in frigate_config or frigate_config['cameras'] is None or camera_name not in frigate_config['cameras']:
        flash('Camera not found!', 'error')
        return redirect(url_for('index'))

    camera = frigate_config['cameras'][camera_name]

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
            del frigate_config['cameras'][camera_name]

        frigate_config['cameras'][new_name] = new_camera_config

        if config.save_frigate_config(frigate_config):
            flash(f'Camera "{new_name}" updated successfully!', 'success')
            return redirect(url_for('index'))

    # Extract current values for the form
    rtsp_path = camera.get('ffmpeg', {}).get('inputs', [{}])[0].get('path', '')
    current_values = config.parse_rtsp_url(rtsp_path)

    return render_template('edit_camera.html', camera_name=camera_name, camera=camera, current_values=current_values)


@app.route('/delete_camera/<camera_name>', methods=['POST'])
def delete_camera(camera_name):
    """Delete a camera configuration."""
    frigate_config = config.load_frigate_config()
    if not frigate_config or 'cameras' not in frigate_config or frigate_config['cameras'] is None or camera_name not in frigate_config['cameras']:
        flash('Camera not found!', 'error')
        return redirect(url_for('index'))

    del frigate_config['cameras'][camera_name]

    if config.save_frigate_config(frigate_config):
        flash(f'Camera "{camera_name}" deleted successfully!', 'success')

    return redirect(url_for('index'))


@app.route('/toggle_camera/<camera_name>', methods=['POST'])
def toggle_camera(camera_name):
    """Toggle camera detection on/off."""
    frigate_config = config.load_frigate_config()
    if not frigate_config or 'cameras' not in frigate_config or frigate_config['cameras'] is None or camera_name not in frigate_config['cameras']:
        return jsonify({'error': 'Camera not found'}), 404

    camera = frigate_config['cameras'][camera_name]
    current_state = camera.get('detect', {}).get('enabled', True)

    if 'detect' not in camera:
        camera['detect'] = {}
    camera['detect']['enabled'] = not current_state

    if config.save_frigate_config(frigate_config):
        return jsonify({'enabled': camera['detect']['enabled']})
    else:
        return jsonify({'error': 'Failed to save config'}), 500


@app.route('/camera_thumbnail/<camera_name>')
def camera_thumbnail(camera_name):
    """Get a thumbnail image from Frigate's latest snapshot."""
    try:
        # First, try to get snapshot from Frigate API
        frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"

        try:
            import requests
            response = requests.get(frigate_url, timeout=5)
            if response.status_code == 200:
                return Response(response.content, mimetype='image/jpeg')
        except Exception as e:
            print(f"Failed to get snapshot from Frigate for {camera_name}: {str(e)}")

        # If Frigate snapshot fails, return a placeholder
        placeholder_svg = ImageProcessor.create_placeholder_svg(
            320, 240, f"{camera_name}\nNo snapshot available"
        )
        return Response(placeholder_svg, mimetype='image/svg+xml')

    except Exception as e:
        print(f"Camera thumbnail exception for {camera_name}: {str(e)}")
        return Response(f'Error: {str(e)}', status=500)


# =============================================================================
# SYSTEM MANAGEMENT ROUTES
# =============================================================================

@app.route('/save_config', methods=['POST'])
def save_config_route():
    """Save current configuration to Frigate config file."""
    frigate_config = config.load_frigate_config()
    if not frigate_config:
        return jsonify({'error': 'Failed to load current config'}), 500

    if config.save_frigate_config(frigate_config):
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


@app.route('/health_check')
def system_health_check():
    """Get health status of all system components."""
    health_data = health_monitor.check_all_components()
    return jsonify(health_data)


# =============================================================================
# FACE RECOGNITION ROUTES
# =============================================================================

@app.route('/images')
def images():
    """Face recognition configuration page."""
    app_config = config.load_app_config()
    return render_template('images.html', config=app_config)


@app.route('/save_compreface_config', methods=['POST'])
def save_compreface_config():
    """Save CompreFace API configuration."""
    api_url = request.form.get('api_url')
    api_key = request.form.get('api_key')

    if not api_url or not api_key:
        return jsonify({'error': 'API URL and Key are required'}), 400

    app_config = config.load_app_config()
    app_config['compreface_url'] = api_url
    app_config['compreface_api_key'] = api_key

    if config.save_app_config(app_config):
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

    from compreface_client import CompreFaceClient
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

        client = config.get_compreface_client()
        if not client:
            return jsonify({'error': 'CompreFace not configured'}), 500

        images_processed = 0
        errors = []

        # Initialize image converter
        converter = get_image_converter()

        for image in images:
            if image and image.filename:
                try:
                    # Read original image data
                    original_data = image.read()

                    # Validate image
                    is_valid, validation_msg = converter.validate_image(original_data)
                    if not is_valid:
                        errors.append(f"{image.filename}: {validation_msg}")
                        continue

                    # Convert to JPEG if needed
                    processed_data, processed_filename, was_converted = converter.process_upload(
                        original_data, image.filename
                    )

                    if was_converted:
                        app.logger.info(f"Converted {image.filename} to JPEG for CompreFace compatibility")

                    # Send to CompreFace
                    result = client.add_subject(subject_name, processed_data)

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
    client = config.get_compreface_client()
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

    client = config.get_compreface_client()
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
    client = config.get_compreface_client()
    if not client:
        return jsonify({'error': 'CompreFace not configured'}), 500

    faces = client.get_subject_faces(subject_name)
    return jsonify({'faces': faces, 'subject': subject_name})


@app.route('/face_image/<face_id>')
def face_image(face_id):
    """Get the actual image for a face ID."""
    # Handle None or 'None' face_id
    if face_id is None or face_id == 'None' or face_id == 'null':
        placeholder_svg = ImageProcessor.create_placeholder_svg(
            100, 100, "Live\nDetection\nNo Stored Image"
        )
        return Response(placeholder_svg, mimetype='image/svg+xml')

    client = config.get_compreface_client()
    if not client:
        return Response('CompreFace not configured', status=500)

    image_data = client.get_face_image(face_id)
    if image_data:
        return Response(image_data, mimetype='image/jpeg')
    else:
        placeholder_svg = ImageProcessor.create_placeholder_svg(100, 100, "No Image")
        return Response(placeholder_svg, mimetype='image/svg+xml')


@app.route('/delete_face', methods=['POST'])
def delete_face():
    """Delete a specific face image."""
    data = request.json
    face_id = data.get('face_id')

    if not face_id:
        return jsonify({'error': 'Face ID is required'}), 400

    client = config.get_compreface_client()
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

        client = config.get_compreface_client()
        if not client:
            return jsonify({'error': 'CompreFace not configured'}), 500

        images_processed = 0
        errors = []
        face_ids = []

        # Initialize image converter
        converter = get_image_converter()

        for image in images:
            if image and image.filename:
                try:
                    # Read original image data
                    original_data = image.read()

                    # Validate image
                    is_valid, validation_msg = converter.validate_image(original_data)
                    if not is_valid:
                        errors.append(f"{image.filename}: {validation_msg}")
                        continue

                    # Convert to JPEG if needed
                    processed_data, processed_filename, was_converted = converter.process_upload(
                        original_data, image.filename
                    )

                    if was_converted:
                        app.logger.info(f"Converted {image.filename} to JPEG for CompreFace compatibility")

                    # Send to CompreFace
                    result = client.add_subject(subject_name, processed_data)

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


# =============================================================================
# MATCH MANAGEMENT ROUTES
# =============================================================================

@app.route('/matches')
def matches():
    """Face recognition matches page."""
    return render_template('matches.html')


@app.route('/get_matches')
def get_matches():
    """Get recent face recognition matches."""
    try:
        matches = db.get_matches()
        return jsonify({'matches': matches})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_last_toddler_position')
def get_last_toddler_position():
    """Get the most recent toddler position on the map."""
    try:
        position = db.get_last_toddler_position()
        if position:
            return jsonify({'status': 'success', 'position': position})
        else:
            return jsonify({'status': 'no_data', 'message': 'No position data available'})
    except Exception as e:
        logger.error(f"Error getting toddler position: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/add_toddler_position', methods=['POST'])
def add_toddler_position():
    """Add a new toddler position record."""
    try:
        data = request.get_json()
        subject = data.get('subject')
        camera = data.get('camera')
        map_x = data.get('map_x')
        map_y = data.get('map_y')
        confidence = data.get('confidence', 0.0)
        timestamp = data.get('timestamp')

        if not all([subject, camera, map_x is not None, map_y is not None]):
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

        position_id = db.add_toddler_position(subject, camera, map_x, map_y, confidence, timestamp)
        return jsonify({'status': 'success', 'position_id': position_id})

    except Exception as e:
        logger.error(f"Error adding toddler position: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/set_active_map', methods=['POST'])
def set_active_map():
    """Set the active yard map for position tracking."""
    try:
        data = request.get_json()
        map_id = data.get('map_id')

        if map_id is None:
            return jsonify({'status': 'error', 'message': 'map_id is required'}), 400

        success = detection_service.set_active_map_id(int(map_id))
        if success:
            return jsonify({'status': 'success', 'map_id': map_id})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to set map ID'}), 500

    except Exception as e:
        logger.error(f"Error setting active map: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_position_tracking_config')
def get_position_tracking_config():
    """Get position tracking configuration and status."""
    try:
        stats = detection_service.get_detection_statistics()

        return jsonify({
            'status': 'success',
            'position_tracking_enabled': stats.get('position_tracking_enabled', False),
            'active_map_id': stats.get('active_map_id'),
            'positions_tracked': stats.get('positions_tracked', 0),
            'hybrid_enabled': stats.get('hybrid_enabled', False)
        })

    except Exception as e:
        logger.error(f"Error getting position tracking config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/clear_projection_cache', methods=['POST'])
def clear_projection_cache():
    """Clear the position tracker projection cache."""
    try:
        success = detection_service.clear_projection_cache()
        if success:
            return jsonify({'status': 'success', 'message': 'Projection cache cleared'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to clear cache'}), 500

    except Exception as e:
        logger.error(f"Error clearing projection cache: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/match_image/<int:match_id>')
def match_image(match_id):
    """Get the camera snapshot image for a specific match."""
    try:
        image_data = db.get_match_image(match_id)
        if image_data:
            return Response(image_data, mimetype='image/jpeg')
        else:
            placeholder_svg = ImageProcessor.create_placeholder_svg(
                100, 100, "No\nSnapshot\nAvailable"
            )
            return Response(placeholder_svg, mimetype='image/svg+xml')
    except Exception as e:
        return Response(f'Error: {str(e)}', status=500)


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

        db.add_match(subject, confidence, camera, face_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear_matches', methods=['POST'])
def clear_matches():
    """Clear all face recognition matches."""
    try:
        db.clear_matches()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# DETECTION CONTROL ROUTES
# =============================================================================

@app.route('/trigger_detection', methods=['POST'])
def trigger_detection():
    """Trigger face detection on camera feeds."""
    try:
        success, new_matches = detection_service.trigger_manual_detection()
        if success:
            return jsonify({'success': True, 'new_matches': new_matches})
        else:
            return jsonify({'error': 'Detection failed', 'new_matches': 0}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'new_matches': 0}), 500


@app.route('/get_thresholds')
def get_thresholds():
    """Get all subject thresholds."""
    try:
        thresholds = config.get_all_thresholds()
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

        success = config.set_subject_threshold(subject, threshold)
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
        threshold = config.get_subject_threshold(subject)
        return jsonify({'subject': subject, 'threshold': threshold})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_detection_settings')
def get_detection_settings():
    """Get current detection settings."""
    try:
        settings = config.load_detection_settings()
        return jsonify({'settings': settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_detection_settings', methods=['POST'])
def update_detection_settings():
    """Update detection settings."""
    try:
        data = request.json
        settings = config.load_detection_settings()

        # Update provided settings
        for key, value in data.items():
            if key in ['enabled', 'scan_interval', 'global_min_confidence', 'max_matches_per_hour', 'cameras_enabled']:
                settings[key] = value

        # Validate settings
        if 'scan_interval' in settings and not (1 <= settings['scan_interval'] <= 300):
            return jsonify({'error': 'Scan interval must be between 1 and 300 seconds'}), 400

        if 'global_min_confidence' in settings and not (0 <= settings['global_min_confidence'] <= 100):
            return jsonify({'error': 'Global min confidence must be between 0 and 100'}), 400

        success = config.save_detection_settings(settings)
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

        success = config.update_detection_setting('enabled', enabled)
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
        settings = config.load_detection_settings()

        # Get recent matches count
        recent_matches = db.get_recent_matches_count(hours=1)

        status = {
            'enabled': settings.get('enabled', False),
            'scan_interval': settings.get('scan_interval', 10),
            'global_min_confidence': settings.get('global_min_confidence', 50),
            'thread_running': detection_service.is_running(),
            'recent_matches_last_hour': recent_matches,
            'last_modified': settings.get('last_modified', 'Never')
        }

        return jsonify({'status': status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# POSE EXTRACTION ROUTES
# =============================================================================

@app.route('/pose')
def pose():
    """Pose extraction page."""
    frigate_config = config.load_frigate_config()
    cameras = frigate_config.get('cameras') or {}
    return render_template('pose.html', cameras=cameras)


@app.route('/download_snapshot/<camera_name>')
def download_snapshot(camera_name):
    """Download high-quality camera snapshot for COLMAP."""
    try:
        # Get camera snapshot from Frigate
        import requests
        frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"

        response = requests.get(frigate_url, timeout=10)
        response.raise_for_status()

        return Response(
            response.content,
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': f'attachment; filename="{camera_name}.jpg"'
            }
        )
    except Exception as e:
        flash(f'Error downloading snapshot for {camera_name}: {str(e)}', 'error')
        return redirect(url_for('pose'))


@app.route('/upload_colmap', methods=['POST'])
def upload_colmap():
    """Upload and process COLMAP binary files."""
    try:
        print(f"Request content length: {request.content_length}")

        cameras_file = request.files.get('cameras_file')
        images_file = request.files.get('images_file')

        if not cameras_file or not images_file:
            return jsonify({
                'status': 'error',
                'message': 'Both cameras.bin and images.bin files are required'
            }), 400

        # Log file sizes
        cameras_file.seek(0, 2)  # Seek to end
        cameras_size = cameras_file.tell()
        cameras_file.seek(0)  # Reset to beginning

        images_file.seek(0, 2)  # Seek to end
        images_size = images_file.tell()
        images_file.seek(0)  # Reset to beginning

        print(f"Cameras file size: {cameras_size / (1024*1024):.2f} MB")
        print(f"Images file size: {images_size / (1024*1024):.2f} MB")

        # Create temporary directory for uploaded files
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            cameras_path = os.path.join(temp_dir, 'cameras.bin')
            images_path = os.path.join(temp_dir, 'images.bin')

            # Save uploaded files
            cameras_file.save(cameras_path)
            images_file.save(images_path)

            # Get camera configurations for matching
            frigate_config = config.load_frigate_config()
            camera_configs = frigate_config.get('cameras', {})

            # Process COLMAP files
            result = pose_manager.process_colmap_files(
                cameras_path, images_path, camera_configs
            )

            return jsonify(result)

    except Exception as e:
        import traceback
        print(f"Error in upload_colmap: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error processing COLMAP files: {str(e)}'
        }), 500


@app.route('/save_poses', methods=['POST'])
def save_poses():
    """Save extracted camera poses to database."""
    try:
        data = request.get_json()
        poses_data = data.get('poses', {})

        if not poses_data:
            return jsonify({
                'status': 'error',
                'message': 'No pose data provided'
            }), 400

        result = pose_manager.save_poses(poses_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saving poses: {str(e)}'
        }), 500


@app.route('/get_poses')
def get_poses():
    """Get all saved camera poses."""
    try:
        poses = pose_manager.get_all_poses()
        return jsonify(poses)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_pose/<camera_name>', methods=['POST'])
def delete_pose(camera_name):
    """Delete a saved camera pose."""
    try:
        success = pose_manager.delete_pose(camera_name)

        if success:
            return jsonify({
                'status': 'success',
                'message': f'Pose for camera "{camera_name}" deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to delete pose for camera "{camera_name}"'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error deleting pose: {str(e)}'
        }), 500


# =============================================================================
# YARD PROCESSING ROUTES
# =============================================================================

@app.route('/yard')
def yard():
    """Yard point cloud processing page."""
    # Check if we have a stored PLY file
    latest_ply = yard_manager.get_latest_ply_data()
    return render_template('yard.html', has_ply_file=latest_ply is not None)


@app.route('/upload_ply', methods=['POST'])
def upload_ply():
    """Upload and store a PLY file permanently."""
    try:
        ply_file = request.files.get('ply_file')
        if not ply_file:
            return jsonify({
                'status': 'error',
                'message': 'PLY file is required'
            }), 400

        # Read file data
        file_data = ply_file.read()
        filename = ply_file.filename or 'fused.ply'

        # Store in database
        result = yard_manager.store_ply_file(file_data, filename)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error uploading PLY file: {str(e)}'
        }), 500


@app.route('/get_latest_ply_info')
def get_latest_ply_info():
    """Get information about the latest stored PLY file."""
    try:
        ply_data = yard_manager.get_latest_ply_data()
        if ply_data:
            # Don't send the actual file data, just metadata
            return jsonify({
                'status': 'success',
                'has_file': True,
                'file_info': {
                    'id': ply_data['id'],
                    'name': ply_data['name'],
                    'vertex_count': ply_data['vertex_count'],
                    'has_color': ply_data['has_color'],
                    'format': ply_data['format'],
                    'uploaded_at': ply_data['uploaded_at']
                }
            })
        else:
            return jsonify({
                'status': 'success',
                'has_file': False
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting PLY info: {str(e)}'
        }), 500


@app.route('/scan_boundaries', methods=['POST'])
def scan_boundaries_stored():
    """Scan boundaries using stored PLY file with CUDA acceleration."""
    try:
        percentile_min = float(request.json.get('percentile_min', 2.0))
        percentile_max = float(request.json.get('percentile_max', 98.0))
        ply_id = request.json.get('ply_id')  # Optional specific PLY ID

        # Try ultra-fast NPY pipeline first, fallback to stored PLY
        if hasattr(yard_manager, 'npy_loader') and yard_manager.npy_loader:
            # Use ultra-fast memory-mapped NPY pipeline
            result = yard_manager.scan_boundaries_ultra_fast(
                None, percentile_min, percentile_max  # None = use latest dataset
            )
        else:
            # Fallback to stored PLY method
            result = yard_manager.scan_boundaries_stored(
                ply_id, percentile_min, percentile_max
            )
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error scanning boundaries: {str(e)}'
        }), 500


@app.route('/scan_boundaries_upload', methods=['POST'])
def scan_boundaries():
    """Scan point cloud boundaries with outlier removal."""
    try:
        ply_file = request.files.get('ply_file')
        if not ply_file:
            return jsonify({
                'status': 'error',
                'message': 'PLY file is required'
            }), 400

        percentile_min = float(request.form.get('percentile_min', 2.0))
        percentile_max = float(request.form.get('percentile_max', 98.0))

        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
            ply_file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Scan boundaries
            result = yard_manager.scan_boundaries(
                tmp_path, percentile_min, percentile_max
            )
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error scanning boundaries: {str(e)}'
        }), 500


@app.route('/project_yard', methods=['POST'])
def project_yard():
    """Project point cloud to create yard map."""
    try:
        # Check if using stored PLY or uploading new one
        ply_file = request.files.get('ply_file')
        use_stored = request.form.get('use_stored', 'false') == 'true'

        boundaries = json.loads(request.form.get('boundaries', '{}'))
        rotation = float(request.form.get('rotation', 0))
        resolution = request.form.get('resolution', '1080p')

        if use_stored or not ply_file:
            # Use stored PLY file
            ply_id = request.form.get('ply_id')
            result = yard_manager.process_stored_ply(ply_id)
            if isinstance(result, dict) and result.get('status') == 'error':
                return jsonify(result), 400

            tmp_path, ply_record = result
        else:
            # Use uploaded file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                ply_file.save(tmp_file.name)
                tmp_path = tmp_file.name

        try:
            # Project yard
            result = yard_manager.project_yard(
                tmp_path, boundaries, rotation, resolution
            )
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error projecting yard: {str(e)}'
        }), 500


@app.route('/save_yard_map', methods=['POST'])
def save_yard_map():
    """Save yard map to database."""
    try:
        # Check if using stored PLY or uploading new one
        ply_file = request.files.get('ply_file')
        use_stored = request.form.get('use_stored', 'false') == 'true'

        name = request.form.get('name')
        boundaries = json.loads(request.form.get('boundaries', '{}'))
        rotation = float(request.form.get('rotation', 0))
        resolution = request.form.get('resolution', '1080p')

        if not name:
            return jsonify({
                'status': 'error',
                'message': 'Name is required'
            }), 400

        if use_stored or not ply_file:
            # Use stored PLY file
            ply_id = request.form.get('ply_id')
            result = yard_manager.process_stored_ply(ply_id)
            if isinstance(result, dict) and result.get('status') == 'error':
                return jsonify(result), 400

            tmp_path, ply_record = result
        else:
            # Use uploaded file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                ply_file.save(tmp_file.name)
                tmp_path = tmp_file.name

        try:
            # Save yard map
            result = yard_manager.save_yard_map(
                name, tmp_path, boundaries, rotation, resolution
            )
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saving yard map: {str(e)}'
        }), 500


@app.route('/save_yard_map_direct', methods=['POST'])
def save_yard_map_direct():
    """Save yard map directly from base64 image data."""
    try:
        data = request.get_json()
        name = data.get('name')
        image_base64 = data.get('image_base64')

        if not name or not image_base64:
            return jsonify({
                'status': 'error',
                'message': 'Name and image data are required'
            }), 400

        # Decode base64 image
        import base64
        from PIL import Image
        from io import BytesIO
        image_data = base64.b64decode(image_base64)

        # Save to database with metadata
        result = yard_manager.save_yard_map_from_image(
            name=name,
            image_data=image_data,
            center_x=data.get('center_x', 0),
            center_y=data.get('center_y', 0),
            rotation=data.get('rotation', 0),
            resolution_x=data.get('width', 1920),
            resolution_y=data.get('height', 1080),
            algorithm=data.get('algorithm', 'unknown')
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error saving yard map: {str(e)}'
        }), 500


@app.route('/get_yard_maps')
def get_yard_maps():
    """Get all saved yard maps metadata."""
    try:
        maps = yard_manager.get_all_maps()
        return jsonify(maps)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/yard_image/<int:map_id>')
def yard_image(map_id):
    """Serve yard map image."""
    try:
        image_data = yard_manager.get_map_image(map_id)
        if image_data:
            return Response(image_data, mimetype='image/png')
        else:
            return 'Image not found', 404
    except Exception as e:
        return str(e), 500


@app.route('/delete_yard_map/<map_name>', methods=['POST'])
def delete_yard_map(map_name):
    """Delete a yard map."""
    try:
        success = yard_manager.delete_map(map_name)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Yard map "{map_name}" deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to delete yard map "{map_name}"'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error deleting yard map: {str(e)}'
        }), 500


@app.route('/use_yard_map/<int:map_id>', methods=['POST'])
def use_yard_map(map_id):
    """Set a yard map as 'used' for projection."""
    try:
        success = yard_manager.use_yard_map(map_id)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Yard map set as active for projection'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to set yard map as active'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error setting active yard map: {str(e)}'
        }), 500


@app.route('/get_used_yard_map')
def get_used_yard_map():
    """Get the currently used yard map."""
    try:
        used_map = yard_manager.get_used_map()
        if used_map:
            # Return without image_data to keep response small
            return jsonify({
                'status': 'success',
                'map': {
                    'id': used_map['id'],
                    'name': used_map['name'],
                    'center_x': used_map['center_x'],
                    'center_z': used_map['center_z'],
                    'width': used_map['width'],
                    'height': used_map['height'],
                    'rotation': used_map['rotation'],
                    'resolution_x': used_map['resolution_x'],
                    'resolution_y': used_map['resolution_y'],
                    'boundaries': used_map['boundaries']
                }
            })
        else:
            return jsonify({
                'status': 'none',
                'message': 'No yard map currently set for projection'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting used yard map: {str(e)}'
        }), 500


@app.route('/project_yard_interactive', methods=['POST'])
def project_yard_interactive():
    """Create interactive yard projection with CUDA acceleration."""
    try:
        # Check if we're receiving JSON data or form data
        if request.is_json:
            data = request.get_json()
            center_x = float(data.get('center_x', 0.0))
            center_y = float(data.get('center_y', 0.0))
            rotation = float(data.get('rotation', 0.0))
            resolution = float(data.get('resolution', 0.01))
            algorithm = data.get('algorithm', 'simple_average')
            use_stored = data.get('use_stored', True)
            ply_file = None
        else:
            # Form data with potential file upload
            params = json.loads(request.form.get('params', '{}'))
            center_x = float(params.get('center_x', 0.0))
            center_y = float(params.get('center_y', 0.0))
            rotation = float(params.get('rotation', 0.0))
            resolution = float(params.get('resolution', 0.01))
            algorithm = params.get('algorithm', 'simple_average')
            use_stored = params.get('use_stored', False)
            ply_file = request.files.get('ply_file')

        # Determine output size based on resolution
        # Default to 800x600, but adjust based on resolution for reasonable file sizes
        # Use 1920x1080 output size to match standalone script quality
        output_size = (1920, 1080)

        if ply_file and not use_stored:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                ply_file.save(tmp_file.name)
                tmp_path = tmp_file.name

            try:
                result = yard_manager.project_yard_interactive(
                    file_path=tmp_path,
                    center_x=center_x,
                    center_y=center_y,
                    rotation=rotation,
                    resolution=resolution,
                    output_size=output_size,
                    algorithm=algorithm
                )
                return jsonify(result)
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            # TEMPORARILY DISABLED: NPY rasterization due to data corruption issue
            # Using PLY-based rasterization until NPY corruption is fixed

            # Fallback to standard PLY-based rasterization
            result = yard_manager.project_yard_interactive(
                file_path=None,
                center_x=center_x,
                center_y=center_y,
                rotation=rotation,
                resolution=resolution,
                output_size=output_size,
                algorithm=algorithm
            )
            return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error creating interactive yard projection: {str(e)}'
        }), 500


# =============================================================================
# MULTI-CAMERA TRACKING ROUTES
# =============================================================================

@app.route('/projection')
def projection():
    """Camera-to-map projection page."""
    # Fetch camera config from Frigate API to get detect resolution
    try:
        response = requests.get(f'{FRIGATE_URL}/api/config', timeout=5)
        response.raise_for_status()
        frigate_config = response.json()
        cameras = frigate_config.get('cameras', {})
    except Exception as e:
        logger.warning(f"Could not fetch Frigate config from API: {e}, falling back to local config")
        frigate_config = config.load_frigate_config()
        cameras = frigate_config.get('cameras') or {}

    return render_template('projection.html', cameras=cameras)


@app.route('/project_camera_to_map', methods=['POST'])
def project_camera_to_map():
    """Project camera pixels onto map using ray tracing."""
    try:
        data = request.get_json()
        camera_name = data.get('camera_name')
        map_id = data.get('map_id')
        camera_width = data.get('camera_width', 1920)
        camera_height = data.get('camera_height', 1080)
        projection_method = data.get('projection_method', 'cuda')  # Default to CUDA

        logger.info(f"Computing projection for camera {camera_name} using {projection_method.upper()} method")

        # Get camera pose
        pose = pose_manager.get_camera_pose(camera_name)
        if not pose:
            return jsonify({
                'status': 'error',
                'message': f'No pose found for camera {camera_name}'
            }), 400

        # Transform pose format for camera projector
        # Extract camera position from camera_to_world transformation
        extrinsics = pose['extrinsics']
        intrinsics = pose['intrinsics']
        camera_to_world = extrinsics['camera_to_world']

        # CRITICAL: Scale intrinsics from COLMAP calibration resolution to actual camera resolution
        colmap_width = intrinsics['width']
        colmap_height = intrinsics['height']
        scale_x = camera_width / colmap_width
        scale_y = camera_height / colmap_height

        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics['fx'] = intrinsics['fx'] * scale_x
        scaled_intrinsics['fy'] = intrinsics['fy'] * scale_y
        scaled_intrinsics['cx'] = intrinsics['cx'] * scale_x
        scaled_intrinsics['cy'] = intrinsics['cy'] * scale_y
        scaled_intrinsics['width'] = camera_width
        scaled_intrinsics['height'] = camera_height

        logger.info(f"Scaled intrinsics from {colmap_width}x{colmap_height} to {camera_width}x{camera_height}")
        logger.info(f"  fx: {intrinsics['fx']:.1f} -> {scaled_intrinsics['fx']:.1f}")
        logger.info(f"  fy: {intrinsics['fy']:.1f} -> {scaled_intrinsics['fy']:.1f}")

        # Extract camera position from camera_to_world matrix
        cam_pos_x = camera_to_world[0][3]
        cam_pos_y = camera_to_world[1][3]
        cam_pos_z = camera_to_world[2][3]

        logger.info(f"Camera position from COLMAP: X={cam_pos_x:.2f}, Y={cam_pos_y:.2f}, Z={cam_pos_z:.2f}")

        # FIX: COLMAP camera orientation is reversed - flip 180° around Z-axis (forward)
        # This corrects cameras looking into the house instead of outward
        rotation_matrix = np.array(extrinsics['rotation_matrix'])
        flip_180_z = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        corrected_rotation = flip_180_z @ rotation_matrix

        logger.info(f"Applied 180° Z-axis rotation to correct camera orientation")

        camera_pose_transformed = {
            'position_x': cam_pos_x,
            'position_y': cam_pos_y,
            'position_z': cam_pos_z,
            'rotation_matrix': corrected_rotation.tolist(),
            'quaternion': extrinsics['quaternion'],
            'intrinsics': scaled_intrinsics  # Use scaled intrinsics for accurate projection
        }

        # Get map info
        used_map = yard_manager.get_used_map()
        if not used_map or used_map['id'] != map_id:
            return jsonify({
                'status': 'error',
                'message': 'Map is not set as active'
            }), 400

        # Normalize boundaries format for camera projection
        # Boundary detector uses x_min/x_max/z_min/z_max
        # Camera projector expects min_x/max_x/min_y/max_y
        boundaries = used_map['boundaries']
        if 'x_min' in boundaries and 'min_x' not in boundaries:
            boundaries = {
                'min_x': boundaries['x_min'],
                'max_x': boundaries['x_max'],
                'min_y': boundaries['z_min'],  # Z in 3D -> Y in 2D map
                'max_y': boundaries['z_max']
            }
            used_map['boundaries'] = boundaries

        # Get point cloud path
        from yard_manager import PLY_FILE_PATH
        if not os.path.exists(PLY_FILE_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Point cloud file not found'
            }), 400

        # Create projector and compute projection based on selected method
        if projection_method == 'cpu':
            # Use CPU version
            from camera_projection import CameraProjector
            projector = CameraProjector(PLY_FILE_PATH)
            logger.info("Using CPU projection (user selected)")

            result = projector.project_camera_to_map(
                camera_pose=camera_pose_transformed,
                map_info=used_map,
                camera_width=camera_width,
                camera_height=camera_height,
                sample_rate=1  # Scan all pixels
            )
        else:
            # Use CuPy CUDA version for 240x speedup
            try:
                from camera_projection_cupy import CameraProjectorCuPy
                projector = CameraProjectorCuPy(PLY_FILE_PATH)
                logger.info("Using CuPy CUDA-accelerated projection")

                # Scan all pixels for complete coverage
                # Sample rate of 1 = all pixels (~5M rays for 2560x1920)
                result = projector.project_camera_to_map(
                    camera_pose=camera_pose_transformed,
                    map_info=used_map,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    sample_rate=1  # Scan all pixels
                )
            except ImportError as e:
                logger.warning(f"CuPy not available, falling back to CPU: {e}")
                from camera_projection import CameraProjector
                projector = CameraProjector(PLY_FILE_PATH)
                logger.info("Using CPU projection (automatic fallback)")

                result = projector.project_camera_to_map(
                    camera_pose=camera_pose_transformed,
                    map_info=used_map,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    sample_rate=1  # Scan all pixels
                )

        # Add camera and map identifiers to result
        result['camera_name'] = camera_name
        result['map_id'] = map_id
        result['camera_width'] = camera_width
        result['camera_height'] = camera_height
        result['status'] = 'success'

        # Debug logging
        logger.info(f"Projection result: {result['pixel_count']} pixels mapped")
        logger.info(f"Projected pixels array length: {len(result.get('projected_pixels', []))}")
        if result.get('projected_pixels'):
            logger.info(f"First 5 projected pixels: {result['projected_pixels'][:5]}")
        logger.info(f"Bounds: {result.get('bounds')}")

        # Store pixel_mappings server-side to avoid sending 118MB to browser
        # We'll use a simple in-memory cache (in production, use Redis or similar)
        global _projection_cache
        if '_projection_cache' not in globals():
            _projection_cache = {}

        projection_id = f"{camera_name}_{map_id}_{int(time.time())}"
        _projection_cache[projection_id] = {
            'pixel_mappings': result.pop('pixel_mappings'),  # Remove from response
            'camera_name': camera_name,
            'map_id': map_id,
            'timestamp': time.time()
        }

        # Send projection_id to frontend for saving later
        result['projection_id'] = projection_id

        logger.info(f"Stored projection with ID: {projection_id}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error computing projection: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error computing projection: {str(e)}'
        }), 500


@app.route('/get_projection_status', methods=['GET'])
def get_projection_status():
    """Get projection status for all cameras for a specific map."""
    try:
        map_id = request.args.get('map_id', type=int)

        if not map_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing map_id'
            }), 400

        # Get all saved projections for this map
        projections = yard_manager.get_all_projections_for_map(map_id)

        return jsonify({
            'status': 'success',
            'projections': projections
        })

    except Exception as e:
        logger.error(f"Error getting projection status: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/save_projection', methods=['POST'])
def save_projection():
    """Save projection mapping to database."""
    try:
        projection_data = request.get_json()

        projection_id = projection_data.get('projection_id')

        if not projection_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing projection_id'
            }), 400

        # Retrieve cached projection data
        global _projection_cache
        if '_projection_cache' not in globals() or projection_id not in _projection_cache:
            return jsonify({
                'status': 'error',
                'message': 'Projection expired. Please re-run the projection.'
            }), 404

        cached = _projection_cache[projection_id]
        camera_name = cached['camera_name']
        map_id = cached['map_id']
        pixel_mappings = cached['pixel_mappings']

        logger.info(f"Saving projection {projection_id}: {len(pixel_mappings):,} pixel mappings")

        # Save to database
        success = yard_manager.save_projection(
            camera_name=camera_name,
            map_id=map_id,
            pixel_mappings=pixel_mappings,
            metadata=projection_data
        )

        # Clean up cache after saving
        del _projection_cache[projection_id]
        logger.info(f"Cleaned up cached projection {projection_id}")

        if success:
            return jsonify({
                'status': 'success',
                'message': f'Projection saved for {camera_name}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to save projection'
            }), 500

    except Exception as e:
        logger.error(f"Error saving projection: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error saving projection: {str(e)}'
        }), 500


@app.route('/tracking')
def tracking():
    """Multi-camera tracking dashboard."""
    return render_template('tracking.html')


@app.route('/tracking/get_active_tracks')
def get_active_tracks():
    """Get all active tracks with statistics."""
    try:
        tracks = tracking_service.get_toddler_tracks()
        all_tracks = tracking_service.tracker.get_active_tracks()
        stats = tracking_service.get_tracking_stats()

        return jsonify({
            'tracks': all_tracks,
            'toddler_tracks': tracks,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tracking/get_handoffs')
def get_handoffs():
    """Get recent camera handoff events."""
    try:
        hours = int(request.args.get('hours', 1))
        handoffs = tracking_service.tracker.get_camera_handoffs(hours)

        return jsonify({
            'handoffs': handoffs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tracking/get_track_history/<global_id>')
def get_track_history(global_id):
    """Get position history for a specific track."""
    try:
        limit = int(request.args.get('limit', 100))
        history = tracking_service.tracker.get_track_history(global_id, limit)

        return jsonify({
            'history': history,
            'global_id': global_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tracking/mark_as_toddler', methods=['POST'])
def mark_as_toddler():
    """Mark a track as belonging to the toddler."""
    try:
        data = request.get_json()
        track_id = data.get('track_id')
        confidence = float(data.get('confidence', 0.95))

        if not track_id:
            return jsonify({
                'status': 'error',
                'message': 'Track ID is required'
            }), 400

        tracking_service.tracker.mark_as_toddler(track_id, confidence)

        return jsonify({
            'status': 'success',
            'message': f'Track {track_id} marked as toddler with {confidence*100:.1f}% confidence'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error marking track as toddler: {str(e)}'
        }), 500


@app.route('/tracking/simulate_detection', methods=['POST'])
def simulate_detection():
    """Simulate a detection for testing purposes."""
    try:
        import random
        import numpy as np

        # Simulate detections from multiple cameras
        cameras = ['backyard', 'garage', 'side_yard']
        detections = []

        for i, camera in enumerate(cameras):
            if random.random() > 0.3:  # 70% chance of detection per camera
                detection = {
                    'camera': camera,
                    'track_id': f'sim_{i}_{random.randint(1, 100)}',
                    'bbox': [
                        random.randint(100, 800),  # x
                        random.randint(100, 500),  # y
                        random.randint(50, 150),   # width
                        random.randint(80, 200)    # height
                    ],
                    'confidence': random.uniform(0.6, 0.95),
                    'features': np.random.rand(512)  # Simulated feature vector
                }
                detections.append(detection)

        # Process detections
        if detections:
            results = tracking_service.process_detections(detections)

            return jsonify({
                'status': 'success',
                'message': f'Simulated {len(detections)} detections',
                'results': {
                    'new_tracks': len(results.get('tracks', [])),
                    'handoffs': len(results.get('handoffs', [])),
                    'disappeared': len(results.get('disappeared', []))
                }
            })
        else:
            return jsonify({
                'status': 'info',
                'message': 'No detections simulated this round'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error simulating detection: {str(e)}'
        }), 500


@app.route('/tracking/clear_all', methods=['DELETE'])
def clear_all_tracking():
    """Clear all tracking data including tracks and handoffs."""
    try:
        success = tracking_service.tracker.clear_all_data()

        if success:
            return jsonify({
                'status': 'success',
                'message': 'All tracking data cleared successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to clear tracking data'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error clearing tracking data: {str(e)}'
        }), 500


@app.route('/tracking/process_detections', methods=['POST'])
def process_detections():
    """Process real detections from external sources (e.g., Frigate)."""
    try:
        data = request.get_json()
        detections = data.get('detections', [])

        if not detections:
            return jsonify({
                'status': 'error',
                'message': 'No detections provided'
            }), 400

        # Validate detection format
        for det in detections:
            required_fields = ['camera', 'track_id', 'bbox', 'confidence']
            for field in required_fields:
                if field not in det:
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing required field: {field}'
                    }), 400

        # Process detections
        results = tracking_service.process_detections(detections)

        return jsonify({
            'status': 'success',
            'message': f'Processed {len(detections)} detections',
            'results': results
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing detections: {str(e)}'
        }), 500


@app.route('/get_cameras')
def get_cameras():
    """Get camera list for tracking interface."""
    try:
        frigate_config = config.load_frigate_config()
        cameras = frigate_config.get('cameras') or {}
        return jsonify(cameras)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    # Background detection disabled - running in Docker container
    # Uncomment to run detection in this process:
    # detection_service.start_background_detection()
    app.run(debug=True, host='0.0.0.0', port=9000)
