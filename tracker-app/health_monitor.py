"""
Health monitoring for system components.
Checks status of Frigate, CompreFace, MQTT, Home Assistant, and detection service.
"""

import requests
from datetime import datetime
from typing import Dict
from config_manager import ConfigManager
from detection_service import DetectionService


class HealthMonitor:
    """Monitors health of all system components."""

    def __init__(self, config: ConfigManager, detection_service: DetectionService):
        self.config = config
        self.detection_service = detection_service

    def check_all_components(self) -> Dict:
        """Check health status of all system components."""
        health_status = {
            'frigate': self.check_frigate_health(),
            'compreface': self.check_compreface_health(),
            'mqtt': self.check_mqtt_health(),
            'homeassistant': self.check_homeassistant_health(),
            'live_detection': self.check_live_detection_health()
        }

        # Overall system status
        all_healthy = all(status['status'] == 'healthy' for status in health_status.values())

        return {
            'overall_status': 'healthy' if all_healthy else 'degraded',
            'components': health_status,
            'timestamp': datetime.now().isoformat()
        }

    def check_frigate_health(self) -> Dict:
        """Check if Frigate is responding."""
        try:
            response = requests.get('http://localhost:5000/api/config', timeout=5)
            if response.status_code == 200:
                return {'status': 'healthy', 'message': 'Frigate API responding'}
            else:
                return {'status': 'unhealthy', 'message': f'Frigate API returned {response.status_code}'}
        except requests.exceptions.RequestException as e:
            return {'status': 'unhealthy', 'message': f'Frigate connection error: {str(e)}'}

    def check_compreface_health(self) -> Dict:
        """Check if CompreFace is responding."""
        try:
            client = self.config.get_compreface_client()
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

    def check_mqtt_health(self) -> Dict:
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

    def check_homeassistant_health(self) -> Dict:
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

    def check_live_detection_health(self) -> Dict:
        """Check live detection system status."""
        try:
            settings = self.config.load_detection_settings()
            enabled = settings.get('enabled', False)

            # Check if detection thread is running
            thread_running = self.detection_service.is_running()

            if enabled and thread_running:
                return {'status': 'healthy', 'message': 'Live detection active and running'}
            elif not enabled:
                return {'status': 'disabled', 'message': 'Live detection disabled by user'}
            else:
                return {'status': 'unhealthy', 'message': 'Live detection enabled but thread not running'}
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Live detection error: {str(e)}'}