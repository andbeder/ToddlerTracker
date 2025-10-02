"""
Configuration management for the Toddler Tracker application.
Handles Frigate config, app settings, thresholds, and detection settings.
"""

import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from compreface_client import CompreFaceClient


class ConfigManager:
    """Manages all configuration files and settings."""

    def __init__(self,
                 frigate_config_path: str = '../frigate/config/config.yaml',
                 app_config_path: str = 'config.yaml',
                 thresholds_path: str = 'thresholds.json',
                 detection_settings_path: str = 'detection_settings.json'):
        self.frigate_config_path = frigate_config_path
        self.app_config_path = app_config_path
        self.thresholds_path = thresholds_path
        self.detection_settings_path = detection_settings_path

    # Frigate Configuration
    def load_frigate_config(self) -> Dict[str, Any]:
        """Load the Frigate configuration from YAML file."""
        try:
            with open(self.frigate_config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as e:
            print(f'Error parsing Frigate YAML: {e}')
            return {}

    def save_frigate_config(self, config: Dict[str, Any]) -> bool:
        """Save the Frigate configuration to YAML file."""
        try:
            with open(self.frigate_config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f'Error saving Frigate config: {e}')
            return False

    # App Configuration
    def load_app_config(self) -> Dict[str, Any]:
        """Load app configuration including CompreFace settings."""
        try:
            if os.path.exists(self.app_config_path):
                with open(self.app_config_path, 'r') as file:
                    return yaml.safe_load(file) or {}
            return {}
        except Exception:
            return {}

    def save_app_config(self, config: Dict[str, Any]) -> bool:
        """Save app configuration."""
        try:
            with open(self.app_config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            return True
        except Exception:
            return False

    def get_compreface_client(self) -> Optional[CompreFaceClient]:
        """Get configured CompreFace client."""
        config = self.load_app_config()
        api_url = config.get('compreface_url')
        api_key = config.get('compreface_api_key')

        if api_url and api_key:
            return CompreFaceClient(api_url, api_key)
        return None

    # Subject Thresholds
    def load_thresholds(self) -> Dict[str, float]:
        """Load subject thresholds from JSON config file."""
        try:
            if os.path.exists(self.thresholds_path):
                with open(self.thresholds_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return {}

    def save_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """Save subject thresholds to JSON config file."""
        try:
            with open(self.thresholds_path, 'w') as f:
                json.dump(thresholds, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving thresholds: {e}")
            return False

    def get_subject_threshold(self, subject: str) -> float:
        """Get threshold for a specific subject (default 75%)."""
        thresholds = self.load_thresholds()
        return thresholds.get(subject, 75.0)

    def set_subject_threshold(self, subject: str, threshold: float) -> bool:
        """Set threshold for a specific subject."""
        thresholds = self.load_thresholds()
        thresholds[subject] = threshold
        return self.save_thresholds(thresholds)

    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all subject thresholds with defaults for known subjects."""
        thresholds = self.load_thresholds()

        # Get list of known subjects from CompreFace
        client = self.get_compreface_client()
        if client:
            subjects = client.list_subjects()
            for subject in subjects:
                if subject not in thresholds:
                    thresholds[subject] = 75.0  # Default threshold

        return thresholds

    # Detection Settings
    def load_detection_settings(self) -> Dict[str, Any]:
        """Load detection settings from JSON config file."""
        try:
            if os.path.exists(self.detection_settings_path):
                with open(self.detection_settings_path, 'r') as f:
                    return json.load(f)
            return self.get_default_detection_settings()
        except Exception as e:
            print(f"Error loading detection settings: {e}")
            return self.get_default_detection_settings()

    def get_default_detection_settings(self) -> Dict[str, Any]:
        """Get default detection settings."""
        return {
            "enabled": False,
            "scan_interval": 10,  # seconds
            "global_min_confidence": 50,  # percentage
            "max_matches_per_hour": 100,  # rate limiting
            "cameras_enabled": True,  # whether to scan cameras
            "last_modified": datetime.now().isoformat()
        }

    def save_detection_settings(self, settings: Dict[str, Any]) -> bool:
        """Save detection settings to JSON config file."""
        try:
            settings["last_modified"] = datetime.now().isoformat()
            with open(self.detection_settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving detection settings: {e}")
            return False

    def update_detection_setting(self, key: str, value: Any) -> bool:
        """Update a specific detection setting."""
        settings = self.load_detection_settings()
        settings[key] = value
        return self.save_detection_settings(settings)

    # RTSP URL Parsing Utility
    @staticmethod
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