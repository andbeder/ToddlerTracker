"""
OSNet Feature Extractor for Person Re-Identification
Integrates OSNet models for extracting person re-identification features
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import logging
from typing import Optional, Dict, List, Tuple
import torchreid
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


class OSNetExtractor:
    """OSNet feature extractor for person re-identification."""

    def __init__(self, model_name: str = 'osnet_x1_0', pretrained: bool = True, device: Optional[str] = None):
        """
        Initialize OSNet feature extractor.

        Args:
            model_name: OSNet model variant ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25')
            pretrained: Whether to use pretrained weights
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # Initialize model
        self.model = self._load_model(model_name, pretrained)
        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessing transforms
        self.transform = self._get_transforms()

        logger.info(f"OSNet extractor initialized with model {model_name} on device {self.device}")

    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Load OSNet model."""
        try:
            # Load model using torchreid
            model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,  # Will be ignored for feature extraction
                pretrained=pretrained,
                use_gpu=(self.device == 'cuda')
            )

            # Remove classifier layers for feature extraction
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            elif hasattr(model, 'fc'):
                model.fc = nn.Identity()

            return model

        except Exception as e:
            logger.error(f"Failed to load OSNet model {model_name}: {e}")
            # Fallback to a basic model if torchreid fails
            return self._create_dummy_model()

    def _create_dummy_model(self) -> nn.Module:
        """Create a dummy model for testing purposes."""
        class DummyOSNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 512)
                )

            def forward(self, x):
                return self.features(x)

        logger.warning("Using dummy OSNet model - install proper torchreid models for production")
        return DummyOSNet()

    def _get_transforms(self) -> T.Compose:
        """Get preprocessing transforms for OSNet."""
        return T.Compose([
            T.Resize((256, 128)),  # Standard Reid input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> np.ndarray:
        """
        Extract OSNet features from person image.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            bbox: Optional bounding box [x, y, width, height] to crop person region

        Returns:
            Feature vector as numpy array (512-dimensional)
        """
        try:
            # Crop person region if bbox provided
            if bbox is not None:
                x, y, w, h = [int(coord) for coord in bbox]
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)

                if w > 0 and h > 0:
                    image = image[y:y+h, x:x+w]
                else:
                    logger.debug("Invalid bbox coordinates, using full image")

            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)

            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)

                # Ensure features are 2D and get the first (and only) sample
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)

                # Normalize features
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                # Convert to numpy
                features_np = features.cpu().numpy()[0]  # Get first sample

            return features_np

        except Exception as e:
            logger.error(f"Error extracting OSNet features: {e}")
            # Return zero vector on error
            return np.zeros(512, dtype=np.float32)

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        try:
            # Ensure features are normalized
            f1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
            f2_norm = features2 / (np.linalg.norm(features2) + 1e-8)

            # Compute cosine similarity
            similarity = np.dot(f1_norm, f2_norm)

            # Clamp to [0, 1] range and convert to Python float
            return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def extract_features_batch(self, images: List[np.ndarray], bboxes: Optional[List[List[float]]] = None) -> List[np.ndarray]:
        """
        Extract features from multiple images in batch.

        Args:
            images: List of input images
            bboxes: Optional list of bounding boxes corresponding to images

        Returns:
            List of feature vectors
        """
        features_list = []

        for i, image in enumerate(images):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            features = self.extract_features(image, bbox)
            features_list.append(features)

        return features_list

    def save_features(self, features: np.ndarray, filepath: str):
        """Save features to file."""
        try:
            np.save(filepath, features)
            logger.debug(f"Features saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")

    def load_features(self, filepath: str) -> Optional[np.ndarray]:
        """Load features from file."""
        try:
            features = np.load(filepath)
            logger.debug(f"Features loaded from {filepath}")
            return features
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'feature_dim': 512,  # Standard OSNet feature dimension
            'input_size': (256, 128),
            'pretrained': True
        }


class OSNetDatabase:
    """Database for storing and retrieving OSNet features."""

    def __init__(self, db_path: str = 'osnet_features.db'):
        """Initialize OSNet feature database."""
        import sqlite3
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables."""
        cursor = self.conn.cursor()

        # Table for storing person features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                features BLOB NOT NULL,
                image_data BLOB,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                confidence REAL DEFAULT 1.0
            )
        ''')

        # Table for similarity matches
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS osnet_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                features BLOB NOT NULL,
                image_data BLOB,
                camera TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                bbox TEXT
            )
        ''')

        self.conn.commit()

    def add_person_features(self, person_id: str, features: np.ndarray,
                           image_data: Optional[bytes] = None, source: str = "manual") -> int:
        """Add person features to database."""
        cursor = self.conn.cursor()

        # Serialize features
        features_blob = features.tobytes()

        cursor.execute('''
            INSERT INTO person_features (person_id, features, image_data, source)
            VALUES (?, ?, ?, ?)
        ''', (person_id, features_blob, image_data, source))

        self.conn.commit()
        return cursor.lastrowid

    def get_person_features(self, person_id: str) -> List[np.ndarray]:
        """Get all features for a person."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT features FROM person_features WHERE person_id = ?
        ''', (person_id,))

        results = cursor.fetchall()
        features_list = []

        for (features_blob,) in results:
            features = np.frombuffer(features_blob, dtype=np.float32)
            features_list.append(features)

        return features_list

    def add_match(self, person_id: str, similarity: float, features: np.ndarray,
                  image_data: Optional[bytes] = None, camera: str = "", bbox: Optional[List[float]] = None):
        """Add a similarity match to database."""
        cursor = self.conn.cursor()

        features_blob = features.tobytes()
        bbox_str = str(bbox) if bbox else None

        cursor.execute('''
            INSERT INTO osnet_matches (person_id, similarity, features, image_data, camera, bbox)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (person_id, similarity, features_blob, image_data, camera, bbox_str))

        self.conn.commit()

    def get_recent_matches(self, hours: int = 24, min_similarity: float = 0.5) -> List[Dict]:
        """Get recent matches above similarity threshold."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT person_id, similarity, camera, timestamp, bbox
            FROM osnet_matches
            WHERE similarity >= ? AND datetime(timestamp) >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours), (min_similarity,))

        results = cursor.fetchall()
        matches = []

        for person_id, similarity, camera, timestamp, bbox in results:
            matches.append({
                'person_id': person_id,
                'similarity': similarity,
                'camera': camera,
                'timestamp': timestamp,
                'bbox': eval(bbox) if bbox else None
            })

        return matches

    def close(self):
        """Close database connection."""
        self.conn.close()


# Global OSNet extractor instance
_osnet_extractor = None

def get_osnet_extractor() -> OSNetExtractor:
    """Get global OSNet extractor instance."""
    global _osnet_extractor
    if _osnet_extractor is None:
        _osnet_extractor = OSNetExtractor()
    return _osnet_extractor

def extract_person_features(image: np.ndarray, bbox: Optional[List[float]] = None) -> np.ndarray:
    """Convenience function to extract features using global extractor."""
    extractor = get_osnet_extractor()
    return extractor.extract_features(image, bbox)

def compute_person_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """Convenience function to compute similarity using global extractor."""
    extractor = get_osnet_extractor()
    return extractor.compute_similarity(features1, features2)