"""
Database operations for the Toddler Tracker application.
Handles matches database and related operations.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class MatchesDatabase:
    """Database manager for face recognition matches."""

    def __init__(self, db_path: str = 'matches.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the matches database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Original matches table
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

        # Enhanced matches table for hybrid identification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hybrid_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                camera TEXT,
                image_data BLOB,
                identification_method TEXT,
                method_scores TEXT,
                osnet_features BLOB,
                color_features BLOB,
                bbox TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # OSNet person features table
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

        # Color features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS color_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                color_histogram BLOB NOT NULL,
                image_data BLOB,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT
            )
        ''')

        # Toddler position tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS toddler_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                camera TEXT NOT NULL,
                map_x INTEGER,
                map_y INTEGER,
                confidence REAL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def add_match(self, subject: str, confidence: float, camera: str = None,
                  face_id: str = None, image_data: bytes = None) -> int:
        """Add a face recognition match to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO matches (subject, confidence, timestamp, camera, face_id, image_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (subject, confidence, datetime.now().isoformat(), camera, face_id, image_data))
        conn.commit()
        match_id = cursor.lastrowid
        conn.close()
        return match_id

    def get_matches(self, limit: int = 100) -> List[Dict]:
        """Get recent matches from the database."""
        conn = sqlite3.connect(self.db_path)
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

    def get_match_image(self, match_id: int) -> Optional[bytes]:
        """Get image data for a specific match."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT image_data FROM matches WHERE id = ?', (match_id,))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] else None

    def clear_matches(self):
        """Clear all matches from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM matches')
        conn.commit()
        conn.close()

    def get_recent_matches_count(self, hours: int = 1) -> int:
        """Get count of matches within the specified hours."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cursor.execute('''
            SELECT COUNT(*) FROM matches
            WHERE datetime(timestamp) > datetime(?)
        ''', (cutoff_time.isoformat(),))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0

    # Hybrid identification methods

    def add_hybrid_match(self, subject: str, confidence: float, camera: str = None,
                        image_data: bytes = None, identification_method: str = "hybrid",
                        method_scores: Dict = None, osnet_features: bytes = None,
                        color_features: bytes = None, bbox: List = None) -> int:
        """Add a hybrid identification match to the database."""
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        method_scores_str = json.dumps(method_scores) if method_scores else None
        bbox_str = json.dumps(bbox) if bbox else None

        cursor.execute('''
            INSERT INTO hybrid_matches (
                subject, confidence, timestamp, camera, image_data,
                identification_method, method_scores, osnet_features,
                color_features, bbox
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (subject, confidence, datetime.now().isoformat(), camera, image_data,
              identification_method, method_scores_str, osnet_features,
              color_features, bbox_str))

        conn.commit()
        match_id = cursor.lastrowid
        conn.close()
        return match_id

    def get_hybrid_matches(self, limit: int = 100) -> List[Dict]:
        """Get recent hybrid matches from the database."""
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, subject, confidence, timestamp, camera, identification_method,
                   method_scores, bbox, created_at
            FROM hybrid_matches
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        matches = []
        for row in cursor.fetchall():
            method_scores = None
            bbox = None

            try:
                if row[6]:  # method_scores
                    method_scores = json.loads(row[6])
                if row[7]:  # bbox
                    bbox = json.loads(row[7])
            except:
                pass

            matches.append({
                'id': row[0],
                'subject': row[1],
                'confidence': row[2],
                'timestamp': row[3],
                'camera': row[4],
                'identification_method': row[5],
                'method_scores': method_scores,
                'bbox': bbox,
                'created_at': row[8]
            })

        conn.close()
        return matches

    def add_person_features(self, person_id: str, features: bytes,
                           image_data: bytes = None, source: str = "manual") -> int:
        """Add OSNet features for a person."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO person_features (person_id, features, image_data, source)
            VALUES (?, ?, ?, ?)
        ''', (person_id, features, image_data, source))
        conn.commit()
        feature_id = cursor.lastrowid
        conn.close()
        return feature_id

    def get_person_features(self, person_id: str) -> List[bytes]:
        """Get all OSNet features for a person."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT features FROM person_features WHERE person_id = ?
        ''', (person_id,))

        results = cursor.fetchall()
        conn.close()
        return [row[0] for row in results if row[0]]

    def add_color_features(self, person_id: str, color_histogram: bytes,
                          image_data: bytes = None, source: str = "manual") -> int:
        """Add color features for a person."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO color_features (person_id, color_histogram, image_data, source)
            VALUES (?, ?, ?, ?)
        ''', (person_id, color_histogram, image_data, source))
        conn.commit()
        feature_id = cursor.lastrowid
        conn.close()
        return feature_id

    def get_color_features(self, person_id: str) -> List[bytes]:
        """Get all color features for a person."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT color_histogram FROM color_features WHERE person_id = ?
        ''', (person_id,))

        results = cursor.fetchall()
        conn.close()
        return [row[0] for row in results if row[0]]

    def clear_all_hybrid_data(self):
        """Clear all hybrid identification data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM hybrid_matches')
        cursor.execute('DELETE FROM person_features')
        cursor.execute('DELETE FROM color_features')
        conn.commit()
        conn.close()

    def get_hybrid_stats(self) -> Dict:
        """Get statistics about hybrid identification."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Count hybrid matches
        cursor.execute('SELECT COUNT(*) FROM hybrid_matches')
        stats['total_hybrid_matches'] = cursor.fetchone()[0]

        # Count person features
        cursor.execute('SELECT COUNT(DISTINCT person_id) FROM person_features')
        stats['trained_persons'] = cursor.fetchone()[0]

        # Count color features
        cursor.execute('SELECT COUNT(DISTINCT person_id) FROM color_features')
        stats['persons_with_color'] = cursor.fetchone()[0]

        # Recent matches (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        cursor.execute('''
            SELECT COUNT(*) FROM hybrid_matches
            WHERE datetime(timestamp) > datetime(?)
        ''', (cutoff_time.isoformat(),))
        stats['recent_matches_24h'] = cursor.fetchone()[0]

        conn.close()
        return stats

    def add_toddler_position(self, subject: str, camera: str, map_x: int, map_y: int,
                            confidence: float, timestamp: str = None) -> int:
        """Add a toddler position record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO toddler_positions (subject, camera, map_x, map_y, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (subject, camera, map_x, map_y, confidence, timestamp))

        position_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return position_id

    def get_last_toddler_position(self, subject: str = None) -> Optional[Dict]:
        """Get the most recent toddler position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if subject:
            cursor.execute('''
                SELECT subject, camera, map_x, map_y, confidence, timestamp
                FROM toddler_positions
                WHERE subject = ?
                ORDER BY datetime(timestamp) DESC
                LIMIT 1
            ''', (subject,))
        else:
            cursor.execute('''
                SELECT subject, camera, map_x, map_y, confidence, timestamp
                FROM toddler_positions
                ORDER BY datetime(timestamp) DESC
                LIMIT 1
            ''')

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'subject': row[0],
                'camera': row[1],
                'map_x': row[2],
                'map_y': row[3],
                'confidence': row[4],
                'timestamp': row[5]
            }
        return None

    def get_toddler_positions(self, subject: str = None, limit: int = 100) -> List[Dict]:
        """Get recent toddler positions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if subject:
            cursor.execute('''
                SELECT id, subject, camera, map_x, map_y, confidence, timestamp
                FROM toddler_positions
                WHERE subject = ?
                ORDER BY datetime(timestamp) DESC
                LIMIT ?
            ''', (subject, limit))
        else:
            cursor.execute('''
                SELECT id, subject, camera, map_x, map_y, confidence, timestamp
                FROM toddler_positions
                ORDER BY datetime(timestamp) DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            positions.append({
                'id': row[0],
                'subject': row[1],
                'camera': row[2],
                'map_x': row[3],
                'map_y': row[4],
                'confidence': row[5],
                'timestamp': row[6]
            })

        return positions