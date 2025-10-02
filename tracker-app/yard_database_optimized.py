"""
Optimized YardDatabase with file-based PLY storage.
Stores PLY files on disk and only keeps metadata in database.
"""

import sqlite3
import json
import os
import shutil
from typing import Dict, List, Optional
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)

class OptimizedYardDatabase:
    """Database manager optimized for file-based PLY storage."""

    def __init__(self, db_path: str = 'yard.db', storage_dir: str = 'ply_storage'):
        self.db_path = db_path
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
        self._init_optimized_schema()

    def _ensure_storage_dir(self):
        """Ensure PLY storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info(f"Created PLY storage directory: {self.storage_dir}")

    def _init_optimized_schema(self):
        """Initialize optimized database schema with file paths instead of blobs."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Create optimized PLY files table (no blob storage)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ply_files_optimized (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    vertex_count INTEGER,
                    has_color BOOLEAN,
                    format TEXT,
                    checksum TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Keep existing yard_maps table as-is
            conn.execute('''
                CREATE TABLE IF NOT EXISTS yard_maps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    boundaries TEXT NOT NULL,
                    rotation REAL DEFAULT 0.0,
                    resolution TEXT DEFAULT '1080p',
                    image_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
        finally:
            conn.close()

    def save_ply_file_optimized(self, name: str, file_path_or_data,
                               vertex_count: int = None, has_color: bool = None,
                               format: str = None) -> Dict:
        """
        Save PLY file to disk and metadata to database.

        Args:
            name: File name
            file_path_or_data: Either path to existing file or bytes data
            vertex_count: Number of vertices
            has_color: Whether file contains color data
            format: File format (binary/ascii)

        Returns:
            Dict with file info including path
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = name.replace(' ', '_').replace('/', '_')
            if not safe_name.endswith('.ply'):
                safe_name += '.ply'
            unique_name = f"{timestamp}_{safe_name}"
            dest_path = os.path.join(self.storage_dir, unique_name)

            # Save file to disk
            if isinstance(file_path_or_data, bytes):
                # Write bytes data to file
                with open(dest_path, 'wb') as f:
                    f.write(file_path_or_data)
                file_size = len(file_path_or_data)
            else:
                # Copy existing file
                shutil.copy2(file_path_or_data, dest_path)
                file_size = os.path.getsize(dest_path)

            # Calculate checksum for integrity
            checksum = self._calculate_checksum(dest_path)

            # Save metadata to database
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute('''
                    INSERT INTO ply_files_optimized
                    (name, file_path, file_size, vertex_count, has_color, format, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (name, dest_path, file_size, vertex_count, has_color, format, checksum))

                file_id = cursor.lastrowid
                conn.commit()

                logger.info(f"✅ Saved PLY to disk: {dest_path} ({file_size:,} bytes)")

                return {
                    'id': file_id,
                    'name': name,
                    'file_path': dest_path,
                    'file_size': file_size,
                    'vertex_count': vertex_count,
                    'has_color': has_color,
                    'format': format,
                    'checksum': checksum
                }
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error saving PLY file: {e}")
            raise

    def get_latest_ply_file_optimized(self) -> Optional[Dict]:
        """Get the most recently uploaded PLY file path and metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, file_path, file_size, vertex_count, has_color,
                       format, checksum, uploaded_at
                FROM ply_files_optimized
                ORDER BY uploaded_at DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()

            if row:
                file_path = row[2]

                # Verify file exists
                if not os.path.exists(file_path):
                    logger.warning(f"PLY file not found on disk: {file_path}")
                    return None

                return {
                    'id': row[0],
                    'name': row[1],
                    'file_path': file_path,
                    'file_size': row[3],
                    'vertex_count': row[4],
                    'has_color': row[5],
                    'format': row[6],
                    'checksum': row[7],
                    'uploaded_at': row[8]
                }
            return None

        except Exception as e:
            logger.error(f"Error getting PLY file: {e}")
            return None
        finally:
            conn.close()

    def get_ply_file_by_id_optimized(self, ply_id: int) -> Optional[Dict]:
        """Get PLY file path and metadata by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, file_path, file_size, vertex_count, has_color,
                       format, checksum, uploaded_at
                FROM ply_files_optimized
                WHERE id = ?
            ''', (ply_id,))
            row = cursor.fetchone()

            if row:
                file_path = row[2]

                # Verify file exists
                if not os.path.exists(file_path):
                    logger.warning(f"PLY file not found on disk: {file_path}")
                    return None

                return {
                    'id': row[0],
                    'name': row[1],
                    'file_path': file_path,
                    'file_size': row[3],
                    'vertex_count': row[4],
                    'has_color': row[5],
                    'format': row[6],
                    'checksum': row[7],
                    'uploaded_at': row[8]
                }
            return None

        except Exception as e:
            logger.error(f"Error getting PLY file by ID: {e}")
            return None
        finally:
            conn.close()

    def migrate_existing_ply(self):
        """Migrate existing PLY files from blob storage to file storage."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Check if old table exists
            cursor = conn.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='ply_files'
            ''')
            if not cursor.fetchone():
                logger.info("No existing PLY files to migrate")
                return

            # Get all PLY files from old table
            cursor = conn.execute('''
                SELECT id, name, file_data, vertex_count, has_color, format
                FROM ply_files
            ''')

            migrated_count = 0
            for row in cursor.fetchall():
                old_id, name, file_data, vertex_count, has_color, format = row

                # Save to new system
                result = self.save_ply_file_optimized(
                    name=name,
                    file_path_or_data=file_data,
                    vertex_count=vertex_count,
                    has_color=has_color,
                    format=format
                )

                if result:
                    migrated_count += 1
                    logger.info(f"Migrated: {name} -> {result['file_path']}")

            conn.close()
            logger.info(f"✅ Migrated {migrated_count} PLY files to file storage")

        except Exception as e:
            logger.error(f"Error during migration: {e}")

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def cleanup_orphaned_files(self):
        """Remove PLY files that are on disk but not in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('SELECT file_path FROM ply_files_optimized')
            db_files = set(row[0] for row in cursor.fetchall())
            conn.close()

            # Check all files in storage directory
            disk_files = set()
            if os.path.exists(self.storage_dir):
                for file in os.listdir(self.storage_dir):
                    if file.endswith('.ply'):
                        disk_files.add(os.path.join(self.storage_dir, file))

            # Find orphaned files
            orphaned = disk_files - db_files

            for file_path in orphaned:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed orphaned file: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")

            if orphaned:
                logger.info(f"✅ Cleaned up {len(orphaned)} orphaned files")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # Keep compatibility with existing yard_maps methods
    def get_yard_maps(self) -> List[Dict]:
        """Get all saved yard maps."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT name, boundaries, rotation, resolution, created_at, updated_at
                FROM yard_maps
                ORDER BY updated_at DESC
            ''')

            maps = []
            for row in cursor.fetchall():
                maps.append({
                    'name': row[0],
                    'boundaries': json.loads(row[1]),
                    'rotation': row[2],
                    'resolution': row[3],
                    'created_at': row[4],
                    'updated_at': row[5]
                })

            return maps
        except Exception as e:
            logger.error(f"Error getting yard maps: {e}")
            return []
        finally:
            conn.close()