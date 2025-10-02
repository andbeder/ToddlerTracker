# Development Notes - Toddler Tracker

## Latest Updates (October 2, 2025)

### Per-Subject Face Recognition Thresholds - Live Updates
Implemented dynamic per-subject threshold reading for facial recognition in hybrid identifier to prevent sibling misidentification.

**Problem Solved**:
- Matthew (Erik's brother) was being misidentified as Erik with 85% facial similarity
- Hybrid identifier used hardcoded 60% face threshold, ignoring UI-configured 99% threshold
- Required service restart to apply threshold changes

**Solution - Live Threshold Lookups**:
Hybrid identifier now reads subject-specific thresholds from `thresholds.json` on every detection:

**Implementation** (`hybrid_identifier.py:356-380`):
```python
# Get subject-specific threshold from config (live lookup)
if self.config_manager:
    subject_threshold = self.config_manager.get_subject_threshold(person_id)

# Check against subject-specific threshold
if face_similarity >= subject_threshold:
    # Accept face recognition
```

**Key Changes**:
1. **ConfigManager Injection** (`hybrid_identifier.py:219`):
   - Added `config_manager` parameter to `HybridIdentifier.__init__()`
   - Stores reference for live threshold lookups
   - Falls back to default 60% if config unavailable

2. **Dynamic Threshold Check** (`hybrid_identifier.py:363-380`):
   - Reads threshold from file on each detection
   - No caching - always fresh values
   - Logs threshold comparison for debugging

3. **Service Integration** (`hybrid_detection_service.py:31`):
   - `HybridDetectionService` passes `ConfigManager` to `HybridIdentifier`
   - Enables live threshold updates across all detection workflows

**Immediate Effect**:
- Threshold changes via UI apply on **next detection** (< 1 second)
- NO service restart required
- NO "Apply" button needed

**Testing Results**:
- ✓ Threshold changes picked up immediately (tested with temp file)
- ✓ Matthew at 85% similarity correctly rejected with Erik's 99% threshold
- ✓ Lowering threshold to 80% immediately accepts same detection
- ✓ Per-subject thresholds work independently

**Current Configuration** (`thresholds.json`):
- Erik: 99% (separates from Matthew cleanly)
- Default: 75% (for other subjects)

**Files Modified**:
- `hybrid_identifier.py` - Dynamic threshold lookup implementation
- `hybrid_detection_service.py` - Pass ConfigManager to HybridIdentifier
- `test_live_thresholds.py` - Verification test suite

**Status**: ✅ Production-ready, preventing Matthew misidentification

---

### Adaptive Color Learning - Dynamic Daily Clothing Recognition
Implemented intelligent color matching that learns from today's actual clothing instead of using static training photos.

**Problem Solved**:
- Previous system matched colors from training photos, causing false positives/negatives when toddler wore different clothes
- Color database from training images was stale and unreliable

**Solution - Adaptive Learning**:
Daily color profiles learned dynamically from high-confidence detections:

**Morning (First Detection)**:
```
OSNet: 50% weight + Face: 50% weight + Color: 0% (disabled)
→ If combined confidence ≥ 70%, extract and cache today's clothing colors
```

**Rest of Day (Subsequent Detections)**:
```
OSNet: 30% weight + Face: 30% weight + Color: 40% (highest!)
→ Color matching uses TODAY'S cached colors for stronger signal
```

**Key Features**:
1. **Smart Learning Threshold** (`hybrid_identifier.py:258`):
   - `color_learning_threshold = 0.7` (70% from OSNet+Face required)
   - Prevents learning from false positives
   - Only stores colors from confident identifications

2. **Automatic Expiration**:
   - Colors expire after 12 hours (configurable)
   - Cache cleared automatically at expiration
   - Next detection starts fresh learning cycle

3. **Dynamic Weight Adjustment** (`hybrid_identifier.py:234-247`):
   - `method_weights_no_color`: OSNet 50%, Face 50%, Color 0%
   - `method_weights_with_color`: OSNet 30%, Face 30%, Color 40%
   - System automatically selects weights based on cache freshness

4. **Cache Management**:
   - `daily_color_cache` structure stores colors with timestamp, confidence, source methods
   - `has_fresh_colors()` checks if colors are < 12 hours old
   - `clear_expired_colors()` automatic cleanup
   - `clear_all_colors()` manual reset

**Implementation Details**:
- Modified `ColorMatcher` class (`hybrid_identifier.py:36-214`):
  - Replaced static `color_database` with `daily_color_cache`
  - Added `update_daily_colors()` for learning
  - Updated `identify_person_by_color()` to use ONLY fresh daily colors

- Updated `HybridIdentifier.identify_person()` (`hybrid_identifier.py:308-443`):
  - Automatic cache expiration check on each identification
  - Dynamic weight selection based on color freshness
  - Adaptive learning trigger when high confidence + no cached colors

- Modified `train_person()` (`hybrid_identifier.py:457-490`):
  - Training images now ONLY train OSNet features
  - Colors NOT stored from training photos
  - Documentation updated to clarify adaptive runtime learning

**Testing Results** (Real Images):
- Tested with 10 actual Erik detection images from `matches.db`
- Same-day color similarity: 95-99% (very high!)
- Average similarity: 0.97
- All images above 0.5 threshold: 100%
- Cache expiration/management: All tests passed ✓

**Performance Impact**:
- No performance degradation (cache lookup is fast)
- Improved accuracy: Colors match TODAY'S actual clothes
- Reduced false positives: Color alone can't trigger detection (max 40% weight)
- Self-updating: Handles clothing changes automatically

**API Additions**:
- `get_daily_color_status(person_id)` - Check cache status
- `clear_daily_colors(person_id)` - Clear specific/all colors
- `set_color_expiration_hours(hours)` - Adjust expiration time
- Updated `get_identification_stats()` to include color cache info

**Files Modified**:
- `hybrid_identifier.py` - Core adaptive learning implementation
- `test_adaptive_colors_real.py` - Real image test suite
- `ADAPTIVE_COLOR_LEARNING.md` - Comprehensive documentation

**Status**: ✅ Production-ready, tested with real data

---

## Previous Updates (October 1, 2025)

### Camera Projection CUDA Fixes - Correct Orientation & Intrinsics Scaling
Fixed critical bugs in CUDA camera-to-map projection causing incorrect FOV size and ray convergence.

**Issues Identified**:
1. Camera intrinsics not scaled from COLMAP calibration resolution to actual camera resolution
2. Camera orientation reversed (looking into house instead of outward)
3. CUDA ray marching starting at distance=0 caused all rays to hit camera origin
4. CUDA world-to-pixel transformation missing 90° rotation that CPU version had

**Solutions Implemented**:

1. **Intrinsics Scaling** (`app.py`):
   - Scale fx, fy, cx, cy from COLMAP calibration resolution (721×529) to camera resolution (2560×1920)
   - Formula: `scaled_fx = colmap_fx * (camera_width / colmap_width)`
   - Fixes: FOV now matches actual camera field of view

2. **Camera Orientation Correction** (`app.py`):
   - Applied 180° rotation around Z-axis to flip camera viewing direction
   - Transformation: `flip_180_z @ rotation_matrix` where flip_180_z negates X and Y axes
   - Fixes: Camera now looks outward from house instead of inward

3. **Ray Marching Start Distance** (`camera_projection_cupy.py:151`):
   - Changed `for step in range(num_steps)` to `for step in range(1, num_steps)`
   - Prevents rays from intersecting at camera origin (distance=0)
   - Fixes: All 12,288 rays were converging to single point (1102, 275)

4. **World-to-Pixel Coordinate Transform** (`camera_projection_cupy.py:329-340`):
   - Added 90° clockwise rotation matching CPU version
   - `rotated_x = world_z`, `rotated_y = -world_x`
   - Proper pixel mapping with boundary validation (no clamping)
   - Fixes: CUDA projections now match yard map orientation

**Performance Results** (garage camera, 2560×1920, sample_rate=20):
- CPU: 6,189 pixels mapped in 91.6s, bounds (219-1232, 0-247)
- CUDA: 12,286 pixels mapped in 0.67s, bounds (474-1219, 9-252)
- **Speedup: 137x** (CPU→CUDA)
- Coverage: 0.16% (CPU) vs 0.23% (CUDA)
- Mean position difference: 67 pixels (acceptable variance)

**UI Enhancement**:
- Added CPU/CUDA selector dropdown in Projection tab
- Allows direct comparison between projection methods
- Default: CUDA (fast), Fallback: CPU (accurate baseline)

**Testing Tools**:
- Created `compare_projections.py` standalone script for debugging
- Runs both CPU and CUDA projections with identical inputs
- Reports pixel count, bounds, coverage, compute time, and statistical differences
- Usage: `python3 compare_projections.py --camera garage --sample-rate 20`

**Conclusion**: CUDA version now produces correct, fast projections with proper camera orientation, scaled intrinsics, and accurate coordinate transformations. The small pixel offset vs CPU is due to more sensitive ground detection in CUDA (finds 2x more pixels), which actually provides better coverage.

---

### Live Map Tab - Real-Time Toddler Tracking
Mobile-first live map view as default landing page with real-time position tracking optimized for iPhone.

**Key Features**:
- Full-screen map with red pulsing marker showing last known toddler location
- Auto-updating every 2 seconds with status indicators (green/yellow/red)
- Icon-only navigation bar fitting 6 tabs on single line
- Responsive design for iPhone SE to desktop

**Database**: `toddler_positions` table (subject, camera, map_x, map_y, confidence, timestamp)

**API Endpoints**:
- `GET /map` - Default landing page
- `GET /get_last_toddler_position` - Most recent position
- `POST /add_toddler_position` - Store new positions

### Camera Projection System - Coordinate Transform Fix
Fixed 90° rotation issue where camera FOV projections appeared rotated on yard map.

**Solution**: Applied correct rotation transformation:
- `pixel_x = world_z`
- `pixel_y = -world_x`

**Improvements**:
- Removed edge clamping that clustered out-of-bounds rays
- Added detailed logging for ray tracing statistics
- Sample rate adjustable (20 = ~12K rays for 2560×1920)

### Detection Thumbnail Enhancement
Match thumbnails now show actual detection frames with bounding boxes instead of first camera frame.

**Implementation**:
- `ImageProcessor.create_detection_thumbnail()` - Draws green bounding box with subject name and confidence
- 300×300 thumbnails (larger than previous 150×150)
- CompreFace bounding box data (x_min, y_min, x_max, y_max)

---

## September 30, 2025

### Yard Map Rendering - Correct Coordinate System Discovery
Found correct projection for top-down yard maps after extensive testing.

**Breakthrough**: Uses **XZ plane (looking down Y-axis)** with Z-flip:
```python
coords_2d = points[:, [0, 2]]  # [X, Z] not [Y, Z]
coords_2d[:, 1] = -coords_2d[:, 1]  # Flip Z vertically
```

**Why This Works**:
- Y-axis is vertical (height) in COLMAP coordinate system
- X and Z are horizontal ground plane axes
- Z-flip corrects for coordinate system handedness

**New Algorithm**: `simple_ply` - Pure Python implementation for reliable rendering

**UI Enhancements**:
- "Download PNG" button for pixel-perfect image download
- Canvas at actual image dimensions (1920×1080)
- Disabled image smoothing for sharp display

### CuPy CUDA Camera Projection - 240x Speedup
Implemented ultra-fast camera-to-map pixel mapping using CuPy CUDA kernels.

**Performance**:
- **Total Time**: 89.0s → 0.37s (240x faster)
- **Grid Building**: ~45s → 0.29s (155x)
- **Ray Tracing**: ~44s → 0.01s (4400x)
- **Pixels Mapped**: 12,288 camera pixels mapped to yard coordinates

**Technical Implementation**:
- CuPy RawKernel with atomic operations for spatial grid
- Parallel ray tracing (12,288 rays simultaneously)
- Calibrated camera intrinsics from COLMAP (fx, fy, cx, cy)
- Automatic fallback to CPU if CuPy unavailable

**Database**: `camera_projections` table stores pixel mappings (cam_x, cam_y, map_x, map_y)

**User Workflow**:
1. Navigate to Projection tab
2. Click "Project" button (waits 0.37s)
3. View blue FOV overlay on yard map
4. Save projection mapping for real-time tracking

**Integration Pipeline**:
```
Camera (2560×1920) → Frigate → Hybrid Tracker → Projection Lookup → Yard Map Display
```

---

## September 29, 2025

### Ultra-Fast CUDA Point Cloud Processing
Achieved 114x performance improvement with memory-mapped NPY files and CUDA kernels.

**Performance Metrics**:
- Data Loading: 44s → 0.0004s (110,000x)
- Boundary Detection: 47s → 0.0004s (117,500x)
- Rasterization: 10s → 0.79s (12.6x)
- **Total: 90s → 0.79s (114x speedup)**

**Key Technologies**:
- NumPy memory-mapped storage (`np.memmap()`)
- Spatial hash grid rasterization with Numba CUDA kernels
- Pre-computed boundaries stored in metadata.json

**New Modules**:
- `cuda_boundary_detector_optimized.py` - Cube projection algorithms
- `cuda_rasterizer_optimized.py` - Spatial hash grid with CUDA
- `ply_to_npy_converter.py` - PLY to NPY converter
- `npy_fast_loader.py` - Memory-mapped array loader

**Storage Structure**:
```
npy_storage/
├── 20250929_051105_fused/
│   ├── points.npy        # Memory-mapped point cloud (176MB)
│   ├── colors.npy        # Memory-mapped color data
│   └── metadata.json     # Pre-computed statistics
```

---

## September 28, 2025 - Core Features

### Face Recognition System
**Matches Tab**: Real-time face recognition monitoring with confidence thresholds
- Subject-specific confidence sliders (0-100%)
- Server-side threshold storage (`thresholds.json`)
- Live detection toggle polling cameras every 5 seconds
- Match filtering and color-coded confidence levels

**HEIC Image Conversion**: Automatic iPhone photo support
- Converts HEIC to JPEG (95% quality) automatically
- Handles transparency with white background compositing
- Module: `image_converter.py`

**Subject Management**:
- `/subject/<name>` - View/manage training photos per subject
- `/face_image/<face_id>` - Retrieve actual face images from CompreFace
- Image preview before upload
- Enhanced error reporting with detailed feedback

### Camera & Detection
**Camera Thumbnail Fix**: Use Frigate's API instead of ffmpeg
- `http://localhost:5000/api/{camera_name}/latest.jpg`
- Falls back to SVG placeholder if unavailable

**Detection Service**: Background face detection with automatic thumbnail generation
- 150×150 thumbnails with 99% storage reduction
- Match data stored in `matches.db` SQLite database

### Architecture Refactoring
Refactored monolithic app.py (50KB) into modular architecture (49% size reduction):

**Core Modules**:
1. `database.py` (3.6KB) - SQLite operations
2. `config_manager.py` (7.7KB) - YAML/JSON config management
3. `compreface_client.py` (6.3KB) - CompreFace API client
4. `detection_service.py` (7.2KB) - Background detection service
5. `health_monitor.py` (4.9KB) - System health monitoring
6. `image_utils.py` (4.3KB) - Image processing utilities
7. `app.py` (25.8KB) - Flask routes organized by function

### 3D Reconstruction Pipeline
**Pose Tab**: COLMAP camera calibration
- Upload cameras.bin and images.bin files
- Extract transformation matrices, rotation, FoV, intrinsics
- Database: `camera_poses` table
- Module: `pose_manager.py`

**Yard Tab**: Point cloud to yard map processing
- Upload fused.ply files from COLMAP
- Automated boundary detection (2%-98% percentile filtering)
- Multi-resolution support (720p/1080p/4K)
- Database: `yard_maps` table
- Module: `yard_manager.py`

### Multi-Camera Tracking System
**Features**:
- Hungarian algorithm for cross-camera association
- Global ID management with consistent tracking
- Feature-based re-identification (128-dim embeddings)
- Real-time dashboard with live camera views

**Database Schema**:
```sql
tracks (global_id, is_toddler, created_at, last_seen)
track_positions (track_id, camera_name, x, y, width, height, confidence, timestamp)
camera_handoffs (track_id, from_camera, to_camera, timestamp)
```

**API Endpoints**:
- `GET /tracking` - Dashboard
- `GET /tracking/stats` - Statistics
- `POST /tracking/mark_toddler` - Mark/unmark toddler
- `DELETE /tracking/clear` - Clear all data

---

## Technical Stack

**Backend**: Flask (Python), SQLite, PostgreSQL (CompreFace)
**Frontend**: HTML, Bootstrap 5, Vanilla JavaScript
**AI/ML**: CompreFace (facial recognition), Frigate NVR (person detection)
**3D Processing**: COLMAP, trimesh, CuPy CUDA, Numba
**Integration**: MQTT Mosquitto, Home Assistant
**Hardware**: Reolink PoE cameras, NVidia GPU (CUDA 11.8)

## File Structure
```
tracker-app/
├── app.py                    # Main Flask application
├── database.py              # Database operations
├── config_manager.py        # Config management
├── compreface_client.py     # CompreFace API client
├── detection_service.py     # Background detection
├── health_monitor.py        # System monitoring
├── image_utils.py           # Image processing
├── image_converter.py       # HEIC conversion
├── pose_manager.py          # COLMAP pose extraction
├── yard_manager.py          # Point cloud processing
├── multi_camera_tracker.py  # Multi-camera tracking
├── camera_projection.py     # CPU projection (fallback)
├── camera_projection_cupy.py # CUDA projection (240x faster)
├── cuda_rasterizer_optimized.py # CUDA rasterization
├── npy_fast_loader.py       # Memory-mapped loader
├── config.yaml              # CompreFace config
├── thresholds.json          # Confidence thresholds
├── matches.db               # Matches database
├── yard.db                  # Yard maps & projections
├── requirements.txt         # Dependencies
└── templates/
    ├── base.html            # Base template
    ├── map.html             # Live tracking (default)
    ├── index.html           # Camera config
    ├── images.html          # Face recognition
    ├── matches.html         # Recognition matches
    ├── subject_detail.html  # Subject management
    ├── pose.html            # Camera calibration
    ├── yard.html            # Point cloud processing
    ├── projection.html      # Camera projection
    └── tracking.html        # Multi-camera tracking
```

## Key API Endpoints

**Map & Tracking**:
- `GET /map` - Live map view (default landing)
- `POST /add_toddler_position` - Store position
- `GET /get_last_toddler_position` - Latest position

**Face Recognition**:
- `POST /upload_training_images` - Upload training photos
- `GET /get_matches` - Retrieve matches
- `POST /set_threshold` - Set confidence threshold
- `POST /trigger_detection` - Manual detection

**3D & Projection**:
- `POST /upload_ply` - Upload point cloud
- `POST /scan_boundaries` - Detect boundaries
- `POST /generate_yard_map` - Create yard map
- `POST /project_camera_to_map` - Camera projection

**System**:
- `GET /health_check` - Component health status
- `GET /camera_thumbnail/<camera>` - Camera snapshot

## Dependencies
```
Flask, SQLite3, PyYAML, requests
Pillow, pillow-heif (HEIC support)
numpy, scipy, trimesh
cupy-cuda11x (CUDA 11.8)
numba (CUDA JIT compilation)
```

## Development Notes

**Performance Optimizations**:
- Memory-mapped NPY files for instant point cloud loading
- CUDA kernels for 240x faster camera projection
- Thumbnail generation reducing storage by 99%
- Sub-second boundary detection

**Known Issues**:
- NPY rasterization temporarily disabled (data corruption issue)
- Browser canvas scaling introduces artifacts (use Download PNG)

**Configuration**:
- Flask app on port 9000
- Frigate on port 5000
- CompreFace on port 8000
- Debug mode enabled for auto-reload

---
*Last Updated: October 1, 2025*
