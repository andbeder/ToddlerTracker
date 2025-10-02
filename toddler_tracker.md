# Toddler Tracker Application - Functional Requirements

## Overview

The Toddler Tracker Application is a comprehensive safety monitoring system designed to track the location and movement of a toddler (Erik) within a residential yard environment using computer vision, multi-camera surveillance, and 3D spatial mapping. The system provides real-time position tracking, safety alerts, and historical movement analysis through an intuitive web interface.

## Primary Use Case

**Parents monitoring their toddler's location and safety in a backyard environment through real-time camera surveillance and AI-powered position tracking.**

---

## Application Architecture

### Core Components
1. **Web Application**: React/Vue.js frontend with Flask backend
2. **Computer Vision Pipeline**: Real-time object detection and tracking
3. **Camera System**: Multi-camera surveillance network (Frigate NVR)
4. **3D Reconstruction**: COLMAP-based spatial mapping and camera calibration
5. **Real-time Communication**: MQTT messaging and WebSocket connections
6. **Face Recognition**: CompreFace integration for person identification

### Technology Stack
- **Frontend**: Vue.js 3, HTML5, CSS3, JavaScript
- **Backend**: Flask (Python), SQLite/PostgreSQL
- **Computer Vision**: COLMAP, OpenCV, YOLO/Object Detection
- **Messaging**: MQTT, WebSocket
- **Surveillance**: Frigate NVR
- **Infrastructure**: Docker, Nginx

---

## User Interface Structure

The application uses a **tabbed interface** with four main sections:

### 1. **Map Tab** (Primary View)
- **Purpose**: Real-time tracking visualization
- **Default View**: Always visible on application load

### 2. **Live Tab** 
- **Purpose**: Live camera feed monitoring
- **View**: 4-panel camera grid

### 3. **Matches Tab**
- **Purpose**: Face recognition and detection match review
- **View**: Match history, confidence scores, and manual review interface

### 4. **Config Tab** (Administrative)
- **Purpose**: System configuration and setup
- **Sub-tabs**: Photos, Reconstruct, Orient, Pose, Yard, Cameras, Settings

---

## Detailed Functional Requirements

## 1. MAP TAB - Live Tracking Interface

### 1.1 Primary Display
**FR-MAP-001**: The system SHALL display a top-down 2D map of the monitored yard area
- **Input**: Generated yard map from 3D reconstruction
- **Display**: Bird's-eye view with spatial boundaries
- **Fallback**: Show "No Yard Map Available" message if map not generated

**FR-MAP-002**: The system SHALL show Erik's current position as a real-time dot on the map
- **Indicator**: Red pulsing dot (20px diameter)
- **Position**: Calculated from camera triangulation
- **Update Frequency**: Every 2-3 seconds
- **Animation**: Smooth movement transitions

**FR-MAP-003**: The system SHALL display Erik's movement trail
- **Trail Length**: Last 10-15 position points
- **Visual**: Fading line showing movement path
- **Color**: Semi-transparent with fade effect
- **Persistence**: Trail clears after 5 minutes of inactivity

### 1.2 Status Panel
**FR-MAP-004**: The system SHALL display a real-time status panel showing:
- Erik's detection status (Detected/Not Detected)
- Last seen timestamp
- Current active cameras
- Detection confidence level
- Movement speed indicator

**FR-MAP-005**: The system SHALL provide safety alerts
- **Boundary Alerts**: When Erik approaches yard boundaries
- **No Detection Alerts**: When Erik not detected for >30 seconds
- **Camera Offline Alerts**: When cameras become unavailable
- **Visual Indicators**: Color-coded status lights (Green/Yellow/Red)

### 1.3 Real-time Updates
**FR-MAP-006**: The system SHALL update position data in real-time
- **Protocol**: WebSocket connection for live updates
- **Fallback**: AJAX polling every 2 seconds
- **Data Source**: MQTT messages from Frigate detection events
- **Latency Target**: <1 second from detection to display

---

## 2. LIVE TAB - Camera Monitoring

### 2.1 Camera Grid Display
**FR-LIVE-001**: The system SHALL display a 2x2 grid of live camera feeds
- **Cameras**: front_door, backyard, side_yard, garage
- **Format**: Latest snapshot images (JPEG)
- **Refresh Rate**: Every 5-10 seconds
- **Resolution**: Optimized for web display (800x600 max)

**FR-LIVE-002**: Each camera panel SHALL show:
- Camera name/label
- Current snapshot image
- Timestamp of last update
- Online/offline status indicator
- Click-to-enlarge functionality

### 2.2 Fullscreen Modal
**FR-LIVE-003**: The system SHALL provide fullscreen camera viewing
- **Trigger**: Click on any camera panel
- **Display**: Enlarged view with camera controls
- **Features**: Manual refresh, close button, camera switching
- **Performance**: Optimized image loading

### 2.3 Detection Overlays
**FR-LIVE-004**: The system SHALL overlay detection information on camera feeds
- **Bounding Boxes**: Around detected persons
- **Confidence Scores**: Percentage confidence display
- **Identity Labels**: "Erik" when face recognition matches
- **Timestamp**: When detection occurred

---

## 3. MATCHES TAB - Detection Review Interface

### 3.1 Match History Display
**FR-MATCHES-001**: The system SHALL display a chronological list of face recognition matches
- **Match Records**: Show all Erik identification events
- **Timestamp**: Precise time of each detection
- **Camera Source**: Which camera made the detection
- **Confidence Score**: Percentage confidence of the match
- **Thumbnail**: Small image of the detected face

**FR-MATCHES-002**: The system SHALL provide match filtering and search capabilities
- **Date Range**: Filter matches by date/time range
- **Camera Filter**: Show matches from specific cameras only
- **Confidence Filter**: Filter by minimum confidence threshold
- **Search**: Search by keywords or notes
- **Pagination**: Handle large numbers of matches efficiently

### 3.2 Match Review Interface
**FR-MATCHES-003**: The system SHALL allow manual review of detection matches
- **Review Status**: Mark matches as confirmed/rejected/uncertain
- **Bulk Operations**: Select multiple matches for batch review
- **Notes**: Add comments or notes to specific matches
- **Export**: Export match data for analysis
- **Statistics**: Show match accuracy statistics

**FR-MATCHES-004**: The system SHALL display detailed match information
- **Full Image**: Large view of the detection image
- **Bounding Box**: Show detection area on original image
- **Reference Comparison**: Side-by-side with reference photo
- **Detection Metadata**: Camera settings, lighting conditions
- **Similar Matches**: Show related/similar detections

### 3.3 Real-time Match Notifications
**FR-MATCHES-005**: The system SHALL provide real-time match notifications
- **Live Updates**: New matches appear immediately
- **Visual Indicators**: Highlight new/unreviewed matches
- **Sound Alerts**: Optional audio notification for new matches
- **Badge Counters**: Show count of unreviewed matches
- **Auto-refresh**: Update match list automatically

### 3.4 Match Quality Analysis
**FR-MATCHES-006**: The system SHALL analyze and display match quality metrics
- **Daily Statistics**: Matches per day, average confidence
- **Camera Performance**: Which cameras provide best matches
- **Time Patterns**: When Erik is most commonly detected
- **False Positive Rate**: Track and display accuracy metrics
- **Quality Trends**: Show improvement/degradation over time

### 3.5 Training Data Management
**FR-MATCHES-007**: The system SHALL support training data improvement
- **Positive Samples**: Confirmed matches added to training set
- **Negative Samples**: Rejected matches improve false positive filtering
- **Model Updates**: Trigger face recognition model updates
- **Feedback Loop**: Use review results to improve detection accuracy
- **Performance Tracking**: Monitor improvement from training updates

---

## 4. CONFIG TAB - Administrative Interface

## 4.1 Photos Sub-tab
**FR-CONFIG-PHOTOS-001**: The system SHALL manage reference photo library
- **Upload**: Drag-and-drop photo upload interface
- **Gallery**: Thumbnail grid of uploaded photos
- **Management**: Add, delete, organize reference photos
- **Formats**: Support JPEG, PNG image formats
- **Storage**: Local filesystem with metadata database

## 4.2 Reconstruct Sub-tab
**FR-CONFIG-RECONSTRUCT-001**: The system SHALL provide COLMAP reconstruction workflow
- **Snapshot Download**: Download current camera snapshots for COLMAP processing
- **File Upload**: Upload COLMAP output files (cameras.bin, images.bin, fused.ply)
- **Pose Extraction**: Extract camera poses from COLMAP files
- **Validation**: Verify uploaded reconstruction files
- **Status Display**: Show current reconstruction status

## 4.3 Reconstruct Sub-tab (continued)
**FR-CONFIG-RECONSTRUCT-001**: The system SHALL provide COLMAP reconstruction workflow
- **Snapshot Download**: Download current camera snapshots for COLMAP processing
- **File Upload**: Upload COLMAP output files (cameras.bin, images.bin, fused.ply)
- **Pose Extraction**: Extract camera poses from COLMAP files
- **Validation**: Verify uploaded reconstruction files
- **Status Display**: Show current reconstruction status

**FR-CONFIG-RECONSTRUCT-002**: Snapshot download functionality
- **Individual Downloads**: Download snapshot from each camera
- **Bulk Download**: Download all snapshots as ZIP file
- **Naming Convention**: Automatic naming (camera_name.jpg)
- **File Validation**: Ensure snapshot availability before download

**FR-CONFIG-RECONSTRUCT-003**: COLMAP file upload and processing
- **File Validation**: Verify COLMAP binary format
- **Size Limits**: Maximum 500MB per file
- **Required Files**: cameras.bin, images.bin, fused.ply
- **Progress Indicators**: Upload progress and status
- **Error Handling**: Clear error messages for failed uploads

**FR-CONFIG-RECONSTRUCT-004**: Camera pose extraction
- **Camera Matching**: Match image names to camera identifiers
- **Pose Calculation**: Convert COLMAP poses to world coordinates
- **Validation**: Verify pose reasonableness and quality
- **Results Display**: Show extracted poses in organized table
- **Integration**: Pass poses to Orient tab for verification

## 4.4 Orient Sub-tab
**FR-CONFIG-ORIENT-001**: The system SHALL provide camera orientation verification
- **Camera Selection**: Dropdown to select specific camera
- **Pose Display**: Show calculated 3D position and rotation
- **Visual Verification**: 3D visualization of camera frustum
- **Manual Adjustment**: Fine-tune camera poses if needed
- **Save Functionality**: Persist calibrated poses

## 4.5 Pose Sub-tab
**FR-CONFIG-POSE-001**: The system SHALL manage camera pose pipeline
- **Pose Status**: Display current pose calibration status
- **Quality Metrics**: Show pose accuracy and confidence
- **Recalibration**: Trigger pose recalculation if needed
- **Validation Results**: Display pose validation outcomes

## 4.6 Yard Sub-tab
**FR-CONFIG-YARD-001**: The system SHALL generate yard maps from 3D reconstruction
- **Mesh Processing**: Process uploaded fused.ply point cloud
- **Map Generation**: Create 2D top-down view from 3D data
- **Boundary Detection**: Automatically detect yard boundaries
- **Scale Calibration**: Set real-world scale for map
- **Preview**: Show generated map before saving

**FR-CONFIG-YARD-002**: Yard map configuration
- **Map Settings**: Configure map appearance and boundaries
- **Coordinate System**: Define coordinate mapping
- **Safety Zones**: Define safe/restricted areas
- **Export Options**: Save map in various formats

## 4.7 Cameras Sub-tab
**FR-CONFIG-CAMERAS-001**: The system SHALL manage Frigate camera configuration
- **Camera Grid**: Display all configured cameras with thumbnails
- **Camera Details**: Show RTSP URLs, resolution, FPS settings
- **Detection Settings**: Configure object detection parameters
- **Live Preview**: Show current camera feeds
- **Add/Remove**: Add new cameras or remove existing ones

**FR-CONFIG-CAMERAS-002**: Individual camera configuration
- **RTSP Settings**: Configure camera connection details
- **Detection Zones**: Define areas for object detection
- **Recording Settings**: Configure video recording options
- **Snapshot Settings**: Configure snapshot capture
- **Object Filters**: Configure which objects to detect/track

**FR-CONFIG-CAMERAS-003**: Global Frigate settings
- **MQTT Configuration**: Configure MQTT broker settings
- **Object Detection**: Global detection settings
- **Recording Options**: Global recording configuration
- **Snapshot Options**: Global snapshot settings
- **Performance Settings**: CPU/GPU usage optimization

## 4.8 Settings Sub-tab
**FR-CONFIG-SETTINGS-001**: The system SHALL provide system monitoring and configuration
- **System Status**: Display health of all components
- **Service Status**: Docker containers, MQTT, database status
- **Performance Metrics**: CPU, memory, disk usage
- **Log Access**: View application and error logs
- **Backup/Restore**: System configuration backup

**FR-CONFIG-SETTINGS-002**: Application configuration
- **Tracking Settings**: Configure Erik tracking parameters
- **Alert Settings**: Configure safety alert thresholds
- **Update Intervals**: Configure refresh rates
- **Notification Settings**: Configure alert methods
- **User Interface**: Configure display preferences

---

## 5. User Workflow and Navigation

### 5.1 Primary User Journey
1. **Initial Access**: User opens application → Map tab loads automatically
2. **Live Monitoring**: User monitors Erik's position on yard map
3. **Camera Check**: User switches to Live tab to view camera feeds
4. **Match Review**: User checks Matches tab to review recent detections
5. **Configuration**: User accesses Config tab for system setup/maintenance

### 5.2 Setup Workflow (First-time users)
1. **Camera Configuration**: Configure cameras in Config→Cameras
2. **Photo Upload**: Upload Erik reference photos in Config→Photos
3. **3D Reconstruction**: 
   - Download snapshots in Config→Reconstruct
   - Run COLMAP externally
   - Upload reconstruction files
   - Extract camera poses
4. **Yard Map Generation**: Generate yard map in Config→Yard
5. **Orientation Verification**: Verify camera poses in Config→Orient
6. **Live Monitoring**: Return to Map tab for tracking

### 5.3 Tab Navigation
**FR-NAV-001**: Tab switching SHALL be immediate and preserve state
- **Map Tab**: Always accessible, auto-refreshes when visible
- **Live Tab**: Loads camera feeds when activated
- **Matches Tab**: Shows face recognition match history with real-time updates
- **Config Tab**: Preserves sub-tab selection
- **URL Routing**: Support deep linking to specific tabs

**FR-NAV-002**: Cross-tab integration
- **Map→Matches**: Show recent detections that led to current position
- **Live→Matches**: Review face recognition results from current camera feeds
- **Matches→Config**: Access Photos tab to add reference images
- **Reconstruct→Orient**: "Open in Orient" buttons switch tabs with context
- **Status Messages**: Show integration prompts between tabs
- **Data Sharing**: Share extracted poses between Reconstruct and Orient tabs

---

## 6. Technical Requirements

### 6.1 Performance Requirements
**FR-PERF-001**: Response time requirements
- **Page Load**: <2 seconds initial load
- **Tab Switching**: <500ms transition time
- **Real-time Updates**: <1 second latency for position updates
- **Camera Refresh**: <3 seconds for snapshot updates

**FR-PERF-002**: Scalability requirements
- **Concurrent Users**: Support 5-10 simultaneous users
- **Data Retention**: Store 30 days of position history
- **File Storage**: Support up to 10GB of reference photos and maps
- **Memory Usage**: Optimize for residential hardware deployment

### 6.2 Reliability Requirements
**FR-REL-001**: Availability requirements
- **Uptime Target**: 99% availability during monitoring hours
- **Graceful Degradation**: Continue basic functionality if components fail
- **Error Recovery**: Automatic recovery from temporary failures
- **Backup Systems**: Fallback mechanisms for critical functions

**FR-REL-002**: Data integrity requirements
- **Position Accuracy**: ±1 meter accuracy for Erik's position
- **Detection Confidence**: Minimum 70% confidence for positive identification
- **False Positive Rate**: <5% false positive Erik detections
- **Data Persistence**: Ensure no data loss during system updates

### 6.3 Security Requirements
**FR-SEC-001**: Access control
- **Local Network Only**: Restrict access to local network
- **Authentication**: Optional user authentication for configuration
- **HTTPS**: Secure connections for sensitive operations
- **Data Privacy**: Keep all tracking data local

### 6.4 Integration Requirements
**FR-INT-001**: External system integration
- **Frigate NVR**: Real-time integration with Frigate for camera feeds
- **MQTT Broker**: Reliable message handling for detection events
- **COLMAP**: Support for COLMAP reconstruction file formats
- **CompreFace**: Integration for face recognition capabilities

---

## 7. Data Models and Storage

### 7.1 Core Data Entities
```
Erik Position:
- timestamp
- x_coordinate, y_coordinate (map coordinates)
- camera_source
- confidence_score
- detection_type (face_recognition, object_detection)

Camera Configuration:
- camera_id
- name, location
- rtsp_url
- pose (position, rotation)
- calibration_status

Detection Event:
- timestamp
- camera_id
- bounding_box_coordinates
- object_type
- confidence_score
- match_status

Yard Map:
- map_image_path
- coordinate_system
- scale_factor
- boundaries
- generation_timestamp
```

### 7.2 File Storage Structure
```
/erik_images/           # Reference photos
/reconstruction/        # COLMAP files and extracted poses
/meshes/               # 3D reconstruction files
/yard_maps/            # Generated yard maps
/camera_snapshots/     # Downloaded snapshots
/logs/                 # Application logs
```

---

## 8. Error Handling and Edge Cases

### 8.1 Camera Failures
**FR-ERROR-001**: Handle camera disconnections
- **Detection**: Monitor camera health status
- **User Notification**: Alert when cameras go offline
- **Graceful Degradation**: Continue tracking with available cameras
- **Recovery**: Automatic reconnection when cameras return

### 8.2 Detection Failures
**FR-ERROR-002**: Handle detection failures
- **No Detection Scenario**: Clear messaging when Erik not detected
- **False Positives**: Provide confidence scores and manual review
- **Multiple Detections**: Handle multiple person detections
- **Occlusion**: Account for partial visibility

### 8.3 System Resource Issues
**FR-ERROR-003**: Handle resource constraints
- **High CPU Usage**: Throttle processing if system overloaded
- **Memory Issues**: Implement memory cleanup and optimization
- **Disk Space**: Monitor and alert on low disk space
- **Network Issues**: Handle network connectivity problems

---

## 9. Testing Requirements

### 9.1 Functional Testing
- **End-to-end user workflows**
- **Tab navigation and state management**
- **Real-time position tracking accuracy**
- **Camera feed reliability**
- **File upload and processing**

### 9.2 Performance Testing
- **Load testing with multiple concurrent users**
- **Memory usage optimization**
- **Real-time update latency**
- **Large file upload handling**

### 9.3 Integration Testing
- **Frigate NVR integration**
- **MQTT message handling**
- **COLMAP file processing**
- **Cross-tab data sharing**

---

## 10. Deployment and Infrastructure

### 10.1 Hardware Requirements
- **Minimum**: Raspberry Pi 4 8GB or equivalent
- **Recommended**: x86_64 system with 16GB RAM
- **Storage**: 64GB+ SSD for application and data
- **Network**: Gigabit Ethernet for camera feeds

### 10.2 Software Dependencies
- **Docker and Docker Compose**
- **Python 3.9+ with Flask**
- **Node.js for frontend build**
- **Frigate NVR**
- **MQTT Broker (Mosquitto)**
- **CompreFace (optional)**

### 10.3 Installation Process
1. **Environment Setup**: Configure Docker environment
2. **Service Deployment**: Deploy all required containers
3. **Camera Configuration**: Connect and configure cameras
4. **Initial Calibration**: Complete setup workflow
5. **Testing**: Verify all functionality

---

## 11. Future Enhancements

### 11.1 Mobile Application
- **Native mobile app** for remote monitoring
- **Push notifications** for safety alerts
- **Simplified mobile interface**

### 11.2 Advanced Analytics
- **Movement pattern analysis**
- **Activity recognition** (playing, sitting, running)
- **Heat maps** of frequently visited areas
- **Daily/weekly movement reports**

### 11.3 Additional Safety Features
- **Geofencing** with customizable boundaries
- **Time-based alerts** (nap time monitoring)
- **Weather integration** for outdoor safety
- **Multiple child tracking**

### 11.4 Integration Expansions
- **Smart home integration** (Home Assistant)
- **Cloud backup options**
- **Third-party camera support**
- **Advanced AI models** for behavior recognition

---

## Conclusion

The Toddler Tracker Application provides a comprehensive solution for monitoring child safety through advanced computer vision and spatial tracking. The tabbed interface design ensures ease of use while providing powerful configuration and monitoring capabilities. The system balances real-time performance requirements with user-friendly operation, making it suitable for deployment in residential environments by non-technical users.

The modular architecture allows for future enhancements while maintaining core safety monitoring functionality. Emphasis on local data processing and storage ensures privacy while providing reliable, low-latency tracking capabilities.