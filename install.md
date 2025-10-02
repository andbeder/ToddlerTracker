# Toddler Tracker Application - Installation Guide

## Overview

This guide will walk you through installing and setting up the complete Toddler Tracker Application stack, including all required dependencies and services. The system uses Docker containers for easy deployment and management.

## Prerequisites

### Hardware Requirements
- **Minimum**: x86_64 system with 8GB RAM, 64GB storage
- **Recommended**: x86_64 system with 16GB+ RAM, 256GB+ SSD storage
- **GPU**: NVIDIA GPU recommended for optimal performance (optional)
- **Network**: Gigabit Ethernet connection for camera feeds
- **Cameras**: IP cameras with RTSP streams (4 cameras recommended)

### Software Requirements
- **Operating System**: Ubuntu 20.04 LTS or newer, Debian 11+, or other Docker-compatible Linux distribution
- **Internet Connection**: Required for downloading Docker images and dependencies

---

## Installation Steps

## Step 1: Install Docker and Docker Compose

### Ubuntu/Debian Installation

```bash
# Update package index
sudo apt update && sudo apt upgrade -y

# Install prerequisites
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to docker group (requires logout/login to take effect)
sudo usermod -aG docker $USER

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### Alternative: Docker Desktop (Windows/macOS)

1. Download Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Install following the platform-specific instructions
3. Ensure Docker Desktop is running before proceeding

### Verify Installation

```bash
# Check Docker version
docker --version
docker-compose --version

# Test Docker installation
docker run hello-world
```

---

## Step 2: Install NVIDIA Docker Support (Optional - For GPU Acceleration)

### For NVIDIA GPU Users

```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install -y nvidia-driver-470  # or latest version

# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt update
sudo apt install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

---

## Step 3: Download and Setup the Application

### Clone Repository

```bash
# Clone the repository
git clone <repository-url> toddler-tracker
cd toddler-tracker

# Or if starting from scratch, create directory structure
mkdir -p toddler-tracker
cd toddler-tracker
```

---

## Step 4: Configure Environment Variables

### Create .env File

```bash
cat > .env << 'EOF'
# CompreFace Configuration
registry=exadel/
POSTGRES_VERSION=latest
ADMIN_VERSION=1.0.0
API_VERSION=1.0.0
FE_VERSION=1.0.0
CORE_VERSION=1.0.0

# Database Configuration
postgres_username=postgres
postgres_password=postgres
postgres_db=frs

# CompreFace Settings
enable_email_server=false
email_host=smtp.gmail.com
email_username=
email_from=
email_password=
save_images_to_db=true
compreface_admin_java_options=-Xmx8g
compreface_api_java_options=-Xmx8g
max_file_size=5MB
max_request_size=10MB
connection_timeout=10000
read_timeout=60000
uwsgi_processes=2
uwsgi_threads=1
EOF
```

---


## Step 5: Start the Application

### Build and Start Services

```bash
# Pull required Docker images
docker compose pull

# Build custom images
docker compose build

# Start all services
docker compose up -d

# Check service status
docker compose ps
```


---

## Step 6: Initial Configuration

### Access Web Interfaces

- **Frigate**: http://localhost:5000
- **CompreFace**: http://localhost:8000
- **Toddler Image Manager**: http://localhost:9000
- **Double-Take**: http://localhost:3000
- **Home Assistant**: http://localhost:8123

### CompreFace Setup

1. Open http://localhost:8000
2. Create an admin account
3. Create a new application called "ToddlerTracker"
4. Create a recognition service
5. Note the API key for the next step

### Upload Toddler Reference Images

1. Open Toddler Tracker App at http://localhost:9000
2. Navigate to the **Images** tab
3. Upload high-quality photos of your toddler(s) from different angles
4. Ensure photos are well-lit and show toddler's face clearly
5. Upload photos one-at-a-time for best feedback on recognition errors

---

## Step 7: Camera Integration

### Prepare your cameras for use

1. Specify a IP address with a subnet mask for your ethernet controller so you can assign a static IP to the controlers
 - Instructions for this depend on your linux distribution, the example below works temporarily for Ubantu

```bash
ip address

#Note the name of the network connected to your PoE switch, in this next example it is 'enp16s0'

# This will temporarily assign the your computer the IP of 192.168.0.1 with a subnet mask of 255.255.255.0

sudo ip addr add 192.168.0.1/24 dev enp16s0
sudo ip link set enp16s0 up

```

2. Plug each camera into your switch individually and use the camera's desktop software to assign IP addresses to them
 - In the following examples I have assigned my cameras to use IP's 192.168.0.101, 102, etc.
 - Note the Stream URL for your camera under their RTSP settings, this is important for connecting Frigate.
 - Assign an admin password to be used later to access the RTSP streams


### Add Your Cameras to Frigate

1. Navigate to the **Images** tab
2. Click **Add Camera**
3. Add the camera settings
   - Common name (no spaces)
   - IP Address
   - Stream Address (i.e. for Reolink cameras this is `/h264Preview_01_main`)
   - Username and password (username is usually `admin`, password was set during step 2 above)

1. Edit `frigate/config/config.yaml`
2. Replace example camera configurations with your actual camera details:
   - IP addresses
   - RTSP URLs
   - Usernames and passwords

### Test Camera Feeds

1. Open Frigate web interface at http://localhost:5000
2. Verify all cameras are streaming
3. Check that person detection is working
4. Adjust detection zones if needed


---

## Step 8: 3D Reconstruction Setup (Optional)

### Install COLMAP for 3D Reconstruction

```bash
# Option 1: Install COLMAP locally
sudo apt install colmap

# Option 2: Use COLMAP Docker image
docker pull colmap/colmap

# Create reconstruction workspace
mkdir -p reconstruction/workspace
```

### Prepare for Camera Calibration

1. Use Toddler Image Manager to download camera snapshots
2. Process snapshots with COLMAP to generate 3D reconstruction
3. Upload reconstruction files back to Toddler Image Manager
4. Generate yard map for position tracking

---

## Troubleshooting

### Common Issues

#### Docker Permission Issues
```bash
# Add user to docker group and restart
sudo usermod -aG docker $USER
newgrp docker
```

#### Port Conflicts
```bash
# Check what's using a port
sudo netstat -tlnp | grep :5000

# Stop conflicting services
sudo systemctl stop <service-name>
```

#### CompreFace Memory Issues
```bash
# Increase Java heap size in .env file
compreface_admin_java_options=-Xmx16g
compreface_api_java_options=-Xmx16g

# Restart services
docker-compose restart compreface-admin compreface-api
```

#### Camera Connection Issues
```bash
# Test RTSP stream directly
ffplay rtsp://admin:password@192.168.1.100:554/h264Preview_01_main

# Check network connectivity
ping 192.168.1.100
```

#### GPU Support Issues
```bash
# Verify NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi

# Check GPU usage in containers
docker stats
```

### Log Analysis

```bash
# Check all logs
docker-compose logs -f

# Check specific service logs
docker-compose logs -f frigate
docker-compose logs -f compreface-api
docker-compose logs -f hybrid-Toddler-tracker

# Check system resources
docker stats
htop
```

### Performance Optimization

#### For Low-End Hardware
```bash
# Reduce detection resolution and FPS in frigate config
detect:
  width: 640
  height: 480
  fps: 2

# Limit CompreFace resources
compreface_admin_java_options=-Xmx4g
compreface_api_java_options=-Xmx4g
```

#### For High-End Hardware
```bash
# Increase detection quality
detect:
  width: 1920
  height: 1080
  fps: 10

# Enable GPU acceleration in Frigate config
detectors:
  tensorrt:
    type: tensorrt
    device: 0
```

---

## Maintenance

### Regular Tasks

#### Daily
- Check service status: `docker-compose ps`
- Monitor system resources: `htop`, `df -h`

#### Weekly
- Review detection accuracy in Toddler Image Manager
- Check storage usage: `du -sh frigate/media/*`
- Update face recognition training data if needed

#### Monthly
- Update Docker images: `docker-compose pull && docker-compose up -d`
- Backup configuration files
- Clean old recordings: Frigate auto-cleanup should handle this

### Backup Strategy

```bash
# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz \
  frigate/config/ \
  double-take/ \
  mosquitto/config/ \
  nginx/ \
  Toddler_images/ \
  .env

# Backup CompreFace database
docker-compose exec compreface-postgres-db pg_dump -U postgres frs > compreface-backup.sql
```

### Updates

```bash
# Update application
git pull origin main

# Update Docker images
docker-compose pull

# Rebuild custom images
docker-compose build

# Restart with new images
docker-compose up -d
```

---

## Security Considerations

### Production Deployment

1. **Change Default Passwords**: Update all default passwords in configuration files
2. **Enable Authentication**: Configure authentication for web interfaces
3. **Network Security**: Use firewall rules to restrict access
4. **HTTPS**: Configure SSL certificates for web interfaces
5. **Regular Updates**: Keep Docker images and host system updated

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 5000/tcp  # Frigate
sudo ufw allow 8000/tcp  # CompreFace
sudo ufw allow 9000/tcp  # Toddler Image Manager
```

---

## Support

### Getting Help

1. **Check Logs**: Always check Docker logs first
2. **Documentation**: Refer to individual component documentation:
   - [Frigate Documentation](https://docs.frigate.video/)
   - [CompreFace Documentation](https://github.com/exadel-inc/CompreFace)
   - [Double-Take Documentation](https://github.com/jakowenko/double-take)
3. **Community**: Check project issues and discussions
4. **System Requirements**: Ensure your hardware meets minimum requirements

### Useful Commands

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart frigate

# View real-time logs
docker-compose logs -f

# Check resource usage
docker stats

# Clean up unused images
docker system prune -a

# Backup database
docker-compose exec compreface-postgres-db pg_dump -U postgres frs > backup.sql
```

---

This installation guide provides a complete setup for the Toddler Tracker Application. Follow each step carefully and refer to the troubleshooting section if you encounter any issues. The system requires some initial configuration and training to work optimally, but once set up, it provides comprehensive toddler monitoring capabilities.
