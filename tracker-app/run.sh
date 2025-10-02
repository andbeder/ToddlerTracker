#!/bin/bash

# Frigate Camera Manager Startup Script

echo "Frigate Camera Manager"
echo "====================="

# Check if virtual environment should be used
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import flask, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        echo "Please install manually: pip3 install flask pyyaml"
        exit 1
    fi
fi

# Check if Frigate config exists
if [ ! -f "../frigate/config/config.yaml" ]; then
    echo "Warning: Frigate config file not found at ../frigate/config/config.yaml"
    echo "The app will still run but may not function properly"
fi

echo "Starting Flask application..."
echo "Access the web interface at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo

python3 app.py