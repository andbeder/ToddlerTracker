#!/usr/bin/env python3
"""
Test the projection endpoint directly to verify it works.
"""

import requests
import json
import time

url = "http://127.0.0.1:9000/project_camera_to_map"

payload = {
    "camera_name": "garage",
    "map_id": 2,
    "camera_width": 2560,
    "camera_height": 1920,
    "projection_method": "cuda"
}

print("Sending projection request...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

start = time.time()

try:
    response = requests.post(url, json=payload, timeout=180)  # 3 minute timeout
    elapsed = time.time() - start

    print(f"\n✓ Response received in {elapsed:.1f}s")
    print(f"Status code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nResult keys: {list(result.keys())}")
        print(f"Pixel count: {result.get('pixel_count', 'N/A')}")
        print(f"Coverage: {result.get('coverage_percent', 'N/A')}%")
        print(f"Compute time: {result.get('compute_time', 'N/A')}s")
        print(f"Projected pixels length: {len(result.get('projected_pixels', []))}")
        print("\n✓ Endpoint working correctly!")
    else:
        print(f"\n❌ Error: {response.text}")

except requests.exceptions.Timeout:
    elapsed = time.time() - start
    print(f"\n❌ Request timed out after {elapsed:.1f}s")
except requests.exceptions.ConnectionError as e:
    print(f"\n❌ Connection error: {e}")
except Exception as e:
    print(f"\n❌ Error: {e}")
