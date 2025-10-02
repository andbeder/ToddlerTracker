#!/usr/bin/env python3

def parse_rtsp_url(rtsp_url: str) -> dict:
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

if __name__ == '__main__':
    # Test cases
    test_urls = [
        "rtsp://admin:password123@192.168.1.100:554/stream",
        "rtsp://user:pass@10.0.0.5/cam1",
        "rtsp://viewer:secret@camera.local:8554/live",
        "rtsp://test:test123@192.168.1.50",
    ]

    print("Testing RTSP URL parsing function:")
    print("=" * 50)

    for url in test_urls:
        result = parse_rtsp_url(url)
        print(f"URL: {url}")
        print(f"Parsed: {result}")

        # Reconstruct URL to verify
        if result:
            reconstructed = f"rtsp://{result['username']}:{result['password']}@{result['ip_address']}:{result['port']}{result['stream_path']}"
            print(f"Reconstructed: {reconstructed}")
            print(f"Match: {'✓' if reconstructed == url or (url.endswith(result['ip_address']) and result['stream_path'] == '/stream') else '✗'}")
        print()

    print("RTSP URL parsing test complete!")