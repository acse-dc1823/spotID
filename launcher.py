import webbrowser
import threading
import time
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from interface.app import app
from waitress import serve

def open_browser():
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://127.0.0.1:5000')

def main():
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    # Try different ports if 5000 is in use
    ports = [5000, 5001, 5002, 5003, 5004]
    for port in ports:
        try:
            serve(app, host='127.0.0.1', port=port)
            break
        except OSError:
            if port == ports[-1]:
                raise
            continue

if __name__ == '__main__':
    main()
