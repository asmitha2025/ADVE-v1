import os
import sys

# Ensure adve_v2 is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "adve_v2"))

# Import demo and run it
from adve_v2.demo import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
