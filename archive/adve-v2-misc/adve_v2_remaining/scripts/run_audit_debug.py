import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve_realworld_test.scripts.enterprise_audit import run_enterprise_audit

print("Starting Enterprise Audit debug run...")
run_enterprise_audit(
    'adve_realworld_test/datasets/sample_manifest.json',
    'training/checkpoints/reconstructor_v3.pt',
    'results/my_full_audit.json',
    device='cpu',
    max_frames_per_video=150
)
