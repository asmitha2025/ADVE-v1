import sys
import os
import subprocess
import json

def run_isolated(mode):
    cmd = [
        sys.executable, "-c", f"""
import os, sys, psutil, time, gc, torch, json
sys.path.append('.')
from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

video = r'C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Vodra/North.mp4'

if '{mode}' == 'normal':
    p = ADVEEnterprisePipeline(reconstructor_path='training/checkpoints/reconstructor_v3.pt', device='cpu', use_ego_motion=False, use_ema=False)
    p.process_video(video, max_frames=100, no_validation=True)
else:
    p = ADVEEnterprisePipeline(reconstructor_path='training/checkpoints/reconstructor_v3.pt', device='cpu', use_ego_motion=True, use_ema=True)
    p.process_video(video, max_frames=100, no_validation=True)

mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
print(json.dumps({{"mode": "{mode}", "mem_mb": round(mem_mb, 1)}}))
"""
    ]
    res = subprocess.check_output(cmd, cwd=os.path.abspath("c:/Users/harih/OneDrive/Documents/codex try/adve/adve_v2")).decode("utf-8")
    for line in res.strip().split("\n"):
        if line.startswith("{") and "mode" in line:
            return json.loads(line)
    return {}

def main():
    print("=========================================================")
    print("      STANDALONE ISOLATED PROCESS MEMORY AUDIT           ")
    print("=========================================================")
    res_normal = run_isolated("normal")
    res_adve = run_isolated("adve")

    print(f"Normal Model Standalone RAM : {res_normal.get('mem_mb')} MB")
    print(f"ADVE Model Standalone RAM   : {res_adve.get('mem_mb')} MB")
    print("=========================================================")

if __name__ == "__main__":
    main()
