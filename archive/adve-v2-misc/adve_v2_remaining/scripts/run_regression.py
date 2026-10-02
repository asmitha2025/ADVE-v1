import os
import sys
import json
import subprocess

def run_step(step_name, cmd):
    print(f"\n---> {step_name}...")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[FAIL] {step_name} FAILED!")
        sys.exit(1)
    print(f"[OK] {step_name} PASSED")

def main():
    python_exe = sys.executable
    print("=========================================================")
    print("   ADVE Regression & Verification Test Runner")
    print("=========================================================")

    # 1. Sanity Check
    run_step("1. Pre-Flight Sanity Check", f'"{python_exe}" scripts/sanity_check.py')

    # 2. Unit Tests
    run_step("2. Unit Test Suite", f'"{python_exe}" scripts/run_unit_tests.py')

    # 3. Integration Audit Test
    audit_out = "results/regression_audit.json"
    audit_cmd = f'"{python_exe}" -c "from adve_realworld_test.scripts.enterprise_audit import run_enterprise_audit; run_enterprise_audit(\'adve_realworld_test/datasets/sample_manifest.json\', \'training/checkpoints/reconstructor_v3.pt\', \'{audit_out}\', device=\'cpu\', max_frames_per_video=200)"'
    run_step("3. Integration Audit Execution", audit_cmd)

    # 4. Score Validation
    print("\n---> 4. Score Validation...")
    if not os.path.exists(audit_out):
        print(f"[FAIL] Audit output '{audit_out}' missing!")
        sys.exit(1)

    with open(audit_out, "r") as f:
        data = json.load(f)

    score = data.get("overall_score", 0.0)
    fails = data.get("critical_fails", 99)
    metrics = data.get("metrics", {})
    mean_sim = metrics.get("mean_cosine_sim", 0.0)
    min_sim  = metrics.get("min_cosine_sim", 0.0)

    print(f"   Audit Score : {score:.1f} / 100")
    print(f"   Mean CosSim : {mean_sim:.4f}")
    print(f"   Min CosSim  : {min_sim:.4f}")
    print(f"   Crit Fails  : {fails}")

    assert fails == 0, f"Critical failures found ({fails})"
    assert score >= 75.0, f"Audit score fell below threshold ({score:.1f} < 75.0)"
    assert mean_sim >= 0.97, f"Mean similarity regressed ({mean_sim:.4f} < 0.97)"

    print("[OK] Score Validation PASSED")

    print("\n=========================================================")
    print(" [PASSED] ALL REGRESSION TESTS PASSED — Safe to Push to Git!")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
