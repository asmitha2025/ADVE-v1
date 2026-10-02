import os
import shutil

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=========================================================")
    print("      ADVE ENTERPRISE CODEBASE CLEANUP AUDIT            ")
    print("=========================================================")

    files_to_remove = [
        "INTEGRATE_FIXES.md",
        "README_PHASE1.md",
        "requirements_phase1.txt",
        "benchmark_test.mp4",
        "benchmark_results.json",
        "generate_test_video.py"
    ]

    dirs_to_clean = [
        "data/cctv_client_audit_index",
        "data/cctv_live_audit_index",
        "data/traffic_semantic_search_index",
        "test_idx",
        "adve_v2"
    ]

    for f in files_to_remove:
        path = os.path.join(root, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"   • Removed temporary file: {f}")

    for d in dirs_to_clean:
        path = os.path.join(root, d)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"   • Cleaned temporary dir : {d}")

    print("\n=========================================================")
    print("ADVE ENTERPRISE CODEBASE IS CLEAN & PRODUCTION READY!")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
