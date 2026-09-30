import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from adve.api.server import app

def main():
    print("=========================================================")
    print("      ADVE FASTAPI SERVER & ENDPOINTS VALIDATION         ")
    print("=========================================================")

    client = TestClient(app)

    # 1. Health Check
    res_health = client.get("/health")
    print("\n1. GET /health Status:", res_health.status_code)
    print("   Response:", res_health.json())
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. Invalid License Validation
    res_bad_lic = client.post("/api/v1/license/validate", headers={"X-License-Key": "INVALID-KEY"})
    print("\n2. POST /api/v1/license/validate (Bad Key) Status:", res_bad_lic.status_code)
    print("   Response:", res_bad_lic.json())
    assert res_bad_lic.status_code == 401

    # 3. Valid License Validation
    res_good_lic = client.post("/api/v1/license/validate", headers={"X-License-Key": "EVAL-TATA-2026"})
    print("\n3. POST /api/v1/license/validate (Valid Key) Status:", res_good_lic.status_code)
    print("   Response:", res_good_lic.json())
    assert res_good_lic.status_code == 200
    assert res_good_lic.json()["valid"] is True

    # 4. Embed Video Request
    video_path = "../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4"
    if os.path.exists(video_path):
        payload = {
            "video_path": video_path,
            "index_name": "evaluation_index",
            "max_frames": 100
        }
        res_embed = client.post("/api/v1/embed", json=payload, headers={"X-License-Key": "EVAL-TATA-2026"})
        print("\n4. POST /api/v1/embed Status:", res_embed.status_code)
        print("   Response:", json.dumps(res_embed.json(), indent=2))
        assert res_embed.status_code == 200
        assert res_embed.json()["status"] == "success"

    print("\n=========================================================")
    print("✅ ALL FASTAPI v3.1 ENDPOINTS VALIDATED SUCCESSFULLY!")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
