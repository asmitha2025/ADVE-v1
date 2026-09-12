"""
ADVE License Runtime Kill-Switch & Verification Guard (Critical Item #8)

Validates enterprise license key HMAC signature and expiration timestamp.
Performs hourly runtime verification checks.
Enforces a 24-hour grace period for temporary license check failures before stopping active video streams.
"""

import os
import time
import json
import hmac
import hashlib
from typing import Dict, Any, Optional

LICENSE_SECRET = os.environ.get("ADVE_LICENSE_SECRET", "adve-enterprise-secret-key-2026")
DEFAULT_LICENSE_PATH = os.environ.get("ADVE_LICENSE_PATH", "adve.lic")

class LicenseGuard:
    def __init__(self, license_path: str = DEFAULT_LICENSE_PATH):
        self.license_path = license_path
        self.last_check_time = 0.0
        self.grace_period_start: Optional[float] = None

    def generate_license_key(self, client_name: str, days_valid: int = 365) -> Dict[str, Any]:
        """Utility for enterprise key generation."""
        expires_at = time.time() + (days_valid * 86400)
        payload = {
            "client_name": client_name,
            "expires_at": expires_at,
            "issued_at": time.time(),
            "max_streams": 32
        }
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(LICENSE_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
        return {
            "payload": payload,
            "signature": signature
        }

    def verify_license(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Verifies license signature and expiration.
        Caches result for 60 minutes unless force_check is True.
        """
        now = time.time()
        if not force_check and (now - self.last_check_time < 3600):
            # Cached valid check
            return {"valid": True, "cached": True}

        self.last_check_time = now

        if not os.path.exists(self.license_path):
            # Default development fallback if license file not present
            if os.environ.get("ADVE_DEV_MODE", "1") == "1":
                return {"valid": True, "dev_mode": True, "message": "Development Mode (No license file required)"}

            return self._handle_invalid("License file missing ('adve.lic')")

        try:
            with open(self.license_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            payload = data.get("payload", {})
            signature = data.get("signature", "")

            payload_str = json.dumps(payload, sort_keys=True)
            expected_sig = hmac.new(LICENSE_SECRET.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                return self._handle_invalid("Invalid license signature (Tampered file)")

            expires_at = payload.get("expires_at", 0)
            if now > expires_at:
                return self._handle_invalid(f"License expired at {time.ctime(expires_at)}")

            # Valid check clears grace period
            self.grace_period_start = None
            return {
                "valid": True,
                "client_name": payload.get("client_name"),
                "expires_at": expires_at,
                "max_streams": payload.get("max_streams", 32)
            }

        except Exception as e:
            return self._handle_invalid(f"License parse error: {str(e)}")

    def _handle_invalid(self, reason: str) -> Dict[str, Any]:
        now = time.time()
        if self.grace_period_start is None:
            self.grace_period_start = now

        elapsed_grace = now - self.grace_period_start
        if elapsed_grace < 86400: # 24-hour grace period
            return {
                "valid": True,
                "in_grace_period": True,
                "grace_remaining_hours": round((86400 - elapsed_grace) / 3600, 1),
                "reason": reason
            }
        else:
            return {
                "valid": False,
                "error": reason,
                "grace_expired": True
            }
