"""
ADVE Enterprise — RSA License Enforcement Middleware
Validates RSA signature, expiry date, tier limits, and maximum stream count.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional
import rsa
from fastapi import HTTPException, status
from api.config import settings

class LicenseException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"License Enforcement Error: {detail}"
        )


class LicenseExpiredException(LicenseException):
    def __init__(self):
        super().__init__("Enterprise license has expired. Contact sales for renewal.")


class LicenseInvalidException(LicenseException):
    def __init__(self, detail: str = "Invalid license signature or key payload."):
        super().__init__(detail)


class StreamLimitExceededException(LicenseException):
    def __init__(self, current: int, max_allowed: int):
        super().__init__(f"Maximum stream limit reached ({current}/{max_allowed}). Upgrade tier to add more streams.")


def load_public_key() -> Optional[rsa.PublicKey]:
    key_path = settings.public_key_path
    if not os.path.exists(key_path):
        return None
    with open(key_path, "rb") as f:
        return rsa.PublicKey.load_pkcs1(f.read())


def verify_license_key(license_key_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Verifies the base64 RSA-signed license payload string.
    Format: base64(json_bytes).base64(signature_bytes)
    """
    key = license_key_str or settings.license_key
    if not key:
        # Development fallback / evaluation default
        return {
            "valid": True,
            "customer": "Development / Evaluation Mode",
            "tier": "evaluation",
            "max_streams": 5,
            "max_gpus": 1,
            "expiry": "2099-12-31",
            "days_remaining": 9999
        }

    try:
        json_b64, sig_b64 = key.strip().split(".")
        raw_json = base64.b64decode(json_b64.encode("utf-8"))
        signature = base64.b64decode(sig_b64.encode("utf-8"))

        pub_key = load_public_key()
        if pub_key:
            rsa.verify(raw_json, signature, pub_key)

        data = json.loads(raw_json.decode("utf-8"))
        expiry_dt = datetime.strptime(data["expiry"], "%Y-%m-%d")
        now = datetime.utcnow()

        days_remaining = (expiry_dt - now).days
        data["days_remaining"] = days_remaining

        if now > expiry_dt:
            data["valid"] = False
            raise LicenseExpiredException()

        data["valid"] = True
        return data

    except (ValueError, rsa.VerificationError, KeyError) as e:
        raise LicenseInvalidException(f"License verification failed: {str(e)}")


def check_stream_limit(active_streams_count: int, license_data: Dict[str, Any]):
    max_streams = license_data.get("max_streams", 5)
    if active_streams_count >= max_streams:
        raise StreamLimitExceededException(active_streams_count, max_streams)
