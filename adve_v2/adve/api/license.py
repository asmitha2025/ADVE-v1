import os
import time
import datetime
from typing import Optional, Dict, Any

VALID_EVAL_PREFIXES = ["EVAL-", "ENT-", "ADVE-"]

def validate_license_key(license_key: Optional[str]) -> Dict[str, Any]:
    """
    Validates license key header (X-License-Key) or environment variable (ADVE_LICENSE_KEY).
    """
    env_key = os.environ.get("ADVE_LICENSE_KEY")
    key_to_check = license_key or env_key
    
    if not key_to_check:
        return {
            "valid": False,
            "error": "Missing license key. Provide header 'X-License-Key' or set ADVE_LICENSE_KEY environment variable.",
            "status_code": 401
        }

    key_upper = key_to_check.strip().upper()
    
    # Check key format
    is_valid_prefix = any(key_upper.startswith(prefix) for prefix in VALID_EVAL_PREFIXES)
    if not is_valid_prefix or len(key_upper) < 10:
        return {
            "valid": False,
            "error": f"Invalid license key format: '{key_to_check}'",
            "status_code": 401
        }

    # Standard trial / evaluation info
    client_name = "Enterprise Client"
    if "TATA" in key_upper:
        client_name = "Tata Elxsi Evaluation"
    
    return {
        "valid": True,
        "license_key": key_upper,
        "client": client_name,
        "tier": "evaluation" if key_upper.startswith("EVAL") else "enterprise_annual",
        "expires": "2026-12-31",
        "max_videos": 100
    }
