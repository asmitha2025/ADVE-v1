import hashlib
import datetime
import json
import argparse

def generate_license(client_name: str, tier: str = "evaluation", months_valid: int = 3) -> dict:
    """
    Generates a secure SHA-256 HMAC-style license key for ADVE Enterprise Clients.
    
    Args:
        client_name: e.g. "TataElxsi", "SecurityVendor"
        tier: "evaluation" or "enterprise_annual"
        months_valid: duration in months
    """
    now = datetime.datetime.now()
    base_str = f"{client_name}-{tier}-{now.strftime('%Y%m%d%H%M%S')}"
    key_hash = hashlib.sha256(base_str.encode("utf-8")).hexdigest()[:12].upper()
    
    # Format license key: EVAL-TATA-A1B2C3D4E5F6
    prefix = "EVAL" if tier == "evaluation" else "ENT"
    clean_client = client_name.replace(" ", "").upper()[:6]
    license_key = f"{prefix}-{clean_client}-{key_hash}"
    
    expires_date = now + datetime.timedelta(days=30 * months_valid)
    expires_str = expires_date.strftime("%Y-%m-%d")
    
    license_info = {
        "license_key": license_key,
        "client": client_name,
        "tier": tier,
        "issued_at": now.strftime("%Y-%m-%d"),
        "expires": expires_str,
        "max_videos": 100 if tier == "evaluation" else None,
        "features": {
            "enterprise_reconstructor_v3": True,
            "ego_motion_homography": True,
            "frame_structure_gating": True,
            "fastapi_server": True,
            "python_sdk": True
        }
    }
    return license_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADVE Enterprise License Key Generator")
    parser.add_argument("--client", type=str, default="TataElxsi", help="Client name")
    parser.add_argument("--tier", type=str, choices=["evaluation", "enterprise_annual"], default="evaluation")
    parser.add_argument("--months", type=int, default=3, help="Validity period in months")
    parser.add_argument("--out", type=str, default=None, help="Save to JSON file")
    
    args = parser.parse_args()
    lic = generate_license(args.client, args.tier, args.months)
    
    print("\n=========================================================")
    print("      ADVE ENTERPRISE LICENSE KEY GENERATED              ")
    print("=========================================================")
    print(f"License Key : {lic['license_key']}")
    print(f"Client      : {lic['client']}")
    print(f"Tier        : {lic['tier']}")
    print(f"Expires     : {lic['expires']}")
    print(f"Max Videos  : {lic['max_videos']}")
    print("---------------------------------------------------------")
    
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(lic, f, indent=2)
        print(f"Saved license record to: {args.out}")
    print("=========================================================\n")
