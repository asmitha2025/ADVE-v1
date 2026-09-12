"""
ADVE Enterprise — Offline License Key Generator
Generates RSA-2048 key pairs and issues signed, base64-encoded enterprise license tokens.

Usage:
  python license_generator.py --customer "Tata Elxsi" --tier evaluation --max-streams 5 --days 90
  python license_generator.py --customer "Hikvision India" --tier edge --max-streams 50 --days 365
  python license_generator.py --customer "L&T Defense" --tier enterprise --max-streams 999999 --days 365
"""

import os
import sys
import json
import base64
import argparse
from datetime import datetime, timedelta
import rsa

KEYS_DIR = os.path.join(os.path.dirname(__file__), "..", "keys")
PUB_KEY_PATH = os.path.join(KEYS_DIR, "public_key.pem")
PRIV_KEY_PATH = os.path.join(KEYS_DIR, "private_key.pem")


def ensure_keys_exist(bits: int = 2048):
    os.makedirs(KEYS_DIR, exist_ok=True)
    if not os.path.exists(PUB_KEY_PATH) or not os.path.exists(PRIV_KEY_PATH):
        print(f"[+] Generating new RSA {bits}-bit key pair in {KEYS_DIR}...")
        pub_key, priv_key = rsa.newkeys(bits)
        with open(PUB_KEY_PATH, "wb") as f:
            f.write(pub_key.save_pkcs1("PEM"))
        with open(PRIV_KEY_PATH, "wb") as f:
            f.write(priv_key.save_pkcs1("PEM"))
        print("[+] Keys successfully generated.")


def load_private_key() -> rsa.PrivateKey:
    ensure_keys_exist()
    with open(PRIV_KEY_PATH, "rb") as f:
        return rsa.PrivateKey.load_pkcs1(f.read())


def load_public_key() -> rsa.PublicKey:
    ensure_keys_exist()
    with open(PUB_KEY_PATH, "rb") as f:
        return rsa.PublicKey.load_pkcs1(f.read())


def generate_license(
    customer: str,
    tier: str = "evaluation",
    max_streams: int = 5,
    max_gpus: int = 1,
    days: int = 90
) -> str:
    priv_key = load_private_key()
    expiry_date = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")

    payload = {
        "customer": customer,
        "tier": tier.lower(),
        "max_streams": max_streams,
        "max_gpus": max_gpus,
        "issued_at": datetime.utcnow().strftime("%Y-%m-%d"),
        "expiry": expiry_date
    }

    raw_json = json.dumps(payload, sort_keys=True).encode("utf-8")

    # RSA signature over SHA-256 digest of payload
    signature = rsa.sign(raw_json, priv_key, "SHA-256")

    # License token format: base64(json_bytes) . base64(signature_bytes)
    json_b64 = base64.b64encode(raw_json).decode("utf-8")
    sig_b64 = base64.b64encode(signature).decode("utf-8")

    license_token = f"{json_b64}.{sig_b64}"
    return license_token


def verify_license_token(token: str) -> dict:
    pub_key = load_public_key()
    try:
        json_b64, sig_b64 = token.strip().split(".")
        raw_json = base64.b64decode(json_b64.encode("utf-8"))
        signature = base64.b64decode(sig_b64.encode("utf-8"))

        # Verify signature
        rsa.verify(raw_json, signature, pub_key)
        data = json.loads(raw_json.decode("utf-8"))

        expiry_dt = datetime.strptime(data["expiry"], "%Y-%m-%d")
        if datetime.utcnow() > expiry_dt:
            data["valid"] = False
            data["reason"] = "License expired"
        else:
            data["valid"] = True
            data["reason"] = "License valid"

        return data
    except Exception as e:
        return {"valid": False, "reason": f"Verification failed: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(description="ADVE License Generator")
    parser.add_argument("--customer", type=str, required=True, help="Customer name")
    parser.add_argument("--tier", type=str, default="evaluation", choices=["evaluation", "edge", "enterprise"])
    parser.add_argument("--max-streams", type=int, default=5, help="Max RTSP streams permitted")
    parser.add_argument("--max-gpus", type=int, default=1, help="Max GPUs permitted")
    parser.add_argument("--days", type=int, default=90, help="License validity duration in days")
    parser.add_argument("--out", type=str, default=None, help="Output license file path")

    args = parser.parse_args()

    token = generate_license(
        customer=args.customer,
        tier=args.tier,
        max_streams=args.max_streams,
        max_gpus=args.max_gpus,
        days=args.days
    )

    print("\n========================================================")
    print("               ADVE LICENSE GENERATED                   ")
    print("========================================================")
    print(f"Customer:    {args.customer}")
    print(f"Tier:        {args.tier.upper()}")
    print(f"Max Streams: {args.max_streams}")
    print(f"Max GPUs:    {args.max_gpus}")
    print(f"Validity:    {args.days} days")
    print("--------------------------------------------------------")
    print("License Key Token:\n")
    print(token)
    print("========================================================")

    # Verification self-check
    verified = verify_license_token(token)
    print(f"Self-Verification Status: {verified}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(token)
        print(f"[+] License saved to {args.out}")


if __name__ == "__main__":
    main()
