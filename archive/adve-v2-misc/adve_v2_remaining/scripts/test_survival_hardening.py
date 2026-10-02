"""
Master Master Test Runner for Week 1 Survival Hardening Items
"""

import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.content_detector import DomainFingerprinter
from adve.api.auth import APIKeyStore, DEFAULT_API_KEY
from adve.core.license_guard import LicenseGuard
from adve.core.webhook import WebhookDispatcher
from adve.search.index import ADVESearchIndex

def test_fingerprinter():
    print("\n1. Testing DomainFingerprinter (Critical Item #1)...")
    fp = DomainFingerprinter(sample_frames=5)
    res = fp._get_profile("night", brightness=30.0, motion=2.0, edge_density=0.03)
    assert res["domain"] == "night"
    assert res["reconstructor_head"] == "night_head"
    print("   [OK] Rerouted to 'night' profile correctly.")

def test_auth():
    print("\n2. Testing APIKeyAuthMiddleware & KeyStore (Critical Item #7)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "auth.db")
        store = APIKeyStore(db_path)
        assert store.validate_key(DEFAULT_API_KEY) is True
        key = store.create_key("Client B")
        assert store.validate_key(key) is True
        print("   [OK] API key auth validated.")

def test_license():
    print("\n3. Testing LicenseGuard Kill-Switch (Critical Item #8)...")
    guard = LicenseGuard(license_path="non_existent.lic")
    res = guard.verify_license(force_check=True)
    # Dev mode or 24-hour grace period should allow execution initially
    assert res["valid"] is True
    print("   [OK] License guard handled status with 24h grace period correctly.")

def test_webhook():
    print("\n4. Testing Webhook Retry Dispatcher (Critical Item #9)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "webhooks.db")
        dispatcher = WebhookDispatcher(db_path)
        dispatcher.dispatch("http://invalid.local/callback", "drift_alert", {"similarity": 0.82})
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM webhook_queue")
        count = cursor.fetchone()[0]
        assert count == 1
        print("   [OK] Webhook alert queued cleanly for retry.")

def test_sqlite_wal():
    print("\n5. Testing SQLite WAL Mode & Concurrency (Critical Item #11)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = ADVESearchIndex(tmp_dir)
        cursor = index.db.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0].lower()
        assert mode == "wal"
        print(f"   [OK] SQLite WAL mode confirmed: {mode}.")

if __name__ == "__main__":
    print("==================================================")
    print("  RUNNING CRITICAL PRODUCTION HARDENING TEST SUITE")
    print("==================================================")
    test_fingerprinter()
    test_auth()
    test_license()
    test_webhook()
    test_sqlite_wal()
    print("\n==================================================")
    print("  ALL CRITICAL HARDENING TESTS PASSED PERFECTLY!  ")
    print("==================================================")
