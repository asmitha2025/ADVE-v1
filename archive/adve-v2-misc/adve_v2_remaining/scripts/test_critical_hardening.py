"""
Test Suite for Critical Technical Debt Hardening Items (#1, #7, #11, #14)
"""

import os
import sys
import tempfile
import sqlite3
import numpy as np
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.content_detector import DomainFingerprinter, ContentDetector
from adve.api.auth import APIKeyStore, APIKeyAuthMiddleware, DEFAULT_API_KEY
from adve.search.index import ADVESearchIndex
from adve.api.server import app

def test_domain_fingerprinter():
    print("\n1. Testing DomainFingerprinter (Critical Item #1)...")
    fingerprinter = DomainFingerprinter(sample_frames=10)
    profile = fingerprinter._get_profile("traffic", brightness=120.0, motion=8.0, edge_density=0.09)
    
    assert profile["domain"] == "traffic"
    assert profile["reconstructor_head"] == "traffic_head"
    assert profile["spatial_threshold"] == 0.38
    print("   [OK] Domain Fingerprinter routed to 'traffic' profile correctly.")

def test_api_key_auth():
    print("\n2. Testing APIKeyAuthMiddleware & Store (Critical Item #7)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "auth_test.db")
        store = APIKeyStore(db_path)
        
        # Test default dev key
        assert store.validate_key(DEFAULT_API_KEY) is True
        print("   [OK] Default dev key validated.")
        
        # Test custom key generation
        new_key = store.create_key("Enterprise Client A")
        assert store.validate_key(new_key) is True
        assert store.validate_key("invalid-key-12345") is False
        print("   [OK] Custom API key generated and validated cleanly.")

def test_sqlite_wal_mode():
    print("\n3. Testing SQLite WAL Mode & Concurrency (Critical Item #11)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = ADVESearchIndex(tmp_dir)
        cursor = index.db.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].lower()
        
        assert journal_mode == "wal"
        print(f"   [OK] SQLite journal_mode is '{journal_mode}'.")

def test_health_endpoint():
    print("\n4. Testing Enhanced Subsystem Health Probes (Critical Item #14)...")
    client = TestClient(app)
    res = client.get("/health")
    
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "subsystems" in data
    assert data["subsystems"]["metadata_db_ping"] is True
    print(f"   [OK] /health endpoint returned healthy status with subsystem diagnostics: {data['subsystems']}")

if __name__ == "__main__":
    print("==================================================")
    print("  TESTING CRITICAL PRODUCTION HARDENING MODULES")
    print("==================================================")
    test_domain_fingerprinter()
    test_api_key_auth()
    test_sqlite_wal_mode()
    test_health_endpoint()
    print("\n==================================================")
    print("  ALL CRITICAL HARDENING TESTS PASSED PERFECTLY!")
    print("==================================================")
