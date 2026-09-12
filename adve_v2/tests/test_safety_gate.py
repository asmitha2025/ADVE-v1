"""
Unit tests for SafetyGate module (Enterprise Hard Floor Assurance)
"""

import torch
from adve.core.safety_gate import SafetyGate

def test_safety_gate_blocks_garbage():
    gate = SafetyGate(hard_floor=0.88, consecutive_max=3)
    
    # Generate random vectors -> low CosSim
    bad_pred = torch.randn(1, 512)
    anchor = torch.randn(1, 512)
    
    safe_emb, was_forced, sim = gate.check(bad_pred, anchor)
    
    # Verify SafetyGate triggered and emitted anchor as safe fallback
    assert was_forced is True
    assert sim < 0.88
    assert torch.allclose(safe_emb, torch.nn.functional.normalize(anchor, dim=-1))
    print("[PASS] test_safety_gate_blocks_garbage passed!")

def test_safety_gate_passes_high_similarity():
    gate = SafetyGate(hard_floor=0.88, consecutive_max=3)
    
    # High similarity vectors
    anchor = torch.randn(1, 512)
    good_pred = anchor + 0.01 * torch.randn(1, 512)
    
    safe_emb, was_forced, sim = gate.check(good_pred, anchor)
    
    assert was_forced is False
    assert sim >= 0.88
    print("[PASS] test_safety_gate_passes_high_similarity passed!")

def test_safety_gate_forces_full_clip_after_consecutive_drift():
    gate = SafetyGate(hard_floor=0.88, consecutive_max=3)
    
    anchor = torch.randn(1, 512)
    for _ in range(3):
        bad_pred = torch.randn(1, 512)
        gate.check(bad_pred, anchor)
        
    assert gate.should_force_full_clip() is True
    print("[PASS] test_safety_gate_forces_full_clip_after_consecutive_drift passed!")

if __name__ == "__main__":
    test_safety_gate_blocks_garbage()
    test_safety_gate_passes_high_similarity()
    test_safety_gate_forces_full_clip_after_consecutive_drift()
