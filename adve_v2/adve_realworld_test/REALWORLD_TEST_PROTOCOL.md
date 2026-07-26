# ADVE Real-World Enterprise Testing Protocol

This document outlines the testing protocol, domain matrix, acceptance criteria, and failure analysis rules for validating ADVE on enterprise-grade video datasets prior to commercial licensing.

---

## 1. Five-Domain Validation Matrix

| Domain ID | Target Environment | Key Challenges |
| :--- | :--- | :--- |
| **`urban_surveillance`** | Street cameras, plazas, pedestrian walkways | Dense crowds, frequent occlusions, camera jitter |
| **`retail_indoor`** | Store interiors, office rooms | Illumination changes, glass reflections, close proximity |
| **`traffic_monitoring`** | Highways, intersections, toll plazas | Fast object speeds, scale variations |
| **`low_light`** | Nighttime security, parking garages | High sensor noise, low contrast, IR mode shifts |
| **`ego_motion`** | Handheld cameras, drones, bodycams | Rapid background movement, pan/zoom motion |

---

## 2. Enterprise Acceptance Criteria

| Criterion | Commercial Threshold | Rationale |
| :--- | :--- | :--- |
| **Overall Mean CosSim** | $\ge 0.9850$ (or $\ge 0.9400$ real-world) | Ensures search recall parity with full CLIP |
| **Overall Min CosSim** | $\ge 0.9200$ (or $\ge 0.8500$) | Prevents catastrophic failure on worst-case frames |
| **Per-Domain Mean** | $\ge 0.9400$ | Guarantees reliability across all target verticals |
| **Encoder Savings** | $\ge 70\%$ | Proves computational cost reduction |
| **Effective FPS** | $\ge 30.0\text{ FPS}$ | Real-time processing minimum |
| **Frames Below 0.85** | $\le 1.0\%$ | Extreme error rate cap |

---

## 3. Failure Analysis & Remediation Rules

* **If Mean CosSim $< 0.9400$**: Re-run `generate_training_data.py` on clips from the failing domain and fine-tune `train_reconstructor.py` for 15 epochs.
* **If Min CosSim $< 0.8500$**: Enable confidence gating in `pipeline.py` (if YOLO confidence drops $< 0.4$, force CLIP anchor refresh).
* **If Savings $< 70\%$**: Adjust spatial delta threshold (`SPATIAL_THRESHOLD = 0.25`) to prevent over-triggering anchor refreshes.
