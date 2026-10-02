# ADVE Real-World Testing Framework

This package provides a multi-domain validation framework to benchmark ADVE on enterprise-grade video datasets.

---

## 🚀 Quick-Start Execution

### Step A: Pre-Flight Sanity Check
```bash
python adve_realworld_test/scripts/sanity_check.py --reconstructor training/checkpoints/reconstructor_v2.pt --device cuda
```

### Step B: Generate Dataset Manifest
```bash
python adve_realworld_test/scripts/prepare_datasets.py --generate-manifest /path/to/videos --output datasets/my_manifest.json
```

### Step C: Run Multi-Domain Benchmark
```bash
python adve_realworld_test/scripts/unified_benchmark.py --manifest adve_realworld_test/datasets/sample_manifest.json --reconstructor training/checkpoints/reconstructor_v2.pt --output results/realworld_report.json --device cuda
```

### Step D: Generate Executive Report
```bash
python adve_realworld_test/scripts/generate_report.py --results results/realworld_report.json --output results/ADVE_RealWorld_Validation_Report.md
```

---

## 📊 Commercial Acceptance Thresholds

* **Overall Mean CosSim**: $\ge 0.9400$
* **Overall Min CosSim**: $\ge 0.8500$
* **Encoder Savings**: $\ge 70\%$
* **Mean Effective FPS**: $\ge 30.0\text{ FPS}$
