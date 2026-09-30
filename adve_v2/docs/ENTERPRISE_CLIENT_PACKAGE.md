# 📦 ADVE Enterprise v3.1 — Client Delivery Package & Delivery Manifest

## 1. What We Share with the Client (The Client Delivery Package)

When delivering ADVE v3.1 to enterprise client engineering and leadership teams (e.g. Tata Elxsi, CCTV vendors, Smart City procurement), share the following **5 core enterprise deliverables**:

```
ADVE-Enterprise-v3.1-Package/
├── 1_Client_ROI_Pitch.pdf            <-- Executive ROI & Cost Savings Deck
├── 2_Integration_Guide.pdf           <-- API & Python SDK Developer Documentation
├── 3_Benchmark_Reports.pdf           <-- Verified CCTV & Traffic Empirical Benchmarks
├── adve-enterprise-v3.1.tar.gz       <-- Docker Production Container Image
└── EVAL-LICENSE-KEY.txt               <-- SHA-256 HMAC Client License Key
```

---

## 2. Directory Structure of the Enterprise Codebase

The codebase has been cleaned up, audited, and formatted into enterprise production standards:

```text
c:\Users\harih\OneDrive\Documents\codex try\adve\adve_v2\
├── Dockerfile                         # Production Docker container configuration
├── docker-compose.yml                 # Docker Compose multi-container deployment
├── setup.py                           # Python package distribution setup
├── adve/                              # Core ADVE AI Pipeline Package
│   ├── api/                           # FastAPI REST Server & License Auth
│   │   ├── server.py                  # Endpoints (/health, /api/v1/embed, etc.)
│   │   └── license.py                 # SHA-256 HMAC License Validator
│   ├── core/                          # Core Anchor-Delta AI Engine
│   │   ├── pipeline_enterprise.py     # Production Pipeline Engine
│   │   ├── reconstructor_v3.py        # DeltaReconstructorV3 Neural Model
│   │   ├── anchor.py                  # YOLOv8 Anchor Gating & Motion Tracker
│   │   └── config.py                  # Enterprise Config Presets
│   ├── search/                        # FAISS & SQLite Vector Index Engine
│   └── stream/                        # Multi-Camera RTSP Stream Manager
├── sdk/                               # Client Python SDK
│   └── adve_sdk.py                    # Official Python Client API Wrapper
├── docs/                              # Enterprise Commercial Documentation
│   ├── CLIENT_ROI_PITCH.md            # Client ROI & Server Cost Savings Pitch Deck
│   ├── INTEGRATION_GUIDE.md           # Developer & SDK Integration Manual
│   ├── NOVEL_ALGORITHM_SPECIFICATION.md# NRHDA Patent Algorithm Specification
│   ├── PRODUCT_ROADMAP_V4_V5.md       # Multi-Generational Version Roadmap
│   └── COMMERCIAL_GO_TO_MARKET_STRATEGY.md # Commercial Strategy
├── results/                           # Verified Empirical Benchmark JSON Reports
│   ├── cctv_surveillance_benchmark_report.json
│   ├── traffic_benchmark_report.json
│   └── traffic_search_verification.json
└── scripts/                           # Enterprise Utility & License Tools
    ├── generate_license.py            # HMAC License Key Generator Tool
    ├── test_api_server.py             # Automated REST Endpoint Test Suite
    └── test_cctv_dataset.py           # CCTV Action Benchmark Runner
```

---

## 3. Client License Generation Command

To generate a custom license key for a new client evaluation:

```powershell
# Generate 3-Month Evaluation License for Tata Elxsi
python scripts/generate_license.py --client "Tata Elxsi" --tier evaluation --months 3
```
