# ADVE Quality Test Report

## Summary
- Videos tested: 4
- Excellent/Good: 3
- Needs Work: 0
- Mean cosine sim: 0.9496
- Mean encoder savings: 64.9%
- Production ready: ✅ YES

## Per-Type Results

| Video Type | Mean Sim | Min Sim | Savings | Search | Verdict |
|------------|----------|---------|---------|--------|---------|
| traffic         | 0.9526 | 0.7948 | 69.3% | ✅ | GOOD |
| education       | 0.8902 | 0.4184 | 89.5% | ✅ | ACCEPTABLE |
| action          | 0.9573 | 0.9443 | 96.0% | ✅ | EXCELLENT |
| whatsapp        | 0.9984 | 0.9601 | 4.8% | ✅ | EXCELLENT |

## Issues Found

### education
- QUALITY FLOOR: min cosine sim 0.418 — some frames very poorly reconstructed

### whatsapp
- LOW SAVINGS: only 4.8% — video has high scene change rate


## Quality Plots

### Traffic Quality Plot
![traffic plot](quality_plots/traffic_similarity.png)

### Education Quality Plot
![education plot](quality_plots/education_similarity.png)

### Action Quality Plot
![action plot](quality_plots/action_similarity.png)

### Whatsapp Quality Plot
![whatsapp plot](quality_plots/whatsapp_similarity.png)
