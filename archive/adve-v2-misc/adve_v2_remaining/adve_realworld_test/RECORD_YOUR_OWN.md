# Protocol for Recording Proprietary Test Data

Follow this protocol to record customer-specific test videos using a standard smartphone or camera:

---

## 1. Recording Guidelines

1. **Resolution & Frame Rate**: 1080p (1920x1080) at 30 FPS.
2. **Format**: Save as standard H.264 / MP4.
3. **Length**: 60 to 90 seconds per scene.

---

## 2. Test Scene Checklist

| Target Domain | Scenes to Record | Key Movements |
| :--- | :--- | :--- |
| **Office / Room** | Indoor workspace | Person enters $\rightarrow$ object interaction $\rightarrow$ exit |
| **Street / Parking** | Outdoor sidewalk | Pedestrians walking $\rightarrow$ vehicle passes $\rightarrow$ occlusion |
| **Night / Low Light** | Low-light hallway / outdoor night | Pedestrian walking in low illumination |

---

## 3. Preparing Videos for Manifest

Move your recorded MP4 files to a directory (e.g. `my_videos/`), then run:

```bash
python adve_realworld_test/scripts/prepare_datasets.py \
    --generate-manifest my_videos/ \
    --output datasets/my_manifest.json
```
