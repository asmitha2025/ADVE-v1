# ADVE Enterprise — Legal & IP Filing Guide (Phase 1, Day 1)

This document contains the exact patent text, claims, filing instructions, and security guidelines required to lock down intellectual property for **ADVE (Anchor-Delta Video Embedding System)**.

---

## 1. IP India Provisional Patent Filing (Form 2)

**Portal:** [https://ipindia.gov.in](https://ipindia.gov.in)  
**Form:** Form 2 (Provisional Specification)  
**Official Fee:** ₹2,000 (Individual / Micro Entity / Startup)  

### Title of Invention
`Anchor-Delta Video Embedding System for Efficient Neural Video Encoding`

### Abstract
An AI-driven video processing and embedding system that dramatically reduces neural encoder compute requirements (e.g., CLIP, Vision Transformers) across continuous video streams. The system dynamically establishes anchor frames, computes spatial bounding box graphs and appearance deltas, and reconstructs deep neural embeddings for intermediate non-anchor (delta) frames using object tracking state transitions without executing full deep neural network inference on every frame. High fidelity and cosine similarity threshold compliance are maintained via closed-loop real-time verification and adaptive keyframe refresh triggers.

### Detailed Description & Technical Specification
The invention provides a hardware-efficient architecture for continuous real-time neural video encoding across surveillance (RTSP), video search, and edge computing environments.

1. **Anchor Frame Processor**: Evaluates incoming frames, triggering a full deep vision transformer (e.g. CLIP ViT-B/32, ViT-L/14) and object detection pass only on anchor frames. The spatial graph module constructs a relational graph of detected object bounding boxes, centroids, area ratios, and localized histograms.
2. **Delta Frame Tracker**: Tracks detected bounding box transitions between frames using lightweight multi-object tracking (e.g., ByteTrack). Computes normalized centroid displacement vectors ($\Delta G$) and frame appearance histogram correlation drops ($\Delta A$).
3. **Embedding Reconstructor**: Interpolates and updates intermediate feature representations using tracking motion vectors, homography transformation matrices, and GRU/EMA hidden state updates, bypassing raw pixel transformer forward passes.
4. **Closed-Loop Validator**: Continuously samples generated embeddings and calculates real-time cosine similarity against ground-truth representations. If cosine similarity drops below a dynamic confidence threshold $T_{cos}$ (default 0.85), an immediate anchor refresh is forced.

### Patent Claims (6 Claims)

1. **Claim 1**: An automated video embedding system comprising:
   - an anchor detection module configured to trigger full neural vision encoder execution on anchor frames;
   - a spatial graph module configured to extract object bounding box spatial topologies and appearance features;
   - a delta tracking module configured to compute inter-frame spatial and motion displacement vectors without invoking said neural vision encoder; and
   - an embedding reconstruction module configured to generate intermediate frame feature embeddings by applying spatial displacement updates to anchor frame embeddings.

2. **Claim 2**: The system of Claim 1, wherein said anchor detection module forces an anchor frame refresh upon any of:
   - normalized spatial displacement magnitude exceeding a configurable threshold $T_{spatial}$;
   - histogram appearance correlation drop exceeding an appearance threshold $T_{appearance}$;
   - inter-frame count reaching a maximum keyframe delta budget $N_{max}$; or
   - detection of a newly appearing object class or ID.

3. **Claim 3**: The system of Claim 1, wherein said delta tracking module utilizes a single persistent object tracker instance to track bounding box IDs, centroids, and bounding box scale across contiguous non-anchor frames.

4. **Claim 4**: The system of Claim 1, wherein said embedding reconstruction module estimates homography matrices between successive frame pairs using feature point matching to compensate for camera panning, tilt, and zoom movements.

5. **Claim 5**: The system of Claim 1, further comprising a closed-loop validation engine that calculates cosine similarity between reconstructed embeddings and full-encoder reference embeddings, automatically forcing an anchor frame refresh when cosine similarity drops below a preset quality target $T_{cos}$.

6. **Claim 6**: A method for real-time video stream embedding optimization, comprising:
   - receiving an RTSP or media stream into a buffer queue;
   - selectively executing full vision transformer inference on anchor frames;
   - executing lightweight object tracking on intermediate delta frames;
   - reconstructing intermediate feature vectors via spatial graph transformation; and
   - streaming reconstructed embeddings to a vector database or downstream analytical pipeline.

---

## 2. US Provisional Patent Filing

**Portal:** USPTO EFS-Web / Patent Center ([https://patentcenter.uspto.gov](https://patentcenter.uspto.gov))  
**Filing Fee:** ~$250 (Micro Entity)  
**Specification:** Use the title, abstract, detailed description, and 6 claims specified above.

---

## 3. Legal Lock & Trademark Setup

| Item | Action Required | Status |
|------|----------------|--------|
| **GitHub Private** | Go to GitHub Repo Settings → Danger Zone → Change Visibility to Private. Take a screenshot for compliance audit. | Required |
| **Trademark Filing** | File "ADVE" trademark under Class 9 (Software for video processing and artificial intelligence) on `ipindia.gov.in` (Fee: ~₹4,000). | Required |
| **NDA Template** | Use standard 1-page mutual NDA for prospective enterprise client evaluations (Tata Elxsi, Hikvision, L&T, etc.). | Ready |
