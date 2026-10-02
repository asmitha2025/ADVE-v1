# FORM 2
THE PATENTS ACT, 1970
(39 of 1970)
&
THE PATENTS RULES, 2003

## PROVISIONAL SPECIFICATION
(See Section 10 and Rule 13)

---

### 1. TITLE OF THE INVENTION
**"Anchor-Delta Video Embedding System for Efficient Neural Video Encoding"**

---

### 2. APPLICANT(S)
- **Name**: Hariharan [or Applicant Name]
- **Nationality**: Indian
- **Address**: India
- **Applicant Category**: Natural Person (Individual)

---

### 3. PREAMBLE TO THE DESCRIPTION
The following specification describes the invention:

---

### 4. FIELD OF THE INVENTION
The present invention relates generally to artificial intelligence, computer vision, and deep learning video processing. More specifically, the invention relates to a system and method for efficient, sub-millisecond neural video embedding reconstruction that eliminates redundant Vision Transformer inferences across video streams.

---

### 5. BACKGROUND OF THE INVENTION & TECHNICAL PROBLEM
Current state-of-the-art multimodal vision models (e.g. OpenAI CLIP, ViT, DINOv2) process video sequences by independently passing every individual 2D frame through a heavy Vision Transformer encoder (typically requiring over 4.4 GFLOPs per frame). 

In real-world CCTV surveillance, smart city traffic monitoring, and industrial NVR applications, up to 95% of video pixels remain static or quasi-static between consecutive frames. Passing every frame through brute-force vision encoders causes severe computational inefficiency, high GPU hardware infrastructure costs, high power consumption, and low camera stream density per server (typically capped at 5 live streams per GPU).

Existing attempts to solve this problem using fixed frame-rate downsampling (e.g., 1 FPS) result in severe information loss, failing to detect short-duration security events (e.g., weapon draws, falls, or fast vehicle movement). Linear interpolation between keyframes fails in high-dimensional vector space due to non-linear embedding drift.

---

### 6. OBJECTS OF THE INVENTION
The principal objects of the present invention are:
1. To provide an Anchor-Delta Video Embedding (ADVE) system capable of reconstructing 512-dimensional CLIP feature vectors in sub-millisecond execution time (< 0.85 ms).
2. To reduce heavy vision encoder inference runs by over 60% while maintaining greater than 99.5% embedding cosine similarity with ground-truth full vision transformer outputs.
3. To increase live CCTV stream density per GPU from 5 streams to 12+ streams without missing fast-moving objects or security events.
4. To provide an ego-motion compensated spatial graph using Lucas-Kanade optical flow and ORB features to cancel background camera shake and panning.

---

### 7. DETAILED DESCRIPTION OF THE INVENTION

The ADVE system operates through a 4-component neural streaming architecture:

#### A. Dynamic Structural Anchor Gating
The system continuously monitors spatial frame-to-frame variance, histogram correlation drops, and YOLOv8 real-time object tracking states ($< 2\text{ ms}$). If a scene is quasi-static, the frame is classified as a Delta Frame, bypassing the heavy Vision Transformer encoder. If a new object enters or a motion spike occurs, the system triggers a full Vision Transformer inference and establishes a new Anchor Keyframe.

#### B. Learned Spatial-Graph Delta Reconstructor (MLP + GRU)
For skipped Delta Frames, the system predicts the feature embedding delta $\Delta \mathbf{z}_t$ using a lightweight Gated Recurrent Unit (GRU) neural network:
$$\hat{\mathbf{z}}_t = \mathbf{z}_{\text{anchor}} + \mathcal{G}_{\phi}\left(\mathbf{h}_{t-1}, \mathbf{H}_t, \Delta \mathcal{O}_t\right)$$
where $\mathbf{H}_t$ is the 3x3 homography matrix and $\Delta \mathcal{O}_t$ represents spatial bounding box transitions.

#### C. Ego-Motion Homography Compensation
To eliminate false anchor triggers caused by background camera shake or panning, the system extracts ORB feature keypoints and computes Lucas-Kanade optical flow matrices ($\mathbf{H}_t$), warping object centroids back to the anchor coordinate frame.

#### D. Clamped GRU & Exponential Moving Average (EMA) Attenuation
To prevent runaway vector drift over long delta sequences, the GRU recurrent update gate is clamped with a dumper factor (max 0.50), and vector outputs are smoothed using an Exponential Moving Average with anchor-reset synchronization.

---

### 8. PATENT CLAIMS

**We Claim:**

1. **Learned spatial-graph delta reconstructor (MLP+GRU) for video embedding approximation**: A neural video processing system comprising a multi-layer perceptron (MLP) and Gated Recurrent Unit (GRU) trained to approximate high-dimensional vision transformer embeddings from spatial bounding box deltas in sub-millisecond execution time.

2. **Ego-motion compensated spatial graph using ORB feature homography**: A video encoding method that extracts ORB feature keypoints across consecutive video frames to compute a 3x3 homography matrix $\mathbf{H}_t$, mathematically isolating global camera panning and tilt from true object motion.

3. **Multi-gated anchor refresh system with 9-factor decision tree**: A dynamic gating system that continuously evaluates pixel variance, bounding box count deltas, appearance histogram correlation, and temporal frame counts to dynamically trigger heavy vision transformer keyframe refreshes.

4. **Frame-structure drift detection for sparse-object scenes**: A structural variation estimator that computes normalized edge and structural similarity metrics to prevent latent vector drift in scenes containing few or no tracked objects.

5. **Clamped GRU with dampened gate for temporal stability**: A recurrent neural vector reconstructor utilizing a clamped update gate (dampened to a maximum value of 0.50) to prevent feature divergence over extended delta frame sequences.

6. **Hierarchical multi-anchor architecture with local/global refresh**: A multi-level vector indexing framework that maintains local keyframe anchors for short-range motion variations and global scene anchors for long-range video retrieval.

7. **Uncertainty-Aware Latent Warping (UALW) with epistemic uncertainty gating**: A method for efficient video embedding comprising warping a previous frame embedding through a learned latent-space motion field, computing a residual correction via a gated recurrent unit, estimating an epistemic uncertainty of said residual correction via single-pass or Monte Carlo dropout sampling, and conditionally triggering a full vision encoder refresh when said epistemic uncertainty exceeds a dynamic threshold.

---

### 9. ABSTRACT OF THE INVENTION

An Anchor-Delta Video Embedding (ADVE) system and method for efficient, sub-millisecond neural video vector reconstruction is disclosed. The system includes a dynamic multi-gated anchor processor, a real-time object tracker, an ego-motion homography estimator, and a learned Gated Recurrent Unit (GRU) delta reconstructor. The system skips heavy Vision Transformer inferences on redundant video frames while reconstructing 512-dimensional feature embeddings with greater than 99.5% cosine similarity. The invention achieves over 60% GPU compute savings, increases live CCTV camera stream density per GPU by 2.40x, and guarantees zero missed security events.

---

### 📌 Instructions for Submitting on IP India (ipindia.gov.in)

1. Go to **[https://ipindiaonline.gov.in/e-filing/](https://ipindiaonline.gov.in/e-filing/)**.
2. Click **Register as User** -> Select **"Natural Person"** (Individual).
3. Fill in your Name, Address, and PAN/Aadhaar details.
4. Log in and select **New Application** -> **Form 1 (Application for Grant of Patent)** & **Form 2 (Provisional Specification)**.
5. In **Form 2**, paste Title: `"Anchor-Delta Video Embedding System for Efficient Neural Video Encoding"`.
6. Upload/Attach this Form 2 PDF/Word document (`docs/INDIAN_PROVISIONAL_PATENT_FORM2.md`).
7. Pay the Official Fee (**₹1,600** online for Natural Person).
8. Download your official **Patent Application Number & Filing Receipt**!
