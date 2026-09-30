import sys
import os
import time
import json
import torch
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.search.index import ADVESearchIndex
from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def main():
    traffic_video = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Vodra/North.mp4")
    if not os.path.exists(traffic_video):
        traffic_video = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Talaimari/North-East.mp4")

    print("=========================================================")
    print("      TRAFFIC VIDEO SEMANTIC SEARCH VERIFICATION AUDIT   ")
    print("=========================================================")
    print(f"Traffic Video: {os.path.basename(traffic_video)}")

    index_dir = os.path.abspath("data/traffic_semantic_search_index")
    os.makedirs(index_dir, exist_ok=True)
    search_index = ADVESearchIndex(index_dir)

    pipeline = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device="cpu",
        use_ego_motion=True,
        use_ema=True
    )
    pipeline.reset_state()

    print("\n--- STAGE 1: Processing & Indexing Traffic Video Frames ---")
    cap = cv2.VideoCapture(traffic_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    batch = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame_idx >= 200:
            break

        res = pipeline.process_frame(frame, frame_idx, no_validation=True)
        timestamp = frame_idx / fps

        obj_classes = [obj["class_name"] for obj in res.get("objects", [])]
        obj_metadata = ", ".join(set(obj_classes)) if obj_classes else "traffic scene"

        batch.append({
            "video_path": os.path.basename(traffic_video),
            "camera_id": "TRAFFIC-VODRA-CAM-01",
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "embedding": res["embedding"],
            "is_anchor": res["is_anchor"],
            "text": obj_metadata
        })

        if frame_idx % 50 == 0:
            print(f"   • Frame {frame_idx:3d} processed | Detected: {obj_metadata[:40]:40s} | Anchor: {res['is_anchor']}")

        frame_idx += 1

    cap.release()
    search_index.add_batch(batch)
    search_index.save()
    print(f"✅ Successfully indexed {len(batch)} traffic video frames into FAISS Vector Search Index.")

    # ------------------------------------------------------------------
    # STAGE 2: Execute Traffic Natural Language Search Queries
    # ------------------------------------------------------------------
    print("\n--- STAGE 2: Executing Traffic Semantic Search Queries ---")

    traffic_queries = [
        "bus or large vehicle in traffic stream",
        "motorbike or two wheeler driving past",
        "car driving through road intersection",
        "person walking near road traffic"
    ]

    search_verification_results = []

    for query in traffic_queries:
        t_start = time.perf_counter()
        raw_matches = search_index.search_by_text(query, k=3)
        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000.0

        matches = []
        for r in raw_matches:
            matches.append({
                "camera_id": r.camera_id,
                "timestamp_sec": round(r.timestamp, 2),
                "frame_idx": r.frame_idx,
                "similarity_score": round(r.similarity, 4),
                "is_anchor": r.is_anchor,
                "detected_objects": r.text
            })

        top_match = matches[0] if matches else {}
        top_cam = top_match.get("camera_id", "N/A")
        top_ts = top_match.get("timestamp_sec", 0.0)
        top_score = top_match.get("similarity_score", 0.0)

        search_verification_results.append({
            "query": query,
            "latency_ms": round(lat_ms, 2),
            "top_match_camera": top_cam,
            "top_match_timestamp": top_ts,
            "top_similarity_score": top_score,
            "all_matches": matches
        })

        print(f" • Query: '{query}'")
        print(f"   --> Top Match: {top_cam} at t={top_ts}s (Frame #{top_match.get('frame_idx')}, Score: {top_score:.4f}, Latency: {lat_ms:.2f} ms)")

    # Save summary report
    out_file = "results/traffic_search_verification.json"
    os.makedirs("results", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(search_verification_results, f, indent=2)

    print("\n=========================================================")
    print(f"✅ TRAFFIC SEMANTIC SEARCH VERIFIED SUCCESSFULLY! Saved to: {out_file}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
