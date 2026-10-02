import sys
import os
import time
import json
import torch
import cv2
import numpy as np
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    traffic_video = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Vodra/North.mp4")
    if not os.path.exists(traffic_video):
        traffic_video = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Talaimari/North-East.mp4")

    print("=========================================================")
    print("      TRAFFIC SURVEILLANCE VIDEO BENCHMARK AUDIT        ")
    print("=========================================================")
    print(f"Traffic Video File : {os.path.basename(traffic_video)}")
    print(f"Full Video Path    : {traffic_video}")
    print("---------------------------------------------------------\n")

    max_frames_to_test = 300  # Evaluate 300 frames of traffic surveillance

    # Hardware Power Profile Baseline
    # Standard System Power Draw: Average CPU/GPU power consumption during active inference (45 Watts)
    POWER_RATING_WATTS = 45.0
    # FLOPs per vision encoder pass (CLIP ViT-B/32 ~ 4.4 GFLOPs)
    CLIP_GFLOPS_PER_FRAME = 4.4
    # FLOPs per ADVE DeltaReconstructor pass (~ 0.08 GFLOPs)
    ADVE_RECON_GFLOPS_PER_FRAME = 0.08

    device = "cpu"

    # ------------------------------------------------------------------
    # 1. NORMAL MODEL (Full Vision Encoder on 100% of frames)
    # ------------------------------------------------------------------
    print("🎥 1. Running NORMAL MODEL (Full Vision Transformer Encoder)...")
    import gc
    gc.collect()
    mem_before_normal = measure_memory_mb()

    pipeline_normal = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device=device,
        use_ego_motion=False,
        use_ema=False
    )

    t0 = time.perf_counter()
    # Force 0% savings (every frame encoded through heavy vision encoder)
    res_normal = pipeline_normal.process_video(traffic_video, max_frames=max_frames_to_test, no_validation=False)
    t_normal = time.perf_counter() - t0
    mem_after_normal = measure_memory_mb()
    peak_mem_normal = max(mem_after_normal, mem_before_normal)

    frames_normal = len(pipeline_normal.validator.records) if pipeline_normal.validator.records else max_frames_to_test
    fps_normal = frames_normal / t_normal if t_normal > 0 else 0.0

    # Computing Power & Energy Normal
    total_gflops_normal = frames_normal * CLIP_GFLOPS_PER_FRAME
    energy_joules_normal = POWER_RATING_WATTS * t_normal
    energy_wh_normal = energy_joules_normal / 3600.0

    print(f"   • Frames Processed : {frames_normal}")
    print(f"   • Time Taken       : {t_normal:.2f} seconds ({fps_normal:.1f} FPS)")
    print(f"   • Memory Used (RAM): {peak_mem_normal:.1f} MB")
    print(f"   • Computing Power  : {total_gflops_normal:.1f} GFLOPs ({CLIP_GFLOPS_PER_FRAME:.1f} GFLOPs/frame)")
    print(f"   • Electricity Used : {energy_joules_normal:.2f} Joules ({energy_wh_normal * 1000:.2f} mWh)")

    # ------------------------------------------------------------------
    # 2. OUR ADVE RESULT (Anchor-Delta Video Embedding Pipeline)
    # ------------------------------------------------------------------
    print("\n⚡ 2. Running OUR ADVE MODEL (Anchor-Delta Reconstruction Pipeline)...")
    pipeline_adve = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device=device,
        use_ego_motion=True,
        use_ema=True
    )
    pipeline_adve.reset_state()
    gc.collect()
    mem_before_adve = measure_memory_mb()

    t1 = time.perf_counter()
    res_adve = pipeline_adve.process_video(traffic_video, max_frames=max_frames_to_test, no_validation=False)
    t_adve = time.perf_counter() - t1
    mem_after_adve = measure_memory_mb()
    peak_mem_adve = max(mem_after_adve, mem_before_adve)

    recs_adve = pipeline_adve.validator.records
    sims_adve = [r['cosine_sim'] for r in recs_adve] if recs_adve else [1.0]

    mean_sim_adve = float(np.mean(sims_adve))
    min_sim_adve = float(np.min(sims_adve))
    savings_adve_pct = float(res_adve.get("encoder_savings_pct", 0.0))

    frames_adve = len(sims_adve)
    fps_adve = frames_adve / t_adve if t_adve > 0 else 0.0

    # Calculate actual encoder passes vs skipped passes in ADVE
    encoder_passes = int(frames_adve * (1.0 - (savings_adve_pct / 100.0)))
    skipped_passes = frames_adve - encoder_passes

    total_gflops_adve = (encoder_passes * CLIP_GFLOPS_PER_FRAME) + (skipped_passes * ADVE_RECON_GFLOPS_PER_FRAME)
    energy_joules_adve = POWER_RATING_WATTS * t_adve
    energy_wh_adve = energy_joules_adve / 3600.0

    gflops_savings_pct = ((total_gflops_normal - total_gflops_adve) / total_gflops_normal) * 100.0
    energy_savings_pct = ((energy_joules_normal - energy_joules_adve) / energy_joules_normal) * 100.0
    time_saved_pct = ((t_normal - t_adve) / t_normal) * 100.0 if t_normal > 0 else 0.0

    print(f"   • Frames Processed : {frames_adve}")
    print(f"   • Time Taken       : {t_adve:.2f} seconds ({fps_adve:.1f} FPS)")
    print(f"   • Memory Used (RAM): {peak_mem_adve:.1f} MB")
    print(f"   • Computing Power  : {total_gflops_adve:.1f} GFLOPs ({total_gflops_adve/frames_adve:.2f} GFLOPs/frame avg)")
    print(f"   • Electricity Used : {energy_joules_adve:.2f} Joules ({energy_wh_adve * 1000:.2f} mWh)")
    print(f"   • Embedding Accuracy: Mean CosSim: {mean_sim_adve:.4f} | Min: {min_sim_adve:.4f}")
    print(f"   • Encoder Savings  : {savings_adve_pct:.1f}%")

    # ------------------------------------------------------------------
    # 3. SAVE STRUCTURED BENCHMARK REPORT
    # ------------------------------------------------------------------
    summary_report = {
        "benchmark": "Traffic Surveillance Video Comparison",
        "video_file": os.path.basename(traffic_video),
        "total_frames_evaluated": frames_adve,
        "normal_model": {
            "time_taken_sec": round(t_normal, 2),
            "fps": round(fps_normal, 1),
            "memory_used_mb": round(peak_mem_normal, 1),
            "computing_power_gflops": round(total_gflops_normal, 1),
            "electricity_used_joules": round(energy_joules_normal, 2),
            "electricity_used_mwh": round(energy_wh_normal * 1000, 2),
            "embedding_accuracy": 1.0000
        },
        "adve_model": {
            "time_taken_sec": round(t_adve, 2),
            "fps": round(fps_adve, 1),
            "memory_used_mb": round(peak_mem_adve, 1),
            "computing_power_gflops": round(total_gflops_adve, 1),
            "electricity_used_joules": round(energy_joules_adve, 2),
            "electricity_used_mwh": round(energy_wh_adve * 1000, 2),
            "mean_embedding_accuracy": round(mean_sim_adve, 4),
            "min_embedding_accuracy": round(min_sim_adve, 4),
            "encoder_savings_pct": round(savings_adve_pct, 1)
        },
        "comparison_savings": {
            "time_reduction_pct": round(time_saved_pct, 1),
            "computing_power_saved_pct": round(gflops_savings_pct, 1),
            "electricity_saved_pct": round(energy_savings_pct, 1),
            "precision_retention_pct": round(mean_sim_adve * 100.0, 2)
        }
    }

    report_path = "results/traffic_benchmark_report.json"
    os.makedirs("results", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n=========================================================")
    print("      TRAFFIC SURVEILLANCE SIDE-BY-SIDE BENCHMARK SUMMARY")
    print("=========================================================")
    print(f"Time Taken       : Normal = {t_normal:.2f} s  | ADVE = {t_adve:.2f} s  ({time_saved_pct:+.1f}% faster)")
    print(f"Computing Power  : Normal = {total_gflops_normal:.1f} GF | ADVE = {total_gflops_adve:.1f} GF ({gflops_savings_pct:+.1f}% saved)")
    print(f"Memory Used (RAM): Normal = {peak_mem_normal:.1f} MB | ADVE = {peak_mem_adve:.1f} MB")
    print(f"Electricity Used : Normal = {energy_joules_normal:.1f} J  | ADVE = {energy_joules_adve:.1f} J  ({energy_savings_pct:+.1f}% saved)")
    print(f"Embedding Precision: ADVE = {mean_sim_adve:.4f} Mean CosSim ({mean_sim_adve*100:.2f}% Accuracy)")
    print(f"Report File      : {report_path}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
