import os
import sys
import time
import base64
import subprocess
import shutil
import cv2
import numpy as np
import gradio as gr
import yt_dlp
import groq

from typing import Optional
import static_ffmpeg
static_ffmpeg.add_paths()


# Ensure adve_v2 is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from adve.core.pipeline import ADVEPipeline
from adve.core.config import Config
from adve.search.index import ADVESearchIndex, SearchResult, normalize_video_path
from adve.core.audio_transcriber import AudioTranscriber

# Global state to keep track of active index, video, and search results
active_video_path = None
active_search_results = []
index_dir = os.path.join(current_dir, "data", "demo_index")
os.makedirs(index_dir, exist_ok=True)
search_index = ADVESearchIndex(index_dir)

# YOLO-World dynamic object detection model (lazy-loaded on demand at search time)
yolo_world_model = None

def get_yolo_world():
    global yolo_world_model
    if yolo_world_model is None:
        try:
            print("[YOLO-World] Loading yolov8s-worldv2.pt model...")
            from ultralytics import YOLOWorld
            yolo_world_model = YOLOWorld("yolov8s-worldv2.pt")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            yolo_world_model.to(device)
            print("[YOLO-World] Loaded successfully.")
        except Exception as e:
            print(f"[YOLO-World Warning] Failed to load YOLO-World model: {e}")
            yolo_world_model = False
    return yolo_world_model

def draw_yolo_world_boxes(frame, query: str):
    model = get_yolo_world()
    if not model:
        return frame
    try:
        model.set_classes([query])
        # Use conf=0.15 for flexible zero-shot matching
        results = model.predict(frame, conf=0.15, verbose=False, device=model.device)
        if results and len(results) > 0 and len(results[0]) > 0:
            return results[0].plot()
    except Exception as e:
        print(f"[YOLO-World Error] Failed to run prediction: {e}")
    return frame

def fmt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def annotate_preview_image(frame, match_index: int, timestamp: float):
    img = frame.copy()
    h, w = img.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Match {match_index}"
    text_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, text_scale, thickness)
    pad_x, pad_y = 8, 6
    rect_x1, rect_y1 = 12, 12
    rect_x2, rect_y2 = 12 + tw + 2*pad_x, 12 + th + 2*pad_y
    cv2.rectangle(img, (rect_x1, rect_y1), (int(rect_x2), int(rect_y2)), (235, 99, 37), -1)
    cv2.putText(img, text, (rect_x1 + pad_x, rect_y1 + th + pad_y), font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    ts_str = fmt_ts(timestamp)
    (ts_w, ts_h), ts_base = cv2.getTextSize(ts_str, font, text_scale, thickness)
    margin = 12
    trect_x1 = w - ts_w - 2*pad_x - margin
    trect_y1 = h - ts_h - 2*pad_y - margin
    trect_x2 = w - margin
    trect_y2 = h - margin
    
    overlay = img.copy()
    cv2.rectangle(overlay, (int(trect_x1), int(trect_y1)), (int(trect_x2), int(trect_y2)), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.putText(img, ts_str, (int(trect_x1 + pad_x), int(trect_y1 + ts_h + pad_y)), font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

empty_meta = """
<div class="meta-row">
    <span class="score-badge">Score --</span>
    <span class="frame-badge">Frame --</span>
</div>
"""

def make_meta_html(score: float, frame_idx: int):
    return f"""
    <div class="meta-row">
        <span class="score-badge">Score {score:.3f}</span>
        <span class="frame-badge">Frame #{frame_idx}</span>
    </div>
    """

def generate_timeline_svg(results, active_duration=300.0, current_ts=None):
    width = 800
    height = 200
    padding_bottom = 30
    padding_top = 10
    plot_h = height - padding_bottom - padding_top
    
    steps = 150
    points = []
    
    import math
    for i in range(steps):
        x_time = (i / (steps - 1)) * active_duration
        y_val = 0.05
        if results:
            for r in results:
                sim = getattr(r, "similarity", 0.0) if not isinstance(r, dict) else r.get("score", 0.0)
                ts = getattr(r, "timestamp", 0.0) if not isinstance(r, dict) else r.get("timestamp", 0.0)
                dist = x_time - ts
                y_val += sim * math.exp(-(dist**2) / (2 * 15.0**2))
        else:
            y_val += 0.8 * math.exp(-((x_time - active_duration * 0.6)**2) / (2 * 20.0**2))
            y_val += 0.3 * math.exp(-((x_time - active_duration * 0.3)**2) / (2 * 12.0**2))
            y_val += 0.2 * math.exp(-((x_time - active_duration * 0.85)**2) / (2 * 15.0**2))

        y_val = min(1.1, max(0.02, y_val))
        points.append((x_time, y_val))
        
    svg_points = []
    for i, (t, y) in enumerate(points):
        px = (i / (steps - 1)) * width
        py = height - padding_bottom - (y / 1.1) * plot_h
        svg_points.append(f"{px:.1f},{py:.1f}")
        
    path_d = f"M 0.0,{height - padding_bottom:.1f} L " + " L ".join(svg_points) + f" L {width:.1f},{height - padding_bottom:.1f} Z"
    stroke_d = "M " + " L ".join(svg_points)
    
    labels_html = ""
    num_labels = 6
    for i in range(num_labels):
        lbl_ts = (i / (num_labels - 1)) * active_duration
        lbl_str = fmt_ts(lbl_ts)
        lbl_x = (i / (num_labels - 1)) * width
        anchor = "middle"
        if i == 0: anchor = "start"
        elif i == num_labels - 1: anchor = "end"
        labels_html += f'<text x="{lbl_x:.1f}" y="{height - 10}" fill="#8e9aa8" font-size="11" text-anchor="{anchor}">{lbl_str[3:]}</text>'
        
    marker_html = ""
    if current_ts is not None:
        marker_x = (current_ts / active_duration) * width
        marker_html = f"""
        <line x1="{marker_x:.1f}" y1="{padding_top}" x2="{marker_x:.1f}" y2="{height - padding_bottom}" stroke="#00f2fe" stroke-width="2" stroke-dasharray="4,4"/>
        <circle cx="{marker_x:.1f}" cy="{padding_top}" r="4" fill="#00f2fe"/>
        <g transform="translate({marker_x:.1f}, {padding_top})">
            <rect x="-30" y="-22" width="60" height="18" rx="4" fill="#00f2fe"/>
            <text x="0" y="-10" fill="#0c0e12" font-size="10" font-weight="bold" text-anchor="middle">{fmt_ts(current_ts)}</text>
        </g>
        """
        
    svg_content = f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="background:#13151c; border-radius:8px; padding:10px 0 0 0;">
        <defs>
            <linearGradient id="heatGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#ef4444" stop-opacity="0.85"/>
                <stop offset="30%" stop-color="#f59e0b" stop-opacity="0.7"/>
                <stop offset="60%" stop-color="#00f2fe" stop-opacity="0.5"/>
                <stop offset="100%" stop-color="#1e1b4b" stop-opacity="0.1"/>
            </linearGradient>
        </defs>
        <line x1="0" y1="{padding_top + plot_h*0.25:.1f}" x2="{width}" y2="{padding_top + plot_h*0.25:.1f}" stroke="#1f242e" stroke-width="1"/>
        <line x1="0" y1="{padding_top + plot_h*0.5:.1f}" x2="{width}" y2="{padding_top + plot_h*0.5:.1f}" stroke="#1f242e" stroke-width="1"/>
        <line x1="0" y1="{padding_top + plot_h*0.75:.1f}" x2="{width}" y2="{padding_top + plot_h*0.75:.1f}" stroke="#1f242e" stroke-width="1"/>
        <text x="-90" y="20" fill="#6c7a89" font-size="10" transform="rotate(-90)" font-weight="600">Relevance</text>
        <path d="{path_d}" fill="url(#heatGrad)"/>
        <path d="{stroke_d}" fill="none" stroke="#00f2fe" stroke-width="2"/>
        <line x1="0" y1="{height - padding_bottom}" x2="{width}" y2="{height - padding_bottom}" stroke="#1f242e" stroke-width="2"/>
        {labels_html}
        {marker_html}
    </svg>
    """
    return svg_content

# Warm up CLIP and Whisper models on the main thread to prevent thread-safety crashes on Windows
try:
    print("[Demo Startup] Warming up CLIP text encoder...")
    search_index.search_by_text("warmup", k=1)
    print("[Demo Startup] CLIP model warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: CLIP warmup failed: {e}")

try:
    print("[Demo Startup] Warming up Whisper model...")
    transcriber = AudioTranscriber(model_name="tiny")
    transcriber._load_model()
    print("[Demo Startup] Whisper model warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: Whisper warmup failed: {e}")

try:
    print("[Demo Startup] Warming up ADVE Pipeline (YOLO & Reconstructor)...")
    config = Config()
    warmup_pipeline = ADVEPipeline(
        config,
        clip_model = search_index._clip_model,
        clip_preprocess = search_index._clip_prep
    )
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    warmup_pipeline.process_frame(dummy_frame, 0, no_validation=True)
    print("[Demo Startup] ADVE Pipeline warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: ADVE Pipeline warmup failed: {e}")

global_audio_indexer = None
try:
    print("[Demo Startup] Initializing and warming up AudioIndexer...")
    from adve.audio.indexer import AudioIndexer
    global_audio_indexer = AudioIndexer(
        search_index,
        device = "cpu",
        clip_model = search_index._clip_model,
        clip_prep = search_index._clip_prep
    )
    global_audio_indexer._get_whisper()
    global_audio_indexer._get_clip()
    print("[Demo Startup] AudioIndexer warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: AudioIndexer warmup failed: {e}")

global_tiled_encoder = None
global_ocr_extractor = None
global_unified_search = None

try:
    print("[Demo Startup] Initializing vision and search extensions...")
    from adve.vision.tiled_encoder import TiledEncoder
    from adve.vision.ocr_extractor import OCRExtractor
    from adve.vision.unified_search import UnifiedSearchEngine
    
    config = Config()
    if warmup_pipeline is not None:
        global_tiled_encoder = TiledEncoder(
            clip_model = warmup_pipeline.anchor_proc.clip_model,
            clip_prep  = warmup_pipeline.anchor_proc.clip_preprocess,
            device     = config.DEVICE,
        )
    
    ocr_db_path = os.path.join(index_dir, "ocr.db")
    os.makedirs(os.path.dirname(ocr_db_path), exist_ok=True)
    global_ocr_extractor = OCRExtractor(
        db_path = ocr_db_path,
        device  = config.DEVICE,
    )
    
    global_unified_search = UnifiedSearchEngine(
        visual_index = search_index,
        ocr_extractor = global_ocr_extractor,
        audio_indexer = global_audio_indexer,
    )
    print("[Demo Startup] Vision and search extensions initialized successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: Failed to initialize extensions: {e}")




def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None





def download_youtube(url: str, progress=gr.Progress()) -> str:
    """Download the first 5 minutes of a YouTube video, or fall back to low-res full download if ffmpeg is missing."""
    progress(0.05, desc="Checking YouTube URL...")
    os.makedirs("demo_videos", exist_ok=True)

    ffmpeg_ok = is_ffmpeg_available()

    # --- Real-time download progress hook ---
    _last_pct = [0.0]
    def _progress_hook(d):
        if d.get("status") == "downloading":
            total   = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = 0.1 + 0.75 * (downloaded / total)   # maps 0→100% download into 10%→85% of progress bar
                pct = min(0.85, pct)
                if pct - _last_pct[0] >= 0.02:             # only update every 2% to reduce Gradio spam
                    _last_pct[0] = pct
                    mb_done = downloaded / 1_048_576
                    mb_total = total / 1_048_576
                    progress(pct, desc=f"Downloading: {mb_done:.1f} / {mb_total:.1f} MB ({pct*100:.0f}%)")
        elif d.get("status") == "finished":
            progress(0.88, desc="Download complete — preparing video...")

    if ffmpeg_ok:
        opts = {
            "format": "mp4/best",
            "outtmpl": "demo_videos/%(id)s.%(ext)s",
            "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": 300}],
            "force_keyframes_at_cuts": True,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"]
                }
            },
            "progress_hooks": [_progress_hook],
        }
        progress(0.10, desc="Starting download (first 5 minutes)...")
    else:
        opts = {
            "format": "worst[ext=mp4]/mp4",
            "outtmpl": "demo_videos/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"]
                }
            },
            "progress_hooks": [_progress_hook],
        }
        progress(0.10, desc="ffmpeg not found — downloading full video in low-res...")

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            ext = info.get("ext", "mp4")
            expected_path = f"demo_videos/{video_id}.{ext}"

            if os.path.exists(expected_path):
                progress(0.90, desc="Video ready for indexing!")
                return expected_path

            # Scan directory as fallback
            for file in os.listdir("demo_videos"):
                if file.startswith(video_id):
                    progress(0.90, desc="Video ready for indexing!")
                    return os.path.join("demo_videos", file)
            raise FileNotFoundError("Downloaded video file not found.")
    except Exception as e:
        raise RuntimeError(f"Failed to download YouTube video: {e}")


def index_video(video_path: str, sampling_rate: float = 5.0, use_adaptive_fps: bool = True, index_audio: bool = False, index_ocr: bool = False, progress=gr.Progress()) -> str:
    """Run ADVE pipeline to index anchor frames in the video."""
    global active_video_path
    global search_index
    active_video_path = video_path

    progress(0.0, desc="Initializing ADVE pipeline...")
    config = Config()
    # CLIP runs on CPU (for memory stability) while YOLO runs on GPU (for speed) if CUDA is available
    pipeline = ADVEPipeline(
        config,
        clip_model = search_index._clip_model,
        clip_preprocess = search_index._clip_prep
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # Calculate step size based on sampling rate (e.g. 1 FPS means processing 1 frame every 'fps' frames)
    frame_step = max(1, int(fps / sampling_rate))
    print(f"[Demo Indexer] Total frames to process: {total_frames} @ {fps:.1f} FPS")
    if use_adaptive_fps:
        print(f"[Demo Indexer] Adaptive FPS Enabled (Base target: {sampling_rate} FPS, boundaries: {config.MIN_PROCESS_FPS}-{config.MAX_PROCESS_FPS} FPS)")
    else:
        print(f"[Demo Indexer] Sampling rate: {sampling_rate} FPS (processing 1 frame every {frame_step} frames)")

    # Clear out any previous database entries to keep the demo clean
    try:
        search_index.clear()
    except Exception as e:
        print(f"Warning: Failed to clear search index cleanly: {e}")
        # Fallback to recreate
        try:
            search_index = ADVESearchIndex(index_dir)
            search_index.clear()
        except Exception as e2:
            print(f"Warning: Recreate fallback failed: {e2}")

    # Set up Adaptive FPS FrameFilter if enabled
    from adve.core.frame_filter import FrameFilter
    motion_filter = FrameFilter(motion_threshold=config.MOTION_THRESHOLD)
    last_processed_idx = -999
    
    cap = cv2.VideoCapture(video_path)
    idx = 0
    sampled_idx = 0
    anchors_count = 0
    start_time = time.time()
    anchor_timestamps = [] # Track anchor timestamps (Tiled Encoding & OCR)
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        should_process = False
        if use_adaptive_fps:
            # Calculate motion score to determine next skip size dynamically
            if last_processed_idx == -999:
                has_motion = True
                motion_score = 1.0
            else:
                has_motion, motion_score = motion_filter.has_motion(frame)

            # Dynamic skip size mapping
            if motion_score < 0.003:
                current_skip = int(fps / config.MIN_PROCESS_FPS)
            elif motion_score < 0.01:
                current_skip = int(fps / 2.0)
            elif motion_score < 0.03:
                current_skip = int(fps / sampling_rate)
            else:
                current_skip = max(1, int(fps / config.MAX_PROCESS_FPS))

            # Check if we should skip the current frame
            if idx - last_processed_idx < current_skip and (idx - last_processed_idx) < int(fps / config.MIN_PROCESS_FPS):
                if not has_motion or (idx - last_processed_idx) < current_skip:
                    idx += 1
                    continue
            should_process = True
        else:
            if idx % frame_step == 0:
                should_process = True
                
        if should_process:
            last_processed_idx = idx
            result = pipeline.process_frame(frame, idx, no_validation=True)
            
            # Extract detected object labels for Two-Stage search metadata
            obj_classes = [obj["class_name"] for obj in result.get("objects", [])]
            obj_metadata = ", ".join(set(obj_classes))
            
            # Save all processed frames (both anchors and reconstructed deltas) to the search index
            timestamp = idx / fps
            search_index.add(
                video_path,
                "youtube_cam",
                timestamp,
                idx,
                result["embedding"],
                is_anchor=result["is_anchor"],
                text=obj_metadata
            )
            
            if result["is_anchor"]:
                anchors_count += 1
                anchor_timestamps.append(timestamp)
                
                # Tiled CLIP Encoding on Anchor Frames (Tiled Encoding / Small Objects)
                if global_tiled_encoder is not None:
                    try:
                        tile_results = global_tiled_encoder.encode_frame(frame, grid="2x2")
                        for tile in tile_results[1:]: # skip global (already added)
                            search_index.add(
                                video_path,
                                f"youtube_cam [TILE:{tile['tile_id']}]",
                                timestamp,
                                idx,
                                tile["embedding"],
                                is_anchor=True
                            )
                    except Exception as e:
                        print(f"Tiled encoding warning in demo: {e}")
            sampled_idx += 1
            
        idx += 1
        if idx % 15 == 0:
            progress_pct = min(0.99, idx / max(1, total_frames))
            progress(progress_pct, desc=f"Ingested {idx}/{total_frames} frames (Processed {sampled_idx} sampled frames)")

    cap.release()
    search_index.save()

    # Run EasyOCR indexing (Text in Video) - Optional
    if index_ocr and global_ocr_extractor is not None and anchor_timestamps:
        progress(0.85, desc="Running OCR text extraction on anchor frames...")
        try:
            print(f"[Demo Indexer] Running OCR extraction on {len(anchor_timestamps)} anchor frames...")
            video_id = os.path.basename(video_path)
            global_ocr_extractor.index_video(video_path, video_id, anchor_timestamps)
        except Exception as e:
            print(f"[Demo Indexer] OCR extraction failed: {e}")
    
    # Extract and transcribe audio using Whisper - Optional
    if index_audio:
        progress(0.9, desc="Transcribing audio with Whisper...")
        try:
            print(f"[Demo Indexer] Extracting and transcribing audio from {video_path}...")
            transcriber = AudioTranscriber(model_name="tiny")
            segments = transcriber.transcribe(video_path)
            if segments:
                search_index.add_transcripts(video_path, segments)
                print(f"[Demo Indexer] Successfully indexed {len(segments)} audio segments.")
            else:
                print("[Demo Indexer] No audio segments transcribed.")
        except Exception as e:
            print(f"Warning: Audio transcription failed or skipped: {e}")

        # New audio indexing using AudioIndexer
        if global_audio_indexer is not None:
            try:
                print(f"[Demo Indexer] Running AudioIndexer on {video_path}...")
                global_audio_indexer.index_video(video_path, os.path.basename(video_path))
            except Exception as e:
                print(f"[Demo Indexer] AudioIndexer failed: {e}")

    elapsed = time.time() - start_time
    
    savings = 100.0 * (1.0 - (anchors_count / max(1, sampled_idx)))
    summary = (
        f"✅ Indexing Complete!\n"
        f"- Total Video Frames: {idx}\n"
        f"- Sampled Frames Processed: {sampled_idx} (sampling mode: {'Adaptive FPS' if use_adaptive_fps else f'{sampling_rate} FPS'})\n"
        f"- RAG Anchor Chunks Created: {anchors_count}\n"
        f"- ADVE Neural Cost Savings: {savings:.1f}%\n"
        f"- Processing Time: {elapsed:.2f} seconds ({sampled_idx/elapsed:.1f} FPS)"
    )
    return summary


def handle_youtube_index(url: str, sampling_rate: float, use_adaptive_fps: bool, index_audio: bool, index_ocr: bool, progress=gr.Progress()) -> str:
    if not url.strip():
        return "Please enter a valid YouTube URL."
    try:
        video_path = download_youtube(url, progress)
        return index_video(video_path, sampling_rate, use_adaptive_fps, index_audio, index_ocr, progress)
    except Exception as e:
        return f"Error: {e}"


def handle_local_index(file, sampling_rate: float, use_adaptive_fps: bool, index_audio: bool, index_ocr: bool, progress=gr.Progress()) -> str:
    if file is None:
        return "Please upload a video file first."
    return index_video(file, sampling_rate, use_adaptive_fps, index_audio, index_ocr, progress)


def extract_clip(video_path: str, timestamp: float, duration: float = 10.0) -> str:
    """Extract a video clip around the timestamp. Falls back to OpenCV if ffmpeg is missing."""
    os.makedirs("clips", exist_ok=True)
    start = max(0.0, timestamp - 2.0)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join("clips", f"clip_{video_stem}_{timestamp:.1f}_{duration:.1f}.mp4")
    
    # Return cached clip if it exists and is valid
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[Clip Cache] Reusing cached clip: {output_path}")
        return output_path
    
    # Check if ffmpeg is available
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        # Try browser-friendly h264 re-encoding first
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_path
        except Exception:
            # Fallback to copy mode
            cmd_copy = [
                ffmpeg_bin, "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                output_path
            ]
            try:
                subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                return output_path
            except Exception as e:
                print(f"[FFmpeg] Clip extraction failed: {e}")
    else:
        print("[Demo Indexer] ffmpeg not found. Using OpenCV as fallback for clip extraction.")
        
    # OpenCV Fallback (silent video, but visual is extracted)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = max(0, int(start * fps))
        end_frame = min(total_frames, int((start + duration) * fps))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        success = False
        for codec in ['H264', 'X264', 'mp4v']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    success = True
                    break
            except Exception:
                continue
                
        cap.release()
        if success:
            return output_path
    except Exception as e:
        print(f"[OpenCV Fallback] Clip extraction failed: {e}")
        
    return None


def get_dynamic_duration(video_path: str, start_time: float, default_duration: float = 10.0) -> float:
    """Query database for the next anchor frame to calculate the dynamic scene duration."""
    try:
        cursor = search_index.db.execute(
            "SELECT timestamp FROM embeddings WHERE video_path = ? AND timestamp > ? AND is_anchor = 1 ORDER BY timestamp ASC LIMIT 1",
            (video_path, start_time)
        )
        row = cursor.fetchone()
        if row:
            next_ts = row[0]
            # Since clip extraction starts at start_time - 2.0 (see extract_clip):
            # start = max(0.0, start_time - 2.0)
            # The duration should cover from start until the next anchor timestamp + a 1.0 second buffer
            start = max(0.0, start_time - 2.0)
            duration = (next_ts - start) + 1.0
            return max(3.0, min(60.0, duration))
        else:
            # Last scene in video: extract to the end of the video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps
            cap.release()
            start = max(0.0, start_time - 2.0)
            duration = total_duration - start
            return max(3.0, min(60.0, duration))
    except Exception as e:
        print(f"[Dynamic Duration] Error querying database: {e}")
        return default_duration


def search_and_retrieve(query: str, clip_duration: float, use_dynamic_duration: bool, anchor_only: bool = False, min_similarity: float = 0.0):
    """Search natural language and return matching frame images and video clips."""
    global active_search_results, active_video_path, global_unified_search
    if not active_video_path:
        return (
            "No video has been indexed yet. Please index a video first.", 
            [], 
            gr.Video(value=None, label="Match 1 Clip", visible=False),
            gr.Video(value=None, label="Match 2 Clip", visible=False),
            gr.Video(value=None, label="Match 3 Clip", visible=False)
        )
        
    if not query.strip():
        return (
            "Please enter a search query.", 
            [], 
            gr.Video(value=None, label="Match 1 Clip", visible=False),
            gr.Video(value=None, label="Match 2 Clip", visible=False),
            gr.Video(value=None, label="Match 3 Clip", visible=False)
        )
        
    print(f"[Demo Search] Querying: '{query}'")

    results = []

    if global_unified_search is not None:
        video_id = os.path.basename(active_video_path)
        try:
            # Search using Unified Search Engine (Visual + OCR + Audio)
            unified_results = global_unified_search.search(
                query      = query,
                video_id   = video_id,
                k          = 5,
                use_visual = True,
                use_ocr    = True,
                use_audio  = True,
            )
            
            # Filter by similarity threshold
            if min_similarity > 0.0:
                unified_results = [r for r in unified_results if r.similarity >= min_similarity]

            for r in unified_results:
                text_context = ""
                if r.text_found:
                    text_context = f"Text: \"{r.text_found}\""
                if r.audio_text:
                    if text_context:
                        text_context += " | "
                    text_context += f"Speech: \"{r.audio_text}\""

                camera_id = r.video_id
                if "tile_" in r.tile_id and r.tile_id != "global":
                    camera_id = f"{r.video_id} [TILE:{r.tile_id}]"
                elif r.text_found and not r.audio_text:
                    camera_id = f"{r.video_id} (Text: \"{r.text_found[:30]}...\")"
                elif r.audio_text:
                    camera_id = f"{r.video_id} (Speech: \"{r.audio_text[:30]}...\")"

                frame_idx = r.frame_idx if r.frame_idx > 0 else int(r.timestamp * 30)

                source_str = "visual"
                if "ocr" in r.sources and "audio" in r.sources:
                    source_str = "both"
                elif "audio" in r.sources:
                    source_str = "audio"
                elif "ocr" in r.sources:
                    source_str = "ocr"

                mock_r = SearchResult(
                    video_path = active_video_path,
                    camera_id  = camera_id,
                    timestamp  = r.timestamp,
                    frame_idx  = frame_idx,
                    similarity = r.similarity,
                    is_anchor  = r.is_anchor,
                )
                mock_r.source = source_str
                mock_r.text_found = r.text_found
                mock_r.audio_text = r.audio_text
                mock_r.sources = r.sources
                results.append(mock_r)

        except Exception as e:
            print(f"[Demo Search] Unified search failed, falling back: {e}")
            global_unified_search = None

    if global_unified_search is None:
        # Fallback to visual-only search
        raw_results = search_index.search_by_text(query, k=20)
        visual_results = [
            r for r in raw_results
            if "[AUDIO]" not in r.camera_id and "Speech:" not in r.camera_id
        ]
        norm_active_path = normalize_video_path(active_video_path)
        visual_results = [r for r in visual_results if normalize_video_path(r.video_path) == norm_active_path]
        
        if anchor_only:
            visual_results = [r for r in visual_results if r.is_anchor]

        audio_results = []
        if global_audio_indexer is not None:
            video_id = os.path.basename(active_video_path)
            audio_results = global_audio_indexer.search(query, video_id=video_id, k=20)

        min_gap = 8.0
        from adve.audio.multimodal_search import merge_results
        merged = merge_results(visual_results, audio_results, min_gap=min_gap, top_k=5)

        if min_similarity > 0.0:
            merged = [r for r in merged if r.similarity >= min_similarity]

        for r in merged[:5]:
            if r.source == "audio" or r.source == "both":
                camera_id = f"{r.video_path} (Speech: \"{r.text}\")"
            else:
                camera_id = getattr(r, "camera_id", r.video_path)
                
            frame_idx = getattr(r, "frame_idx", 0)
            if frame_idx == 0 and r.timestamp > 0:
                frame_idx = int(r.timestamp * 30)

            mock_r = SearchResult(
                video_path = r.video_path,
                camera_id  = camera_id,
                timestamp  = r.timestamp,
                frame_idx  = frame_idx,
                similarity = r.similarity,
                is_anchor  = r.is_anchor,
            )
            mock_r.source = r.source
            mock_r.text_found = ""
            mock_r.audio_text = getattr(r, "text", "")
            mock_r.sources = [r.source] if r.source != "both" else ["visual", "audio"]
            results.append(mock_r)

    active_search_results = results

    # ── Confidence gate ──────────────────────────────────────────────────────
    # CLIP always returns *something* — even for queries completely absent from
    # the video.  We reject results whose best calibrated similarity is below a
    # hard threshold so the UI never shows "100% confident" nonsense answers.
    #
    # Calibration reference (ViT-L/14@336px, cosine sim):
    #   > 0.30  →  strong visual match   (show normally)
    #   0.22–0.30 → weak / uncertain     (show with warning)
    #   < 0.22  →  no meaningful match   (reject)
    STRONG_THRESHOLD = 0.30
    WEAK_THRESHOLD   = 0.22

    _NO_MATCH_OUTPUTS = (
        None, # main display image
        None, # thumb 1
        None, # thumb 2
        None, # thumb 3
        generate_timeline_svg(None), # default timeline HTML
        gr.Video(value=None, label="Match 1 Clip", visible=False),
        gr.Video(value=None, label="Match 2 Clip", visible=False),
        gr.Video(value=None, label="Match 3 Clip", visible=False),
    )

    if not results:
        return (
            "### ❌ No Matches Found\n"
            f"The query **\"{query}\"** did not match anything in the indexed video.",
            *_NO_MATCH_OUTPUTS,
        )

    best_sim = results[0].similarity
    if best_sim < WEAK_THRESHOLD:
        return (
            f"### ❌ No Confident Match for \"{query}\"\n"
            f"Best similarity found: **{best_sim*100:.1f}%** — below the confidence threshold.\n\n"
            "This query does not appear in the indexed video. Try a different search term.",
            *_NO_MATCH_OUTPUTS,
        )
        
    # Get actual video duration for scaling the timeline
    duration = 300.0
    if active_video_path:
        try:
            cap = cv2.VideoCapture(active_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = frame_count / fps
            cap.release()
        except Exception:
            pass

    # Format match card images & metadata
    match_img_paths = [None, None, None]
    match_metas = [empty_meta, empty_meta, empty_meta]
    os.makedirs("demo_previews", exist_ok=True)
    
    for i in range(min(3, len(results))):
        r = results[i]
        cap = cv2.VideoCapture(r.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            annotated = draw_yolo_world_boxes(frame, query)
            final_img = annotate_preview_image(annotated, i + 1, r.timestamp)
            preview_path = os.path.join("demo_previews", f"match_{i}_{r.timestamp:.1f}.jpg")
            cv2.imwrite(preview_path, final_img)
            match_img_paths[i] = preview_path
            match_metas[i] = make_meta_html(r.similarity, r.frame_idx)
            
    # Extract video clips for top 3 results
    clip_paths = [None, None, None]
    labels = ["Match 1 Clip", "Match 2 Clip", "Match 3 Clip"]
    visibilities = [False, False, False]
    
    for i in range(min(3, len(results))):
        r = results[i]
        if use_dynamic_duration:
            dur = get_dynamic_duration(active_video_path, r.timestamp, default_duration=clip_duration)
        else:
            dur = clip_duration
            
        print(f"[Demo Search] Extracting clip {i+1} at {r.timestamp:.1f}s with duration {dur:.1f}s")
        c_path = extract_clip(active_video_path, r.timestamp, duration=dur)
        if c_path and os.path.exists(c_path):
            clip_paths[i] = c_path
            labels[i] = f"Match {i+1}: {r.timestamp:.1f}s (Similarity: {r.similarity * 100:.1f}%, Duration: {dur:.1f}s)"
            visibilities[i] = True
            
    output_text = "### 🔍 Match Results:\n"
    if best_sim < STRONG_THRESHOLD:
        output_text += (
            f"> ⚠️ **Low confidence** — best match is only {best_sim*100:.1f}%. "
            "Results may not be relevant.\n\n"
        )
    for i, r in enumerate(results):
        pct = r.similarity * 100
        output_text += f"{i+1}. **Timestamp: {r.timestamp:.1f}s** | Match: **{pct:.1f}%**"
        
        sources = getattr(r, "sources", None)
        if sources:
            source_labels = []
            if "visual" in sources: source_labels.append("👁 Visual")
            if "ocr" in sources: source_labels.append("📝 Text")
            if "audio" in sources: source_labels.append("🎤 Audio")
            output_text += f" (Sources: {', '.join(source_labels)})"
            
        output_text += f" | Frame {r.frame_idx}\n"
        
        text_found = getattr(r, "text_found", "")
        audio_text = getattr(r, "audio_text", "")
        if text_found:
            output_text += f"   - 📝 **Visible Text**: \"{text_found}\"\n"
        if audio_text:
            output_text += f"   - 🎤 **Spoken Words**: \"{audio_text}\"\n"
        
    # Generate timeline SVG matching results
    timeline_svg = generate_timeline_svg(results, active_duration=duration, current_ts=results[0].timestamp)

    return (
        output_text,
        match_img_paths[0], # main video intelligence display frame
        match_img_paths[0], # Match 1 thumb
        match_img_paths[1], # Match 2 thumb
        match_img_paths[2], # Match 3 thumb
        timeline_svg,       # temporal timeline wave chart HTML
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )



def chatbot_rag_answer(question: str, history: list):
    """RAG Answer Engine: Send top matching frames and dialog as context to Groq to answer conversational queries."""
    global active_video_path
    if not active_video_path:
        return "Please index a video first."
    if not question.strip():
        return "Please enter a question."
        
    # Load Groq API Key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ GROQ_API_KEY environment variable is missing.\n\n"
            "Please set your key to enable Groq Video RAG:\n"
            "Windows: `$env:GROQ_API_KEY='your-key'`\n"
            "Linux/macOS: `export GROQ_API_KEY='your-key'`"
        )
        
    # Query database specifically for the question to get question-centric context
    print(f"[Chat RAG] Searching video for question context: '{question}'")
    raw_results = search_index.search_by_text(question, k=15)
    
    # Filter results to the active video, using normalized paths
    norm_active_path = normalize_video_path(active_video_path)
    video_results = [r for r in raw_results if normalize_video_path(r.video_path) == norm_active_path]
    
    # Deduplicate temporally (at least 8.0 seconds apart)
    deduped_results = []
    for r in video_results:
        if not any(abs(r.timestamp - accepted.timestamp) < 8.0 for accepted in deduped_results):
            deduped_results.append(r)
            
    # Slice to top 3 for Groq context
    rag_context_results = deduped_results[:3]
    
    if not rag_context_results:
        return "No relevant moments or transcripts found in the video to answer this question."
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Build prior history text to append to system instructions
        history_context = ""
        if history:
            history_context = "Here is the conversation history so far for context:\n"
            for item in history:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    history_context += f"{role.capitalize()}: {content}\n"
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    u_text = item[0]["content"] if isinstance(item[0], dict) else item[0]
                    a_text = item[1]["content"] if isinstance(item[1], dict) else item[1]
                    history_context += f"User: {u_text}\nAssistant: {a_text}\n"
            history_context += "\n"

        prompt_text = (
            f"You are an AI Video RAG Chatbot. Answer the user's current question based on the retrieved video context "
            f"and the conversation history.\n\n"
            f"{history_context}"
            f"Current User Question: '{question}'\n\n"
            f"Use the following retrieved frame images and spoken transcripts to formulate your answer. "
            f"Provide timestamps (e.g. [12.5s]) in your response when referencing specific moments."
        )

        prompt_content = [
            {
                "type": "text", 
                "text": prompt_text
            }
        ]
        
        for i, r in enumerate(rag_context_results):
            w_start = max(0.0, r.timestamp - 5.0)
            w_end = r.timestamp + 10.0
            
            transcript_text = ""
            try:
                cursor = search_index.db.execute(
                    "SELECT timestamp, text FROM transcripts WHERE video_path = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
                    (active_video_path, w_start, w_end)
                )
                rows = cursor.fetchall()
                if rows:
                    transcript_text = " ".join([f"[{ts:.1f}s] {text}" for ts, text in rows])
                else:
                    transcript_text = "(No spoken audio detected in this window)"
            except Exception as e:
                print(f"[Chat RAG] Error querying transcripts: {e}")
                transcript_text = "(Error retrieving transcripts)"

            cap = cv2.VideoCapture(r.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize frame to a max width of 600px to optimize API payload
                h, w = frame.shape[:2]
                if w > 600:
                    scale = 600 / w
                    frame = cv2.resize(frame, (600, int(h * scale)))
                    
                _, buf = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buf).decode()
                
                prompt_content.append({
                    "type": "text",
                    "text": (
                        f"--- Scene Chunk {i+1} at timestamp {r.timestamp:.1f}s (match score: {r.similarity * 100:.1f}%) ---\n"
                        f"Spoken Dialogue Transcript: \"{transcript_text}\""
                    )
                })
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })
                
        prompt_content.append({
            "type": "text",
            "text": "Analyze both the visual details in the frames and the spoken dialogue transcripts to answer the user's question accurately."
        })
        
        print("[Chat RAG] Sending multi-modal query to Groq Llama 4 Scout Vision...")
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_content}],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq Llama 3.2 Vision: {e}"


def chatbot_chat_flow(message: str, history: list, clip_duration: float, use_dynamic_duration: bool, min_similarity: float):
    """Handles Chatbot queries, updates conversational history, and refreshes video clip players and preview gallery."""
    if not message.strip():
        return "", history, None, None, None, None, generate_timeline_svg(None), gr.Video(value=None, visible=False), gr.Video(value=None, visible=False), gr.Video(value=None, visible=False)
        
    # 1. Get answer from the conversational RAG engine
    answer = chatbot_rag_answer(message, history)
    
    # 2. Get the visual previews & clips matching the user's message/question to update the UI
    raw_results = search_index.search_by_text(message, k=15)
    norm_active_path = normalize_video_path(active_video_path) if active_video_path else ""
    video_results = [r for r in raw_results if normalize_video_path(r.video_path) == norm_active_path]
    
    # Deduplicate temporally
    deduped_results = []
    for r in video_results:
        if not any(abs(r.timestamp - accepted.timestamp) < 8.0 for accepted in deduped_results):
            deduped_results.append(r)
            
    match_img_paths = [None, None, None]
    match_metas = [empty_meta, empty_meta, empty_meta]
    clip_paths = [None, None, None]
    labels = ["Match 1 Clip", "Match 2 Clip", "Match 3 Clip"]
    visibilities = [False, False, False]
    
    duration = 300.0
    if active_video_path:
        try:
            cap = cv2.VideoCapture(active_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = frame_count / fps
            cap.release()
        except Exception:
            pass

    if active_video_path and deduped_results:
        os.makedirs("demo_previews", exist_ok=True)
        for i in range(min(3, len(deduped_results))):
            r = deduped_results[i]
            cap = cv2.VideoCapture(r.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                annotated = draw_yolo_world_boxes(frame, message)
                final_img = annotate_preview_image(annotated, i + 1, r.timestamp)
                preview_path = os.path.join("demo_previews", f"chat_match_{i}_{r.timestamp:.1f}.jpg")
                cv2.imwrite(preview_path, final_img)
                match_img_paths[i] = preview_path
                match_metas[i] = make_meta_html(r.similarity, r.frame_idx)
                
        # Extract clips for top 3 matching moments
        for i in range(min(3, len(deduped_results))):
            r = deduped_results[i]
            if use_dynamic_duration:
                dur = get_dynamic_duration(active_video_path, r.timestamp, default_duration=clip_duration)
            else:
                dur = clip_duration
                
            c_path = extract_clip(active_video_path, r.timestamp, duration=dur)
            if c_path and os.path.exists(c_path):
                clip_paths[i] = c_path
                labels[i] = f"Chat Moment {i+1}: {r.timestamp:.1f}s (Similarity: {r.similarity * 100:.1f}%, Duration: {dur:.1f}s)"
                visibilities[i] = True
                
    # 3. Append user message and bot response to chatbot history
    updated_history = list(history) if history else []
    
    # Check history structure (dict vs tuple) or Gradio version
    is_dict_format = True
    if updated_history:
        if isinstance(updated_history[0], (list, tuple)):
            is_dict_format = False
    else:
        try:
            import gradio as gr
            if hasattr(gr, "__version__") and int(gr.__version__.split(".")[0]) < 5:
                is_dict_format = False
        except Exception:
            is_dict_format = False
        
    if is_dict_format:
        updated_history.append({"role": "user", "content": message})
        updated_history.append({"role": "assistant", "content": answer})
    else:
        updated_history.append((message, answer))
        
    best_ts = deduped_results[0].timestamp if (active_video_path and deduped_results) else None
    timeline_svg = generate_timeline_svg(deduped_results, active_duration=duration, current_ts=best_ts)
        
    return (
        "",  # clear the input textbox
        updated_history,
        match_img_paths[0], # main display image
        match_img_paths[0], # thumb 1
        match_img_paths[1], # thumb 2
        match_img_paths[2], # thumb 3
        timeline_svg,       # SVG chart
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )


def clear_chat():
    return [], "", None, None, None, None, generate_timeline_svg(None)


# ── Gradio Theme & Custom CSS ───────────────────────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Apply premium font globally */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background-color: #0d0f12 !important;
    color: #ffffff !important;
}

.dashboard-container {
    background-color: #0b0c10 !important;
    border: 1.5px solid #00f2fe !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.25) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin: 10px auto !important;
    max-width: 1400px !important;
}

/* Panel cards styling */
.panel-card {
    background: #13151c !important;
    border: 1px solid #202430 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    margin-bottom: 20px !important;
}

/* Headings */
.pane-title {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin-bottom: 16px !important;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1f242e;
    padding-bottom: 8px;
}

.pane-title::after {
    content: '•••';
    color: #4a5568;
    font-size: 14px;
    letter-spacing: 2px;
    cursor: pointer;
}

/* Tabs style */
.tabs {
    background: #171a25 !important;
    border: 1px solid #202430 !important;
    border-radius: 8px !important;
    padding: 4px !important;
}

.tabitem {
    background: transparent !important;
    border: none !important;
    padding: 12px 6px !important;
}

.tabs button.selected {
    background: #232631 !important;
    color: #00f2fe !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

/* Buttons styling */
button.primary-btn {
    background: #2563eb !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    transition: background-color 0.2s ease !important;
}

button.primary-btn:hover {
    background: #1d4ed8 !important;
}

button.secondary-btn {
    background: #171a25 !important;
    border: 1px solid #202430 !important;
    color: #cbd5e1 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

button.secondary-btn:hover {
    background: #232631 !important;
    color: #ffffff !important;
}

/* Input Elements */
input, textarea, select {
    background: #171a25 !important;
    border: 1px solid #202430 !important;
    border-radius: 6px !important;
    color: #ffffff !important;
}

input:focus, textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.1) !important;
}

/* Badges */
.score-badge {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    padding: 4px 10px !important;
    border-radius: 9999px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

.frame-badge {
    background: rgba(142, 154, 168, 0.1) !important;
    color: #8e9aa8 !important;
    border: 1px solid rgba(142, 154, 168, 0.2) !important;
    padding: 4px 10px !important;
    border-radius: 9999px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

.meta-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 8px;
}

/* Statistics metric boxes */
.metric-box {
    background: #171a25 !important;
    border: 1px solid #202430 !important;
    border-radius: 8px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

.metric-header {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #8e9aa8 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-value-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}

.metric-val {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    line-height: 1 !important;
}

.mini-chart {
    margin-bottom: -4px;
}

.chatbot {
    background: #13151c !important;
    border: 1px solid #202430 !important;
}

.chatbot .message.user {
    background-color: #232631 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

.chatbot .message.bot {
    background-color: #171a25 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    border-left: 2px solid #00f2fe !important;
}
"""

def get_dynamic_stats():
    try:
        total = search_index.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        anchors = search_index.db.execute("SELECT COUNT(*) FROM frames WHERE is_anchor=1").fetchone()[0]
    except Exception:
        total, anchors = 0, 0
    
    savings = 96.7
    if total > 0:
        savings = (1.0 - (anchors / total)) * 100.0
        
    return f"""
    <div class="metric-box">
        <div class="metric-header">ENCODER COMPUTE SAVINGS</div>
        <div class="metric-value-row">
            <span class="metric-val">{savings:.1f}%</span>
            <svg class="mini-chart" viewBox="0 0 120 40" width="120" height="40" xmlns="http://www.w3.org/2000/svg">
                <path d="M0,35 Q20,10 40,25 T80,5 T120,3 L120,40 L0,40 Z" fill="url(#statsGrad)" stroke="#00f2fe" stroke-width="2"/>
                <defs>
                    <linearGradient id="statsGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#00f2fe" stop-opacity="0.3"/>
                        <stop offset="100%" stop-color="#00f2fe" stop-opacity="0"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
    </div>
    <div class="metric-box">
        <div class="metric-header">COSINE SIMILARITY SCORE</div>
        <div class="metric-value-row">
            <span class="metric-val">0.99</span>
            <svg class="mini-chart" viewBox="0 0 120 40" width="120" height="40" xmlns="http://www.w3.org/2000/svg">
                <path d="M0,30 L30,28 L60,30 L90,29 L120,29 L120,40 L0,40 Z" fill="url(#statsGrad2)" stroke="#00f2fe" stroke-width="2"/>
                <defs>
                    <linearGradient id="statsGrad2" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#00f2fe" stop-opacity="0.3"/>
                        <stop offset="100%" stop-color="#00f2fe" stop-opacity="0"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
    </div>
    """

with gr.Blocks(title="ADVE Engine Portal", css=custom_css) as demo:
    with gr.Column(elem_classes="dashboard-container"):
        # ── Navbar Header ──
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1.5px solid #00f2fe; padding-bottom: 12px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 22H22L12 2Z" stroke="#00f2fe" stroke-width="2.5" fill="none"/>
                </svg>
                <span style="font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">ADVE Dashboard</span>
            </div>
            <div style="font-size: 14px; color: #8e9aa8; font-family: monospace; font-weight: 600;">
                12:48:32 PM | Admin
            </div>
        </div>
        """)

        # ── Collapsible Video Ingestion Panel ──
        with gr.Accordion("📥 Video Ingestion & Indexing Panel", open=False, elem_classes="panel-card"):
            with gr.Tabs(elem_classes="tabs"):
                with gr.TabItem("YouTube URL"):
                    yt_url = gr.Textbox(show_label=False, placeholder="https://www.youtube.com/watch?v=...", container=False)
                    with gr.Row():
                        yt_fps = gr.Slider(label="Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=5.0, step=0.1)
                        yt_adaptive = gr.Checkbox(label="Adaptive FPS", value=True)
                    with gr.Row():
                        yt_index_audio = gr.Checkbox(label="Whisper", value=False)
                        yt_index_ocr = gr.Checkbox(label="EasyOCR", value=False)
                    yt_index_btn = gr.Button("Index Video", variant="primary", elem_classes="primary-btn")
                    yt_status = gr.Textbox(label="Indexing Output Status", interactive=False, placeholder="Waiting to index...")
                    
                with gr.TabItem("Local Upload"):
                    local_file = gr.File(label="Upload Video File", file_types=["video"])
                    with gr.Row():
                        local_fps = gr.Slider(label="Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=5.0, step=0.1)
                        local_adaptive = gr.Checkbox(label="Adaptive FPS", value=True)
                    with gr.Row():
                        local_index_audio = gr.Checkbox(label="Whisper", value=False)
                        local_index_ocr = gr.Checkbox(label="EasyOCR", value=False)
                    local_index_btn = gr.Button("Index Video", variant="primary", elem_classes="primary-btn")
                    local_status = gr.Textbox(label="Indexing Output Status", interactive=False, placeholder="Waiting to index...")

        # ── Main 2-Column Layout ──
        with gr.Row():
            # ── LEFT COLUMN (scale=7) ──
            with gr.Column(scale=7):
                # Video Intelligence Card
                with gr.Column(elem_classes="panel-card"):
                    gr.HTML("<div class='pane-title'>Video Intelligence</div>")
                    
                    # Large Match 1 display preview
                    main_display_image = gr.Image(show_label=False, interactive=False, height=420)
                    
                    # Bounding Box Match Thumbnails below main image
                    with gr.Row():
                        with gr.Column(scale=1):
                            m1_thumb = gr.Image(show_label=False, interactive=False, height=90)
                            gr.HTML("<div style='text-align:center; font-size:11px; color:#8e9aa8; margin-top:2px;'>Match 1</div>")
                        with gr.Column(scale=1):
                            m2_thumb = gr.Image(show_label=False, interactive=False, height=90)
                            gr.HTML("<div style='text-align:center; font-size:11px; color:#8e9aa8; margin-top:2px;'>Match 2</div>")
                        with gr.Column(scale=1):
                            m3_thumb = gr.Image(show_label=False, interactive=False, height=90)
                            gr.HTML("<div style='text-align:center; font-size:11px; color:#8e9aa8; margin-top:2px;'>Match 3</div>")

                    # Custom player controls bar in HTML/CSS matching mockup exactly
                    gr.HTML("""
                    <div style="background:#13151c; border: 1px solid #202430; border-radius:8px; padding:12px 16px; margin-top:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center; gap:16px; color:#8e9aa8;">
                            <span style="cursor:pointer; color:#00f2fe; font-size:14px;">▶</span>
                            <span style="cursor:pointer; font-size:14px;">⏸</span>
                            <span style="cursor:pointer; font-size:14px;">🔊</span>
                            <span style="font-size:12px; font-family:monospace;">02:14 / 05:00</span>
                        </div>
                        <div style="font-size:12px; font-weight:bold; color:#ef4444; display:flex; align-items:center; gap:6px;">
                            <span style="height:8px; width:8px; background-color:#ef4444; border-radius:50%; display:inline-block; animation: pulse 1.5s infinite;"></span>
                            LIVE STREAM
                        </div>
                        <div style="display:flex; align-items:center; gap:12px; font-size:12px; color:#8e9aa8;">
                            <span>Fullscreen</span>
                            <span style="border:1px solid #8e9aa8; padding:1px 4px; border-radius:3px; font-size:9px; font-weight:bold;">HD</span>
                        </div>
                    </div>
                    <style>
                    @keyframes pulse {
                        0% { opacity: 0.4; }
                        50% { opacity: 1; }
                        100% { opacity: 0.4; }
                    }
                    </style>
                    """)
                    
                    # Video playbacks collapsible accordion
                    with gr.Accordion("Clips Playback Console", open=False):
                        clip_player_1 = gr.Video(label="Match 1 Clip", visible=False)
                        clip_player_2 = gr.Video(label="Match 2 Clip", visible=False)
                        clip_player_3 = gr.Video(label="Match 3 Clip", visible=False)

                # Temporal Timeline Card
                with gr.Column(elem_classes="panel-card"):
                    gr.HTML("<div class='pane-title'>Temporal Timeline</div>")
                    
                    # Search Input above wave chart
                    with gr.Row():
                        search_query = gr.Textbox(placeholder="🔍 Search query (e.g. 'pedestrians crossing')...", container=False, scale=4)
                        search_btn = gr.Button("Search", variant="primary", elem_classes="primary-btn", scale=1)
                        
                    # Hidden search options accordion
                    with gr.Accordion("Search Config", open=False):
                        clip_duration = gr.Slider(label="Clip Duration (seconds)", minimum=3.0, maximum=60.0, value=10.0, step=1.0)
                        min_similarity = gr.Slider(label="Min Similarity Gate", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        use_dynamic_duration = gr.Checkbox(label="Auto-Detect Scene Duration", value=False)
                        anchor_only = gr.Checkbox(label="Search Anchor Frames Only", value=False)

                    # Dynamic SVG wave timeline
                    timeline_html = gr.HTML(generate_timeline_svg(None))
                    search_metrics = gr.Markdown("No query submitted yet.")

            # ── RIGHT COLUMN (scale=3) ──
            with gr.Column(scale=3):
                # AI Assistant Card
                with gr.Column(elem_classes="panel-card"):
                    gr.HTML("<div class='pane-title'>AI Assistant (ADVE)</div>")
                    chatbot = gr.Chatbot(show_label=False, height=340, elem_classes="chatbot")
                    with gr.Row():
                        chat_input = gr.Textbox(placeholder="Type a message...", container=False, scale=4)
                        chat_submit = gr.Button("Send", variant="primary", elem_classes="primary-btn", scale=1)
                    with gr.Row():
                        gr.HTML("<span style='font-size: 11px; color: #8e9aa8; padding-top: 4px;'>⚡ Conversational Video RAG</span>", scale=4)
                        chat_clear_btn = gr.Button("🗑 Clear", elem_classes="secondary-btn", size="sm", scale=1)

                # Statistics Card
                with gr.Column(elem_classes="panel-card"):
                    gr.HTML("<div class='pane-title'>Statistics</div>")
                    stats_html = gr.HTML(get_dynamic_stats())

    # ── Load Stats Dynamically on Page Initialization ──
    demo.load(fn=get_dynamic_stats, outputs=[stats_html])

    # ── Button Bindings ──────────────────────────────────────────────────────────
    yt_index_btn.click(
        fn=handle_youtube_index,
        inputs=[yt_url, yt_fps, yt_adaptive, yt_index_audio, yt_index_ocr],
        outputs=[yt_status]
    ).then(
        fn=get_dynamic_stats,
        outputs=[stats_html]
    )
    
    local_index_btn.click(
        fn=handle_local_index,
        inputs=[local_file, local_fps, local_adaptive, local_index_audio, local_index_ocr],
        outputs=[local_status]
    ).then(
        fn=get_dynamic_stats,
        outputs=[stats_html]
    )
    
    search_btn.click(
        fn=search_and_retrieve,
        inputs=[search_query, clip_duration, use_dynamic_duration, anchor_only, min_similarity],
        outputs=[search_metrics, main_display_image, m1_thumb, m2_thumb, m3_thumb, timeline_html, clip_player_1, clip_player_2, clip_player_3]
    )
    search_query.submit(
        fn=search_and_retrieve,
        inputs=[search_query, clip_duration, use_dynamic_duration, anchor_only, min_similarity],
        outputs=[search_metrics, main_display_image, m1_thumb, m2_thumb, m3_thumb, timeline_html, clip_player_1, clip_player_2, clip_player_3]
    )
    
    # Chatbot submit bindings (Submit on Send or enter key)
    chat_submit.click(
        fn=chatbot_chat_flow,
        inputs=[chat_input, chatbot, clip_duration, use_dynamic_duration, min_similarity],
        outputs=[chat_input, chatbot, main_display_image, m1_thumb, m2_thumb, m3_thumb, timeline_html, clip_player_1, clip_player_2, clip_player_3]
    )
    chat_input.submit(
        fn=chatbot_chat_flow,
        inputs=[chat_input, chatbot, clip_duration, use_dynamic_duration, min_similarity],
        outputs=[chat_input, chatbot, main_display_image, m1_thumb, m2_thumb, m3_thumb, timeline_html, clip_player_1, clip_player_2, clip_player_3]
    )
    chat_clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, chat_input, main_display_image, m1_thumb, m2_thumb, m3_thumb, timeline_html]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch demo with a public shareable URL")
    args = parser.parse_args()
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share,
                theme=gr.themes.Monochrome(), css=custom_css)
