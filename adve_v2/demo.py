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
<div class="match-meta">
    <span class="badge score">Score --</span>
    <span class="badge frame">Frame --</span>
</div>
"""

def make_meta_html(score: float, frame_idx: int):
    return f"""
    <div class="match-meta">
        <span class="badge score">Score {score:.3f}</span>
        <span class="badge frame">Frame #{frame_idx}</span>
    </div>
    """

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
        None, empty_meta,
        None, empty_meta,
        None, empty_meta,
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
        
    return (
        output_text,
        match_img_paths[0], match_metas[0],
        match_img_paths[1], match_metas[1],
        match_img_paths[2], match_metas[2],
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )



def extract_message_text(item) -> str:
    if hasattr(item, "content"):
        content = item.content
    elif isinstance(item, dict):
        content = item.get("content", "")
    else:
        content = item

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from list of message dictionaries/objects
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
            elif hasattr(c, "text"):
                texts.append(c.text)
        return " ".join(texts)
    elif isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return str(content)
    return str(content)

def extract_message_role(item) -> str:
    if hasattr(item, "role"):
        return item.role
    elif isinstance(item, dict):
        return item.get("role", "user")
    return "user"


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
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    u_text = extract_message_text(item[0])
                    a_text = extract_message_text(item[1])
                    history_context += f"User: {u_text}\nAssistant: {a_text}\n"
                else:
                    role = extract_message_role(item)
                    content = extract_message_text(item)
                    history_context += f"{role.capitalize()}: {content}\n"
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
        return "", history, None, empty_meta, None, empty_meta, None, empty_meta, gr.Video(value=None, visible=False), gr.Video(value=None, visible=False), gr.Video(value=None, visible=False)
        
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
        updated_history.append(gr.ChatMessage(role="user", content=message))
        updated_history.append(gr.ChatMessage(role="assistant", content=answer))
    else:
        updated_history.append((message, answer))
        
    return (
        "",  # clear the input textbox
        updated_history,
        match_img_paths[0], match_metas[0],
        match_img_paths[1], match_metas[1],
        match_img_paths[2], match_metas[2],
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )


def clear_chat():
    return [], "", None, empty_meta, None, empty_meta, None, empty_meta


# ── Gradio Theme & Custom CSS ───────────────────────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

/* ── CSS Custom Properties from HTML design ── */
:root {
    --bg: #f6f7fb;
    --bg-soft: #eef1fa;
    --card: #ffffff;
    --border: #e6e9f2;
    --border-soft: #eef0f6;
    --text: #0e1220;
    --muted: #6b7280;
    --muted-soft: #9aa2b1;
    --blue: #3466ff;
    --blue-dark: #1d4fd6;
    --blue-light: #eef2ff;
    --violet: #7c5cff;
    --grad-a: #3b82f6;
    --grad-b: #7c5cff;
    --green-bg: #e9f9f0;
    --green-text: #149a63;
    --radius-lg: 16px;
    --radius-md: 11px;
    --radius-sm: 9px;
    --shadow: 0 4px 12px rgba(16,19,34,0.03), 0 12px 32px rgba(16,19,34,0.04);
    --shadow-lift: 0 12px 30px rgba(16,19,34,0.08), 0 24px 60px rgba(16,19,34,0.1);
    --ring: 0 0 0 4px rgba(52,102,255,0.22);
}

.dark {
    --bg: #070913;
    --bg-soft: #0c0f1d;
    --card: rgba(18, 22, 41, 0.7);
    --border: rgba(255, 255, 255, 0.08);
    --border-soft: rgba(255, 255, 255, 0.04);
    --text: #f1f3f9;
    --muted: #9fa7c1;
    --muted-soft: #636b85;
    --blue: #4f80ff;
    --blue-dark: #3765e2;
    --blue-light: rgba(52, 102, 255, 0.15);
    --violet: #9b86ff;
    --grad-a: #4f80ff;
    --grad-b: #9b86ff;
    --green-bg: rgba(63, 214, 148, 0.15);
    --green-text: #3fd694;
    --shadow: 0 4px 12px rgba(0,0,0,0.25), 0 12px 32px rgba(0,0,0,0.3);
    --shadow-lift: 0 16px 36px rgba(0,0,0,0.35), 0 32px 72px rgba(0,0,0,0.45);
    --ring: 0 0 0 4px rgba(79,128,255,0.3);
}

/* ── Animations ── */
@keyframes rise { to { opacity:1; transform:translateY(0); } }
@keyframes bob { 0%,100%{ transform:translateY(0) rotate(var(--r,0deg)); } 50%{ transform:translateY(-9px) rotate(var(--r,0deg)); } }
@keyframes fadein { from { opacity:0; } to { opacity:1; } }
@keyframes spin { to { transform:rotate(360deg); } }

@media (prefers-reduced-motion: reduce) {
    * { animation-duration:0.001ms !important; animation-iteration-count:1 !important; transition-duration:0.001ms !important; }
}

/* ── Global Typography ── */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    box-sizing: border-box;
}
h1, h3, .logo span { font-family: 'Space Grotesk', Inter, sans-serif !important; }
::selection { background: var(--blue); color: #fff; }

body, .gradio-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    transition: background 0.35s cubic-bezier(0.4, 0, 0.2, 1), color 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Panel cards styling (maps to HTML .card with glassmorphism) ── */
.panel-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 22px !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 20px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    transition: background 0.35s ease, border-color 0.35s ease, box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    opacity: 0;
    transform: translateY(10px);
    animation: rise 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.panel-card:hover {
    box-shadow: var(--shadow-lift) !important;
    transform: translateY(-2px) !important;
}

/* ── Card Headings (maps to HTML .card-head) ── */
.pane-title {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 18px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}
.pane-title svg { width: 16px; height: 16px; color: var(--blue); }
.step-num {
    width: 24px; height: 24px; border-radius: 50%;
    background: linear-gradient(135deg, var(--grad-a), var(--blue-dark));
    color: #fff; font-size: 12.5px; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
}

/* ── Tabs (maps to HTML .tabs / .tab) ── */
.tabs {
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
    margin-bottom: 18px !important;
}
.tabitem {
    background: transparent !important;
    border: none !important;
    padding: 10px 4px !important;
}
.tabs button {
    flex: 1; text-align: center;
    font-size: 13.5px !important; font-weight: 600 !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    background: none !important;
    transition: color 0.2s ease !important;
}
.tabs button.selected {
    color: var(--blue) !important;
    border-color: var(--blue) !important;
    background: transparent !important;
}

/* ── Checkbox card styling (maps to HTML .check-box) ── */
.checkbox-card {
    border: 1.5px solid var(--border) !important;
    background: var(--bg-soft) !important;
    border-radius: var(--radius-sm) !important;
    padding: 9px 8px !important;
    text-align: left !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    cursor: pointer !important;
    transition: border-color 0.25s ease, background 0.25s ease, transform 0.2s ease !important;
}
.checkbox-card:hover {
    transform: translateY(-2px) !important;
    border-color: var(--blue) !important;
}
.checkbox-card input[type="checkbox"]:checked ~ span,
.checkbox-card:has(input:checked) {
    border-color: var(--blue) !important;
    background: var(--blue-light) !important;
}
.checkbox-card input[type="checkbox"] {
    width: 14px !important; height: 14px !important;
    accent-color: var(--blue) !important;
    cursor: pointer !important;
}
.checkbox-card span {
    font-size: 12.5px !important; font-weight: 700 !important;
    color: var(--muted) !important; line-height: 1.2 !important; display: block !important;
}
.chk-adaptive span::after {
    content: "Motion-adaptive" !important; display: block !important;
    font-size: 10.5px !important; font-weight: 400 !important; color: var(--muted-soft) !important; margin-top: 2px !important;
}
.chk-whisper span::after {
    content: "Transcripts" !important; display: block !important;
    font-size: 10.5px !important; font-weight: 400 !important; color: var(--muted-soft) !important; margin-top: 2px !important;
}
.chk-easyocr span::after {
    content: "OCR Text" !important; display: block !important;
    font-size: 10.5px !important; font-weight: 400 !important; color: var(--muted-soft) !important; margin-top: 2px !important;
}
.option-card-row {
    gap: 10px !important;
    margin-bottom: 14px !important;
}

/* ── Primary Button (maps to HTML .btn-block / .btn-primary) ── */
button.primary-btn {
    width: 100%;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    background: linear-gradient(135deg, var(--grad-a), var(--blue-dark)) !important;
    color: #fff !important; border: none !important;
    padding: 12px !important; border-radius: var(--radius-sm) !important;
    font-weight: 700 !important; font-size: 14px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(52,102,255,0.3) !important;
}
button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(52,102,255,0.45) !important;
    filter: brightness(1.1) !important;
}
button.primary-btn:active {
    transform: translateY(1px) !important;
}

/* ── Secondary / Ghost Button (maps to HTML .btn-ghost) ── */
button.secondary-btn {
    display: flex; align-items: center; gap: 6px;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    padding: 7px 12px !important; border-radius: 8px !important;
    color: var(--text) !important; font-weight: 600 !important; font-size: 12px !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}
button.secondary-btn:hover {
    background: var(--bg-soft) !important;
    transform: translateY(-1px) !important;
}

/* ── Input Elements ── */
input, textarea, select {
    background: var(--bg-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--blue) !important;
    box-shadow: var(--ring) !important;
    background: var(--card) !important;
}
input::placeholder, textarea::placeholder {
    color: var(--muted-soft) !important;
}

/* ── Status box styling ── */
.status-box textarea {
    background: var(--bg-soft) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px !important;
}

/* ── Match Card Container (maps to HTML .match with scaling) ── */
.match-card {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    padding: 6px !important;
    background: var(--card) !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease !important;
    cursor: pointer;
}
.match-card:hover {
    transform: translateY(-6px) scale(1.02) !important;
    box-shadow: 0 16px 36px rgba(16,19,34,0.08), 0 4px 12px rgba(52,102,255,0.08) !important;
    border-color: var(--blue) !important;
}
.dark .match-card:hover {
    box-shadow: 0 16px 36px rgba(0,0,0,0.45), 0 4px 12px rgba(79,128,255,0.18) !important;
}

/* ── Badges (maps to HTML .badge) ── */
.match-meta {
    display: flex;
    gap: 6px;
    padding: 8px;
    justify-content: center;
    margin-top: 8px;
}
.badge {
    font-size: 10.5px; font-weight: 700;
    padding: 3px 7px; border-radius: 6px;
}
.badge.score {
    background: var(--green-bg) !important;
    color: var(--green-text) !important;
}
.badge.frame {
    background: var(--bg-soft) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}

/* ── Chat bubble overrides ── */
.gradio-container .chatbot-wrap .message.user,
.gradio-container .message.user,
.chatbot .message.user {
    background: linear-gradient(135deg, var(--grad-a), var(--blue-dark)) !important;
    color: #fff !important;
    border-radius: 12px 12px 2px 12px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(52,102,255,0.2) !important;
}
.gradio-container .chatbot-wrap .message.bot,
.gradio-container .message.bot,
.chatbot .message.bot,
.chatbot .message.assistant {
    background: var(--bg-soft) !important;
    color: var(--text) !important;
    border-radius: 12px 12px 12px 2px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
}

/* ── Stat rows (maps to HTML .stat-row) ── */
.stat-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid var(--border-soft); font-size: 13.5px;
}
.stat-row:last-child { border-bottom: none; }
.stat-row .left { display: flex; align-items: center; gap: 9px; color: var(--text); font-weight: 500; }
.stat-row .left svg { width: 15px; height: 15px; color: var(--blue); }
.stat-row .val { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }

/* ── Validation box (maps to HTML .validation-box) ── */
.validation-box {
    background: var(--bg-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px !important;
    margin-top: 16px !important;
}
.validation-box .title {
    font-size: 12.5px; font-weight: 700; color: var(--text); margin: 0 0 10px;
}
.validation-row {
    display: flex; justify-content: space-between;
    font-size: 12.5px; padding: 5px 0; color: var(--muted);
}
.validation-row b { color: var(--text); font-weight: 700; }

/* ── Quick Guide (maps to HTML .quick-guide) ── */
.quick-guide { margin-top: 18px; }
.quick-guide .title {
    display: flex; align-items: center; gap: 7px;
    font-size: 13.5px; font-weight: 700; margin-bottom: 12px; color: var(--text);
}
.quick-guide .title svg { width: 15px; height: 15px; color: #f5a623; }
.qg-item {
    display: flex; gap: 10px; font-size: 13px; color: var(--muted);
    margin-bottom: 10px; line-height: 1.4;
}
.qg-num {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--blue-light); color: var(--blue-dark);
    font-size: 10.5px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}

/* ── Label overrides ── */
label, .gr-label, .label-wrap {
    color: var(--muted) !important;
}

/* ── Accordion overrides ── */
.gr-accordion {
    border-color: var(--border) !important;
}

/* ── Hero section ── */
.hero-section {
    position: relative; overflow: hidden;
    background: linear-gradient(180deg, var(--bg-soft) 0%, var(--bg) 100%);
    padding: 60px 32px 48px; transition: background 0.35s ease;
}
.hero-section::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(600px 300px at 85% 0%, rgba(124,92,255,0.14), transparent 60%),
               radial-gradient(500px 260px at 60% -10%, rgba(52,102,255,0.14), transparent 60%);
    pointer-events: none;
}
.eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 12.5px; font-weight: 700;
    color: var(--blue-dark); background: var(--blue-light);
    padding: 6px 12px; border-radius: 999px;
    margin-bottom: 16px; letter-spacing: 0.2px;
}
.eyebrow svg { width: 13px; height: 13px; }
.hero-deco { position: absolute; right: 60px; top: 30px; width: 280px; height: 220px; pointer-events: none; }
.hero-dots {
    position: absolute; right: 0; top: 0; width: 280px; height: 220px;
    background-image: radial-gradient(circle, var(--muted-soft) 1.3px, transparent 1.3px);
    background-size: 15px 15px; opacity: 0.35;
    -webkit-mask-image: radial-gradient(circle at 70% 30%, black 40%, transparent 75%);
    mask-image: radial-gradient(circle at 70% 30%, black 40%, transparent 75%);
}
.float-card {
    position: absolute; background: var(--card); border-radius: 16px;
    box-shadow: var(--shadow-lift);
    display: flex; align-items: center; justify-content: center;
    border: 1px solid var(--border-soft);
    animation: bob 5s ease-in-out infinite;
}
.float-card.search { width: 64px; height: 64px; left: 26px; top: 70px; transform: rotate(-4deg); animation-delay: 0s; --r: -4deg; }
.float-card.search svg { width: 26px; height: 26px; color: var(--blue); }
.float-card.chat { width: 76px; height: 58px; right: 6px; top: 14px; border-radius: 16px 16px 16px 4px; animation-delay: 0.6s; }
.float-card.chat svg { width: 24px; height: 24px; color: var(--violet); }
.float-card.play { width: 54px; height: 54px; right: 76px; bottom: 2px; border-radius: 14px; animation-delay: 1.2s; }
.float-card.play svg { width: 18px; height: 18px; color: var(--blue); }

/* ── Header / Navbar ── */
.adve-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 32px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(16px);
    transition: background 0.35s ease, border-color 0.35s ease;
    margin: -16px -16px 24px -16px;
}
.logo { display: flex; align-items: center; gap: 9px; font-weight: 700; font-size: 19px; letter-spacing: 0.2px; }
.logo .tri {
    width: 27px; height: 27px; border-radius: 8px;
    background: linear-gradient(135deg, var(--grad-a), var(--grad-b));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(52,102,255,0.35);
}
.logo .tri svg { width: 13px; height: 13px; fill: #fff; }
.logo span { color: var(--blue-dark); font-family: 'Space Grotesk', Inter, sans-serif !important; }
.main-nav { display: flex; align-items: center; gap: 30px; }
.main-nav a {
    font-size: 14.5px; font-weight: 600; color: var(--muted);
    padding: 6px 2px; position: relative;
    display: flex; align-items: center; gap: 5px;
    transition: color 0.2s ease; text-decoration: none;
}
.main-nav a:hover { color: var(--text); }
.main-nav a.active { color: var(--blue); }
.main-nav a.active::after {
    content: ""; position: absolute; left: 0; right: 0; bottom: -15px; height: 2px;
    background: linear-gradient(90deg, var(--grad-a), var(--grad-b)); border-radius: 2px;
}
.main-nav a svg { width: 13px; height: 13px; opacity: 0.7; }
.top-right { display: flex; align-items: center; gap: 12px; }
.icon-btn {
    width: 36px; height: 36px; border-radius: 9px; border: 1px solid var(--border);
    background: var(--card); display: flex; align-items: center; justify-content: center;
    transition: transform 0.2s ease, background 0.2s ease, border-color 0.35s ease;
    cursor: pointer;
}
.icon-btn:hover { transform: translateY(-1px); background: var(--bg-soft); }
.icon-btn svg { width: 16px; height: 16px; color: var(--text); }
.btn-header-primary {
    display: flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, var(--grad-a), var(--blue-dark));
    color: #fff; border: none; padding: 10px 18px; border-radius: 9px;
    font-weight: 700; font-size: 14px;
    box-shadow: 0 6px 16px rgba(52,102,255,0.32);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer; text-decoration: none;
}
.btn-header-primary:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(52,102,255,0.4); }
.btn-header-primary svg { width: 15px; height: 15px; }

/* ── Hide/show theme Toggle Icons ── */
#themeToggle svg.sun { display: block; }
#themeToggle svg.moon { display: none; }
html.dark #themeToggle svg.sun { display: none; }
html.dark #themeToggle svg.moon { display: block; }

/* ── Footer ── */
.adve-footer {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    padding: 16px 32px; border-top: 1px solid var(--border);
    background: var(--card);
    font-size: 13px; color: var(--muted);
    transition: background 0.35s ease, border-color 0.35s ease;
    margin-top: 24px;
}
.adve-footer .side { display: flex; align-items: center; gap: 22px; }
.adve-footer a {
    display: flex; align-items: center; gap: 6px;
    font-weight: 500; transition: color 0.2s ease;
    text-decoration: none; color: var(--muted);
}
.adve-footer a:hover { color: var(--blue); }
.adve-footer svg { width: 14px; height: 14px; }

/* ── Search button (maps to HTML .btn-search) ── */
.btn-search-style {
    display: flex; align-items: center; gap: 6px;
    background: var(--blue) !important; color: #fff !important; border: none !important;
    padding: 0 18px !important; border-radius: var(--radius-sm) !important;
    font-weight: 700 !important; font-size: 13.5px !important;
    transition: transform 0.15s ease, background 0.2s ease !important;
}
.btn-search-style:hover { transform: translateY(-1px) !important; background: var(--blue-dark) !important; }

:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
"""

def get_dynamic_stats():
    try:
        total = search_index.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        anchors = search_index.db.execute("SELECT COUNT(*) FROM frames WHERE is_anchor=1").fetchone()[0]
        trans = search_index.db.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        ocr_count = search_index.db.execute("SELECT COUNT(*) FROM ocr_text").fetchone()[0]
        
        cursor = search_index.db.execute("SELECT COUNT(DISTINCT video_path) FROM frames")
        videos_count = cursor.fetchone()[0]
    except Exception:
        total, anchors, trans, ocr_count, videos_count = 0, 0, 0, 0, 0
        
    return f"""
    <div class="stat-row">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/></svg>Total Embeddings</span>
        <span class="val">{total:,}</span>
    </div>
    <div class="stat-row">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 100 20 10 10 0 000-20z"/><path d="M12 6v6l4 2"/></svg>Anchor (CLIP)</span>
        <span class="val">{anchors:,}</span>
    </div>
    <div class="stat-row">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h4l3 8 4-16 3 8h4"/></svg>Delta (approx.)</span>
        <span class="val">{(total - anchors):,}</span>
    </div>
    <div class="stat-row">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h6"/></svg>Transcripts (Segments)</span>
        <span class="val">{trans:,}</span>
    </div>
    <div class="stat-row">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg>OCR Text Chunks</span>
        <span class="val">{ocr_count:,}</span>
    </div>
    <div class="stat-row">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M10 9l5 3-5 3z"/></svg>Videos Indexed</span>
        <span class="val">{videos_count:,}</span>
    </div>
    """

with gr.Blocks(title="ADVE Engine Portal", css=custom_css) as demo:
    # ── Navbar ──
    gr.HTML("""
    <header class="adve-header">
      <div class="logo">
        <div class="tri"><svg viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg></div>
        <span>ADVE</span>
      </div>
      <nav class="main-nav">
        <a href="#" class="active">Overview</a>
        <a href="#">Search</a>
        <a href="#">Chat</a>
        <a href="#">Docs</a>
        <a href="https://github.com/asmitha2025/ADVE" target="_blank">GitHub
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><path d="M15 3h6v6"/><path d="M10 14L21 3"/></svg>
        </a>
      </nav>
      <div class="top-right">
        <button class="icon-btn" id="themeToggle" aria-label="Toggle dark mode">
          <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
          <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>
        </button>
        <button class="btn-header-primary" id="heroIndexBtn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 00-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 012-3.95A12.88 12.88 0 0122 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 01-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>
          Start Indexing
        </button>
      </div>
    </header>
    """)

    # ── Hero Section ──
    gr.HTML("""
    <section class="hero-section">
      <div class="hero-inner">
        <span class="eyebrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 2l2.4 6.6L21 11l-6.6 2.4L12 20l-2.4-6.6L3 11l6.6-2.4z"/></svg>Up to 90% fewer vision-network calls</span>
        <h1>ADVE — <span class="accent">Semantic Video Search</span> &amp; Chatbot</h1>
        <p>Anchor-Delta Video Embedding (ADVE) reduces neural vision network calls by up to 90% via motion-adaptive keyframe processing for semantic scene search and video Q&amp;A.</p>
      </div>
      <div class="hero-deco">
        <div class="hero-dots"></div>
        <div class="float-card search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg></div>
        <div class="float-card chat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg></div>
        <div class="float-card play"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg></div>
      </div>
    </section>
    """)

    with gr.Row():
        # ── COLUMN 1: Ingestion & Stats (scale=4) ──
        with gr.Column(scale=4):
            # Ingestion Card
            with gr.Column(elem_classes="panel-card"):
                gr.HTML("""<div class='pane-title'><span class='step-num'>1</span><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' style='display:inline;'><path d='M20 17.58A5 5 0 0018 8h-1.26A8 8 0 104 16.25'/><path d='M12 12v9'/><path d='M9 18l3 3 3-3'/></svg>Video Ingestion & Indexing</div>""")
                with gr.Tabs(elem_classes="tabs"):
                    with gr.TabItem("YouTube URL"):
                        yt_url = gr.Textbox(show_label=False, placeholder="https://www.youtube.com/watch?v=...", container=False)
                        with gr.Row(elem_classes="option-card-row"):
                            yt_adaptive = gr.Checkbox(label="Adaptive FPS", value=True, elem_classes=["checkbox-card", "chk-adaptive"], container=False)
                            yt_index_audio = gr.Checkbox(label="Whisper", value=False, elem_classes=["checkbox-card", "chk-whisper"], container=False)
                            yt_index_ocr = gr.Checkbox(label="EasyOCR", value=False, elem_classes=["checkbox-card", "chk-easyocr"], container=False)
                        with gr.Accordion("Advanced Ingestion Settings", open=False):
                            yt_fps = gr.Slider(label="Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=2.0, step=0.1)
                        yt_index_btn = gr.Button("🚀 Index Video", variant="primary", elem_classes="primary-btn")
                        yt_status = gr.Textbox(label="Indexing Output Status", interactive=False, placeholder="Waiting to index...", elem_classes="status-box")
                        
                    with gr.TabItem("Local Upload"):
                        local_file = gr.File(label="Upload Video File", file_types=["video"])
                        with gr.Row(elem_classes="option-card-row"):
                            local_adaptive = gr.Checkbox(label="Adaptive FPS", value=True, elem_classes=["checkbox-card", "chk-adaptive"], container=False)
                            local_index_audio = gr.Checkbox(label="Whisper", value=False, elem_classes=["checkbox-card", "chk-whisper"], container=False)
                            local_index_ocr = gr.Checkbox(label="EasyOCR", value=False, elem_classes=["checkbox-card", "chk-easyocr"], container=False)
                        with gr.Accordion("Advanced Ingestion Settings", open=False):
                            local_fps = gr.Slider(label="Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=2.0, step=0.1)
                        local_index_btn = gr.Button("🚀 Index Video", variant="primary", elem_classes="primary-btn")
                        local_status = gr.Textbox(label="Indexing Output Status", interactive=False, placeholder="Waiting to index...", elem_classes="status-box")

            # Stats Card
            with gr.Column(elem_classes="panel-card"):
                gr.HTML("""<div class='pane-title'><span class='step-num'>5</span><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' style='display:inline;'><path d='M12 20h9M3 20h4M3 12h18M3 4h18'/></svg>Deployed Index Statistics</div>""")
                stats_html = gr.HTML(get_dynamic_stats())
                
                gr.HTML("""
                <div class="validation-box">
                    <p class="title">Validation Reference</p>
                    <div class="validation-row"><span>Synthetic:</span><b>96.7% savings · 0.948 cosine sim</b></div>
                    <div class="validation-row"><span>MOT17:</span><b>60.3% savings · 0.992 cosine sim</b></div>
                    <div class="validation-row"><span>GPU VRAM:</span><b>330 MB (vs 950 MB baseline)</b></div>
                </div>
                """)
                
                gr.HTML("""
                <div class="quick-guide">
                    <div class="title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px;color:#f5a623;"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 00-4 12.7c.6.5 1 1.2 1 2.3h6c0-1.1.4-1.8 1-2.3A7 7 0 0012 2z"/></svg>
                        Quick Guide
                    </div>
                    <div class="qg-item"><span class="qg-num">1</span>Upload or paste a YouTube URL and index the video.</div>
                    <div class="qg-item"><span class="qg-num">2</span>Search by text or ask a question in the chatbot.</div>
                </div>
                """)

        # ── COLUMN 2: Search, Results & Chatbot (scale=6) ──
        with gr.Column(scale=6):
            # Search Card
            with gr.Column(elem_classes="panel-card"):
                gr.HTML("""<div class='pane-title'><span class='step-num'>2</span><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' style='display:inline;'><circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/></svg>Semantic Scene Search</div>""")
                gr.HTML("<p style='font-size: 13px; color: #71717a; margin-top: -8px; margin-bottom: 12px;'>Describe the scene you're looking for</p>")
                with gr.Row():
                    search_query = gr.Textbox(placeholder="E.g., a person typing on a laptop in a cafe", container=False, scale=4)
                    search_btn = gr.Button("🔍 Search", variant="primary", elem_classes="primary-btn", scale=1)
                
                # Hidden settings accordion to keep UI clean
                with gr.Accordion("Search Settings", open=False):
                    clip_duration = gr.Slider(label="Clip Duration (seconds)", minimum=3.0, maximum=60.0, value=10.0, step=1.0)
                    min_similarity = gr.Slider(label="Min Similarity Gate", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    use_dynamic_duration = gr.Checkbox(label="Auto-Detect Scene Duration", value=False)
                    anchor_only = gr.Checkbox(label="Search Anchor Frames Only", value=False)
                    
                search_metrics = gr.Markdown("No query submitted yet.")

            # Match Results Card
            with gr.Column(elem_classes="panel-card"):
                gr.HTML("""<div class='pane-title'><span class='step-num'>3</span><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' style='display:inline;'><circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/></svg>Search & Match Results</div>""")
                gr.HTML("<p style='font-size: 13px; color: #71717a; margin-top: -8px; margin-bottom: 16px;'>Top matching keyframes from your indexed videos.</p>")
                
                with gr.Row():
                    with gr.Column(elem_classes="match-card", scale=1):
                        m1_img = gr.Image(show_label=False, interactive=False, height=180)
                        m1_meta = gr.HTML(empty_meta)
                    with gr.Column(elem_classes="match-card", scale=1):
                        m2_img = gr.Image(show_label=False, interactive=False, height=180)
                        m2_meta = gr.HTML(empty_meta)
                    with gr.Column(elem_classes="match-card", scale=1):
                        m3_img = gr.Image(show_label=False, interactive=False, height=180)
                        m3_meta = gr.HTML(empty_meta)
                        
                # Video playback players (shown dynamically underneath when match is selected/clicked)
                with gr.Accordion("Clips Playback", open=False):
                    clip_player_1 = gr.Video(label="Match 1 Clip", visible=False)
                    clip_player_2 = gr.Video(label="Match 2 Clip", visible=False)
                    clip_player_3 = gr.Video(label="Match 3 Clip", visible=False)

            # Chatbot Card
            with gr.Column(elem_classes="panel-card"):
                gr.HTML("""<div class='pane-title'><span class='step-num'>4</span><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' style='display:inline;'><path d='M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z'/></svg>Conversational Video Chatbot</div>""")
                chatbot = gr.Chatbot(show_label=False, height=220)
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="Ask a question about the video...", container=False, scale=4)
                    chat_submit = gr.Button("✈️ Send", variant="primary", elem_classes="primary-btn", scale=1)
                with gr.Row():
                    gr.HTML("<span style='font-size: 12px; color: #52525b; padding-top: 6px;'>⚡ Powered by vision-language models</span>")
                    chat_clear_btn = gr.Button("🗑️ Clear Chat", elem_classes="secondary-btn", size="sm", scale=1)

    # ── Footer ──
    gr.HTML("""
    <footer class="adve-footer">
      <div class="side">
        <a href="#"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.8 4.6a5.5 5.5 0 00-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 00-7.8 7.8l1 1L12 21l7.8-7.8 1-1a5.5 5.5 0 000-7.8z"/></svg>Built by Asmitha</a>
        <a href="https://github.com/asmitha2025/ADVE" target="_blank"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5a12 12 0 00-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.8.1-.8.1-.8 1.2.1 1.9 1.3 1.9 1.3 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.5-1.3-5.5-6a4.6 4.6 0 011.2-3.2 4.3 4.3 0 010-3.2s1-.3 3.3 1.2a11.5 11.5 0 016 0c2.3-1.5 3.3-1.2 3.3-1.2a4.3 4.3 0 010 3.2 4.6 4.6 0 011.2 3.2c0 4.7-2.8 5.7-5.5 6 .4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0012 .5z"/></svg>GitHub</a>
      </div>
      <div class="side">
        <a href="#"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 18l6-6-6-6"/><path d="M8 6l-6 6 6 6"/></svg>Use via API</a>
        <a href="#"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h7l-1 8 11-14h-7l1-6z"/></svg>Built with Gradio</a>
        <button class="icon-btn" id="themeToggle" style="border:none;background:none;padding:0;display:inline-flex;align-items:center;" aria-label="Toggle dark mode">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1 1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z"/></svg>
        </button>
      </div>
    </footer>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var themeToggle = document.getElementById("themeToggle");
        if (themeToggle) {
            themeToggle.addEventListener("click", function() {
                var html = document.documentElement;
                html.classList.toggle("dark");
            });
        }
    });
    </script>
    """)

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
        outputs=[search_metrics, m1_img, m1_meta, m2_img, m2_meta, m3_img, m3_meta, clip_player_1, clip_player_2, clip_player_3]
    )
    search_query.submit(
        fn=search_and_retrieve,
        inputs=[search_query, clip_duration, use_dynamic_duration, anchor_only, min_similarity],
        outputs=[search_metrics, m1_img, m1_meta, m2_img, m2_meta, m3_img, m3_meta, clip_player_1, clip_player_2, clip_player_3]
    )
    
    # Chatbot submit bindings (Submit on Send or enter key)
    chat_submit.click(
        fn=chatbot_chat_flow,
        inputs=[chat_input, chatbot, clip_duration, use_dynamic_duration, min_similarity],
        outputs=[chat_input, chatbot, m1_img, m1_meta, m2_img, m2_meta, m3_img, m3_meta, clip_player_1, clip_player_2, clip_player_3]
    )
    chat_input.submit(
        fn=chatbot_chat_flow,
        inputs=[chat_input, chatbot, clip_duration, use_dynamic_duration, min_similarity],
        outputs=[chat_input, chatbot, m1_img, m1_meta, m2_img, m2_meta, m3_img, m3_meta, clip_player_1, clip_player_2, clip_player_3]
    )
    chat_clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, chat_input, m1_img, m1_meta, m2_img, m2_meta, m3_img, m3_meta]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch demo with a public shareable URL")
    args = parser.parse_args()
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share,
                theme=gr.themes.Base(
                    primary_hue=gr.themes.colors.purple,
                    secondary_hue=gr.themes.colors.blue,
                    neutral_hue=gr.themes.colors.zinc,
                    font=gr.themes.GoogleFont("Inter"),
                ).set(
                    body_background_fill="#09090b",
                    body_background_fill_dark="#09090b",
                    block_background_fill="#111113",
                    block_background_fill_dark="#111113",
                    block_border_color="rgba(255,255,255,0.06)",
                    block_border_color_dark="rgba(255,255,255,0.06)",
                    input_background_fill="#09090b",
                    input_background_fill_dark="#09090b",
                    input_border_color="rgba(255,255,255,0.08)",
                    input_border_color_dark="rgba(255,255,255,0.08)",
                ), css=custom_css)
